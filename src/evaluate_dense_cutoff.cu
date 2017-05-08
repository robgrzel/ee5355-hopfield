
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 32
#define REDUCTION_BLOCK_SIZE 1024

__global__ void gpu_dense_cutoff_recall_kernel(size_t size,
					       bool * state,
					       float * thresholds,
					       float * weights,
					       unsigned * changed) {
  size_t i = blockIdx.x;

  // Compute values in a strided pattern
  float value = 0.0f;
  for (size_t k = threadIdx.x; k < size; k += BLOCK_SIZE) {
    if (state[k])
      value += weights[i * size + k];
    else
      value -= weights[i * size + k];
  }

  __shared__ float values[BLOCK_SIZE];
  values[threadIdx.x] = value;
  __syncthreads();

  // Perform reduction
  for (uint8_t stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
    if (((threadIdx.x + 1) & ((stride << 1) - 1)) == 0) {
      values[threadIdx.x] += values[threadIdx.x - stride];
    }
    __syncthreads();
  }

  value = values[BLOCK_SIZE - 1];
  __syncthreads();
  
  // Perform update
  bool update = value > thresholds[i];
  changed[i] = update != state[i];
  state[i] = update;
}

__global__ void reduction(unsigned *data, unsigned *result, size_t n) {
  __shared__ unsigned partialSum[REDUCTION_BLOCK_SIZE * 2];
  
  size_t tx = threadIdx.x;
  size_t bx = blockDim.x * blockIdx.x * 2;
  partialSum[tx] = tx + bx < n? data[tx + bx] : 0;
  partialSum[tx + blockDim.x] = tx + bx + blockDim.x < n? data[tx + bx + blockDim.x] : 0;
  
  for (size_t i = blockDim.x; i > 0; i >>= 1) {
    __syncthreads();
    if (i > tx)
      partialSum[tx] += partialSum[tx + i];
  }

  if (threadIdx.x == 0)
    result[blockIdx.x] = partialSum[0];
}

GPUDenseCutoffHopfieldNetwork::GPUDenseCutoffHopfieldNetwork(const std::vector<float> &thresholds,
							     const std::vector<std::vector<float>> &weights) :
  HopfieldNetwork(thresholds, weights),
  cpuNet(thresholds, weights) {
  cudaCheck(cudaMalloc((void**) &thresholdsDev, sizeof(float) * size));
  cudaCheck(cudaMalloc((void**) &weightsDev, sizeof(float) * size * size));

  float (*weightArray)[size] = (float(*)[size])new float[size * size];
  for (size_t i = 0; i < size; ++i) {
    copy(weights[i].begin(), weights[i].end(), weightArray[i]);
  }

  cudaCheck(cudaMemcpy(thresholdsDev, thresholds.data(), size * sizeof(float),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(weightsDev, weightArray, size * size * sizeof(float),
                       cudaMemcpyHostToDevice));
  
  delete[] weightArray; 
}

GPUDenseCutoffHopfieldNetwork::~GPUDenseCutoffHopfieldNetwork() {
  cudaFree(thresholdsDev);
  cudaFree(weightsDev);
}

vector<bool> GPUDenseCutoffHopfieldNetwork::evaluate(const vector<bool> &data) {
  unsigned changedArray[size];
  size_t numChanged;
  size_t prevNumChanged;
  
  bool dataArray[size];

  bool *stateDev;
  unsigned *changedDev;
  unsigned *changedResDev;

  cudaCheck(cudaMalloc((void**) &stateDev, sizeof(unsigned) * size));
  cudaCheck(cudaMalloc((void**) &changedDev, sizeof(unsigned) * size));
  cudaCheck(cudaMalloc((void**) &changedResDev, sizeof(unsigned) * ((size - 1) / (REDUCTION_BLOCK_SIZE * 2) + 1)));

  copy(data.begin(), data.end(), dataArray);
  cudaCheck(cudaMemcpy(stateDev, dataArray, size * sizeof(bool),
                       cudaMemcpyHostToDevice));

  do {
    gpu_dense_cutoff_recall_kernel<<< size, BLOCK_SIZE >>>
      (size, stateDev, thresholdsDev, weightsDev, changedDev);
    cudaCheck(cudaDeviceSynchronize());

    size_t numReductionBlocks = (size - 1) / (REDUCTION_BLOCK_SIZE * 2) + 1;
    reduction<<<numReductionBlocks, REDUCTION_BLOCK_SIZE>>>(changedDev, changedResDev, size);
    
    cudaCheck(cudaMemcpy(changedArray, changedResDev, numReductionBlocks * sizeof(unsigned),
                         cudaMemcpyDeviceToHost));

    prevNumChanged = numChanged;
    numChanged = 0;
    for (size_t i = 0; i < numReductionBlocks; i++) {
      numChanged += changedArray[i];
    }
  } while (numChanged > 0 && numChanged < prevNumChanged);

  cudaCheck(cudaMemcpy(dataArray, stateDev, size * sizeof(bool),
                       cudaMemcpyDeviceToHost));

  cudaCheck(cudaDeviceSynchronize());
  
  cudaFree(stateDev);
  cudaFree(changedDev);
  cudaFree(changedResDev);

  vector<bool> state(dataArray, dataArray + size);
  if (numChanged)
    return cpuNet.evaluate(state);
  else
    return state;
}
