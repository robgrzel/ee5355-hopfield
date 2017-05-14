
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 32

__global__ void gpu_dense_block_coarse_recall_kernel(size_t size,
                                                     size_t startIdx,
                                                     bool * state,
                                                     float * thresholds,
                                                     float * weights,
                                                     bool * stable) {
  size_t i = blockIdx.x + startIdx;
  if (i < size) {
  //for (size_t i = blockIdx.x; i < size; i += gridDim.x) {
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
    if (update != state[i]) {
      *stable = false;
    }
    state[i] = update;
  }
}

GPUDenseBlockCoarseHopfieldNetwork::GPUDenseBlockCoarseHopfieldNetwork(const std::vector<float> &thresholds,
                                                                       const std::vector<std::vector<float>> &weights,
                                                                       size_t parallel) :
  HopfieldNetwork(thresholds, weights),
  parallel(parallel) {
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

GPUDenseBlockCoarseHopfieldNetwork::~GPUDenseBlockCoarseHopfieldNetwork() {
  cudaFree(thresholdsDev);
  cudaFree(weightsDev);
}

vector<bool> GPUDenseBlockCoarseHopfieldNetwork::evaluate(const vector<bool> &data) {
  bool stable;
  bool dataArray[size];

  bool *stateDev;
  bool *stableDev;

  cudaCheck(cudaMalloc((void**) &stateDev, sizeof(bool) * size));
  cudaCheck(cudaMalloc((void**) &stableDev, sizeof(bool)));

  copy(data.begin(), data.end(), dataArray);
  cudaCheck(cudaMemcpy(stateDev, dataArray, size * sizeof(bool),
                       cudaMemcpyHostToDevice));

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  do {
    stable = true;
    cudaCheck(cudaMemcpy(stableDev, &stable, sizeof(bool),
                         cudaMemcpyHostToDevice));

    for (size_t startIdx = 0; startIdx < size; startIdx += parallel) {
      gpu_dense_block_coarse_recall_kernel<<< parallel, BLOCK_SIZE >>>
        (size, startIdx, stateDev, thresholdsDev, weightsDev, stableDev);
      cudaCheck(cudaDeviceSynchronize());
    }
    //gpu_dense_block_coarse_recall_kernel<<< parallel, BLOCK_SIZE >>>
    //  (size, stateDev, thresholdsDev, weightsDev, stableDev);

    cudaCheck(cudaMemcpy(&stable, stableDev, sizeof(bool),
                         cudaMemcpyDeviceToHost));
  } while (!stable);

  cudaCheck(cudaMemcpy(dataArray, stateDev, size * sizeof(bool),
                       cudaMemcpyDeviceToHost));

  cudaCheck(cudaDeviceSynchronize());
  
  vector<bool> state(dataArray, dataArray + size);

  cudaFree(stateDev);
  cudaFree(stableDev);

  return state;
}

