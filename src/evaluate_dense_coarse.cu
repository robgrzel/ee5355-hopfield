
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define COARSEN 2
#define BLOCK_SIZE 32

__global__ void gpu_dense_coarse_recall_kernel(size_t size,
					       bool * state,
					       float * thresholds,
					       float * weights,
					       bool * stable) {
  extern __shared__ float rowWeights[];

  size_t i = blockIdx.x;

  for (size_t j = threadIdx.x; j < size; j += BLOCK_SIZE) {
    rowWeights[j] = weights[i * size + j];
  }
  __syncthreads();

  __shared__ float values[BLOCK_SIZE];

  for (unsigned j = 0; j < COARSEN; j++) {
    // Compute values in a strided pattern
    float value = 0.0f;
    for (size_t k = threadIdx.x; k < size; k += BLOCK_SIZE) {
      if (state[k])
	value += rowWeights[k];
      else
	value -= rowWeights[k];
    }

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
    if (j == COARSEN - 1 && update != state[i])
      *stable = false;
    state[i] = update;

    //__threadfence();
  }
}

GPUDenseCoarseHopfieldNetwork::GPUDenseCoarseHopfieldNetwork(const std::vector<float> &thresholds,
							     const std::vector<std::vector<float>> &weights) :
  HopfieldNetwork(thresholds, weights) {
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

GPUDenseCoarseHopfieldNetwork::~GPUDenseCoarseHopfieldNetwork() {
  cudaFree(thresholdsDev);
  cudaFree(weightsDev);
}

vector<bool> GPUDenseCoarseHopfieldNetwork::evaluate(const vector<bool> &data) {
  bool stable;
  bool dataArray[size];

  bool *stateDev;
  bool *stableDev;

  cudaCheck(cudaMalloc((void**) &stateDev, sizeof(bool) * size));
  cudaCheck(cudaMalloc((void**) &stableDev, sizeof(bool)));

  copy(data.begin(), data.end(), dataArray);
  cudaCheck(cudaMemcpy(stateDev, dataArray, size * sizeof(bool),
                       cudaMemcpyHostToDevice));

  do {
    stable = true;
    cudaCheck(cudaMemcpy(stableDev, &stable, sizeof(bool),
                         cudaMemcpyHostToDevice));

    gpu_dense_coarse_recall_kernel<<< size, BLOCK_SIZE, size * sizeof(float) >>>
      (size, stateDev, thresholdsDev, weightsDev, stableDev);
    cudaCheck(cudaDeviceSynchronize());

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

