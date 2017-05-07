
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define STABLE_CUTOFF 0.95
#define BLOCK_SIZE 32

__global__ void gpu_dense_cutoff_recall_kernel(size_t size,
					      bool * state,
					      float * thresholds,
					      float * weights,
					      bool * stable) {
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
  if (update != state[i]) {
    *stable = false;
  }
  state[i] = update;
}

GPUDenseCutoffHopfieldNetwork::GPUDenseCutoffHopfieldNetwork(const std::vector<float> &thresholds,
							     const std::vector<std::vector<float>> &weights) :
  HopfieldNetwork(thresholds, weights),
  thresholds(thresholds),
  weights(weights) {
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
  bool stable;
  bool stableArray[size];
  bool dataArray[size];

  bool *stateDev;
  bool *stableDev;

  cudaCheck(cudaMalloc((void**) &stateDev, sizeof(bool) * size));
  cudaCheck(cudaMalloc((void**) &stableDev, sizeof(bool) * size));

  copy(data.begin(), data.end(), dataArray);
  cudaCheck(cudaMemcpy(stateDev, dataArray, size * sizeof(bool),
                       cudaMemcpyHostToDevice));

  do {
    cudaCheck(cudaMemcpy(stableDev, &stable, sizeof(bool),
                         cudaMemcpyHostToDevice));

    gpu_dense_cutoff_recall_kernel<<< size, BLOCK_SIZE >>>
      (size, stateDev, thresholdsDev, weightsDev, stableDev);
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(stableArray, stableDev, size * sizeof(bool),
                         cudaMemcpyDeviceToHost));


    size_t numStable = 0;
#pragma omp parallel for reduction(+:numStable)
    for (size_t i = 0; i < size; i++) {
      if (stableArray[i])
	numStable += 1;
    }

    stable = numStable > size * STABLE_CUTOFF;

  } while (!stable);

  cudaCheck(cudaMemcpy(dataArray, stateDev, size * sizeof(bool),
                       cudaMemcpyDeviceToHost));

  cudaCheck(cudaDeviceSynchronize());
  
  cudaFree(stateDev);
  cudaFree(stableDev);

  vector<bool> state(dataArray, dataArray + size);
  CPUDenseHopfieldNetwork cpuNet(thresholds, weights);
  return cpuNet.evaluate(state);
}
