
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 32

__global__ void gpu_dense_block_coarse_recall_kernel(size_t size,
                                                     size_t startIdx,
                                                     bool * state1,
                                                     bool * state2,
                                                     float * thresholds,
                                                     float * weights,
                                                     bool * stable) {
  size_t i = blockIdx.x + startIdx;
  if (i < size) {
  //for (size_t i = blockIdx.x; i < size; i += gridDim.x) {
    // Compute values in a strided pattern
    float value = 0.0f;
    for (size_t k = threadIdx.x; k < size; k += BLOCK_SIZE) {
      if (state1[k])
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
    if (update != state1[i]) {
      *stable = false;
      // if (threadIdx.x == 0) {
      //   printf("Wrote %2lu: %c -> %c\n", i, state1[i]? '1' : '0', update? '1' : '0');
      // }
    }
    state2[blockIdx.x] = update;
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

  bool *stateDev1, *stateDev2;
  bool *stableDev;

  cudaCheck(cudaMalloc((void**) &stateDev1, sizeof(bool) * ((size - 1) / parallel + 1) * parallel));
  cudaCheck(cudaMalloc((void**) &stateDev2, sizeof(bool) * parallel));
  cudaCheck(cudaMalloc((void**) &stableDev, sizeof(bool)));

  copy(data.begin(), data.end(), dataArray);
  cudaCheck(cudaMemcpy(stateDev1, dataArray, size * sizeof(bool),
                       cudaMemcpyHostToDevice));

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  do {
    stable = true;
    cudaCheck(cudaMemcpy(stableDev, &stable, sizeof(bool),
                         cudaMemcpyHostToDevice));

    for (size_t startIdx = 0; startIdx < size; startIdx += parallel) {
      gpu_dense_block_coarse_recall_kernel<<< parallel, BLOCK_SIZE >>>
        (size, startIdx, stateDev1, stateDev2, thresholdsDev, weightsDev, stableDev);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaMemcpy(stateDev1 + startIdx, stateDev2, parallel,
                           cudaMemcpyDeviceToDevice));
      cudaCheck(cudaDeviceSynchronize());
    }

    cudaCheck(cudaMemcpy(&stable, stableDev, sizeof(bool),
                         cudaMemcpyDeviceToHost));
  } while (!stable);

  cudaCheck(cudaMemcpy(dataArray, stateDev1, size * sizeof(bool),
                       cudaMemcpyDeviceToHost));

  cudaCheck(cudaDeviceSynchronize());
  
  vector<bool> state(dataArray, dataArray + size);

  cudaFree(stateDev1);
  cudaFree(stateDev2);
  cudaFree(stableDev);

  return state;
}

