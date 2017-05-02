
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define WORD uint32_t
#define WORD_SIZE 32
#define BLOCK_SIZE 256

__global__ void gpu_dense_bit_recall_kernel(size_t size,
                                            bool *state, //WORD *state,
                                            float *thresholds,
                                            float *weights,
                                            bool *stable) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    float value = 0.0f;
    for (size_t k = 0; k < size; ++k) {
      if (state[k])
        value += weights[i * size + k];
      else
        value -= weights[i * size + k];
    }

    bool update = value > thresholds[i];
    if (update != state[i]) {
      *stable = false;
      state[i] = update;
    }
  }
}

GPUDenseBitHopfieldNetwork::GPUDenseBitHopfieldNetwork(const std::vector<float> &thresholds,
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

GPUDenseBitHopfieldNetwork::~GPUDenseBitHopfieldNetwork() {
  cudaFree(thresholdsDev);
  cudaFree(weightsDev);
}

vector<bool> GPUDenseBitHopfieldNetwork::evaluate(const vector<bool> &data) {
  bool stable;
  bool dataArray[size];

  bool *stateDev;
  bool *stableDev;
  
  unsigned numThreads = 256;
  unsigned numBlocks = size / numThreads;

  if (size % numThreads) numBlocks++;

  cudaCheck(cudaMalloc((void**) &stateDev, sizeof(bool) * size));
  cudaCheck(cudaMalloc((void**) &stableDev, sizeof(bool)));

  copy(data.begin(), data.end(), dataArray);
  cudaCheck(cudaMemcpy(stateDev, dataArray, size * sizeof(bool),
                       cudaMemcpyHostToDevice));

  do {
    stable = true;
    cudaCheck(cudaMemcpy(stableDev, &stable, sizeof(bool),
                         cudaMemcpyHostToDevice));

    gpu_dense_bit_recall_kernel<<< numBlocks, numThreads >>>
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

