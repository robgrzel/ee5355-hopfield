
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
                                            size_t numWords,
                                            WORD *states,
                                            float *thresholds,
                                            float *weights,
                                            bool *stable) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    float value = 0.0f;
    for (unsigned j = 0; j < numWords; j++) {
      WORD s = states[j];
      for (size_t k = 0; k < WORD_SIZE; ++k) {
        size_t idx = j * WORD_SIZE + k;
        if (idx < size) {
          if (s & (1 << k))
            value += weights[i * size + idx];
          else
            value -= weights[i * size + idx];
        }
      }
    }

    bool newState = value > thresholds[i];
    bool oldState = states[i / WORD_SIZE] & (1 << (i % WORD_SIZE)) != 0;
    if (newState != oldState) {
      *stable = false;
    }
    // Set each bit of according to newState in each thread of the warp
    WORD newStates = __ballot(newState);
    if (threadIdx.x % WORD_SIZE == 0)
      states[i / WORD_SIZE] = newStates;
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
  size_t numWords = size / WORD_SIZE;
  if (numWords % WORD_SIZE) numWords++;
  
  bool stable;
  WORD dataArray[numWords];

  WORD *stateDev;
  bool *stableDev;
  
  unsigned numThreads = 256;
  unsigned numBlocks = size / numThreads;

  if (size % numThreads) numBlocks++;

  cudaCheck(cudaMalloc((void**) &stateDev, sizeof(WORD) * numWords));
  cudaCheck(cudaMalloc((void**) &stableDev, sizeof(bool)));

  for (size_t i = 0; i < numWords; i++) {
    WORD s = 0;
    for (size_t j = 0; j < WORD_SIZE; j++) {
      size_t idx = i * WORD_SIZE + j;
      if (idx < size) {
        s |= data[idx] & (1 << j);
      }
    }
    dataArray[i] = s;
  }
  cudaCheck(cudaMemcpy(stateDev, dataArray, numWords * sizeof(WORD),
                       cudaMemcpyHostToDevice));

  do {
    stable = true;
    cudaCheck(cudaMemcpy(stableDev, &stable, sizeof(bool),
                         cudaMemcpyHostToDevice));

    gpu_dense_bit_recall_kernel<<< numBlocks, numThreads >>>
      (size, numWords, stateDev, thresholdsDev, weightsDev, stableDev);
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

