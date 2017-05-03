
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define WORD uint32_t
#define WORD_SIZE 32
#define BLOCK_SIZE 32

__global__ void gpu_dense_bit_recall_kernel(size_t size,
                                            size_t numWords,
                                            WORD *states,
                                            float *thresholds,
                                            float *weights,
                                            bool *stable) {
  size_t i = blockIdx.x;
  size_t wordIdx = i / WORD_SIZE;
  size_t bitIdx = i % WORD_SIZE;

  // Compute values in a strided pattern
  float value = 0.0f;
  for (unsigned j = 0; j < numWords; j++) {
    WORD s = states[j];
    for (size_t k = threadIdx.x; k < size; k += BLOCK_SIZE) {
      size_t idx = j * WORD_SIZE + k;
      if (idx < size) {
	if ((s >> k) & 1)
	  value += weights[i * size + idx];
	else
	  value -= weights[i * size + idx];
      }
    }
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
  bool newState = value > thresholds[i];
  bool oldState = (states[wordIdx] >> bitIdx) & 1;

  if (newState != oldState) {
    if (threadIdx.x == 0)
      atomicXor(states + wordIdx, 1 << bitIdx);
    *stable = false;
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
  if (size % WORD_SIZE) numWords++;
  
  bool stable;
  WORD dataArray[numWords];

  WORD *stateDev;
  bool *stableDev;

  cudaCheck(cudaMalloc((void**) &stateDev, sizeof(WORD) * numWords));
  cudaCheck(cudaMalloc((void**) &stableDev, sizeof(bool)));

  //cout << endl;
  //cout << endl;
  for (size_t i = 0; i < numWords; i++) {
    WORD s = 0;
    for (size_t j = 0; j < WORD_SIZE; j++) {
      size_t idx = i * WORD_SIZE + j;
      if (idx < size) {
        s |= (data[idx] << j);
	cout << data[idx];
      }
    }
    //cout << " ";
    dataArray[i] = s;
  }
  //cout << endl;
  
  cudaCheck(cudaMemcpy(stateDev, dataArray, numWords * sizeof(WORD),
                       cudaMemcpyHostToDevice));

  do {
    stable = true;
    cudaCheck(cudaMemcpy(stableDev, &stable, sizeof(bool),
                         cudaMemcpyHostToDevice));

    gpu_dense_bit_recall_kernel<<< size, BLOCK_SIZE >>>
      (size, numWords, stateDev, thresholdsDev, weightsDev, stableDev);
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(&stable, stableDev, sizeof(bool),
                         cudaMemcpyDeviceToHost));
  } while (!stable);

  cudaCheck(cudaMemcpy(dataArray, stateDev, numWords * sizeof(WORD),
                       cudaMemcpyDeviceToHost));

  cudaCheck(cudaDeviceSynchronize());
  
  vector<bool> state(size, 0);
  for (size_t i = 0; i < numWords; i++) {
    for (size_t j = 0; j < WORD_SIZE; j++) {
      size_t idx = i * WORD_SIZE + j;
      if (idx < size) {
        state[idx] = (dataArray[i] >> j) & 1;
	cout << state[idx];
      }
    }
    //cout << " ";
  }
  //cout << endl;

  cudaFree(stateDev);
  cudaFree(stableDev);

  return state;
}

