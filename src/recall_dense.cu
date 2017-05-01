
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

__global__ void gpu_dense_recall_kernel(size_t size,
                                        bool * state,
                                        float * thresholds,
                                        float * weights,
                                        bool * stable) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  float value = 0.0f;
  bool update;

  bool stableT = true;

  if (i < size) {
    for (size_t k = 0; k < size; ++k) {
      if (state[k])
        value += weights[i * size + k];
      else
        value -= weights[i * size + k];
    }

    update = value > thresholds[i];
    stableT &= update == state[i];

    state[i] = update;

    //TODO: use reduction to find stable
    atomicAnd((int *) stable, (int) stableT);
  }
}

vector<bool> GPUDenseRecall::recall(const vector<bool> &data,
                                    const vector<float> &thresholds,
                                    const vector<vector<float> > &weights) {
  size_t size = data.size();
  bool stable;

  bool dataArray[size];
  float thresholdArray[size];
  float (*weightArray)[size] = (float(*)[size])new float[size * size];

  bool * stateDev;
  float * thresholdDev;
  float * weightDev;
  bool * stableDev;
  unsigned numThreads = 256;
  unsigned numBlocks = size / numThreads;

  if (size % numThreads) numBlocks++;

  for (size_t i = 0; i < size; ++i) {
    dataArray[i] = data[i];
    thresholdArray[i] = thresholds[i];

    for (size_t j = 0; j < size; ++j) {
      weightArray[i][j] = weights[i][j];
    }
  }

  assert(cudaMalloc((void**) &stateDev, sizeof(bool) * size) == cudaSuccess);
  assert(cudaMalloc((void**) &thresholdDev, sizeof(float) * size)
         == cudaSuccess);
  assert(cudaMalloc((void**) &weightDev, sizeof(float) * size * size)
         == cudaSuccess);
  assert(cudaMalloc((void**) &stableDev, sizeof(bool)) == cudaSuccess);

  assert(cudaMemcpy(stateDev, dataArray, size * sizeof(bool),
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(thresholdDev, thresholdArray, size * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(weightDev, weightArray, size * size * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess);

  do {
    stable = true;
    assert(cudaMemcpy(stableDev, &stable, sizeof(bool),
                      cudaMemcpyHostToDevice) == cudaSuccess);

    gpu_dense_recall_kernel<<< numBlocks, numThreads >>>
      (size, stateDev, thresholdDev, weightDev, stableDev);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    assert(cudaMemcpy(&stable, stableDev, sizeof(bool),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
  } while (!stable);

  assert(cudaMemcpy(dataArray, stateDev, size * sizeof(bool),
                    cudaMemcpyDeviceToHost) == cudaSuccess);

  assert(cudaDeviceSynchronize() == cudaSuccess);
  
  vector<bool> state(size);
  for (size_t i = 0; i < size; ++i) {
    state[i] = dataArray[i];
  }

  delete[] weightArray;

  cudaFree(stateDev);
  cudaFree(thresholdDev);
  cudaFree(weightDev);
  cudaFree(stableDev);

  return state;
}

