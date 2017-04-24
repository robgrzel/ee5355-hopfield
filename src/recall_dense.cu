
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
  float value = 0;
  bool update;

  bool stableT = true;

  if (i < size) {
    for (size_t k = 0; k < size; ++k) {
      if (state[k])
        value += weights[i][k];
      else
        value -= weights[i][k];
      update = value > thresholds[i];
      stableT &= update == state[i];
    }

    state[i] = update;

    //TODO: use reduction to find stable
    atomicAnd(stable, stableT);
  }

}

vector<bool> GPUDenseRecall::recall(const vector<bool> &data,
                                    const vector<float> &thresholds,
                                    const vector<vector<float> > &weights) {
  vector<bool> state;
  size_t size = data.size();
  bool stable;

  bool dataArray[size];
  float thresholdArray[size];
  float weightArray[size][size];

  bool * stableDev;
  bool * stateDev;
  float * thresholdDev;
  float * weightDev;
  unsigned numThreads = 256;
  unsigned numBlocks = size / numThreads;

  if (size % numThreads) numBlocks++;

  for (size_t i = 0; i < size; ++i) { {
    dataArray[i] = data[i];
    thresholdArray[i] = thresholds[i];

    for (size_t j = 0; j < size; ++j) {
      weightArray[i][j] = weights[i][j];
    }
  }

  assert(cudaMemcpy(stateDev, dataArray, size * sizeof(bool),
                    cudaMemcpyHostToDevice) == cudaSuccess);

  assert(cudaMemcpy(thresholdDev, thresholdArray, size * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess);

  do {
    stable = true;
    assert(cudaMemcpy(stableDev, &stable, sizeof(bool),
                      cudaMemcpyHostToDevice) == cudaSuccess);

    gpu_dense_recall_kernel<<< numBlocks, numThreads >>>
      (size, stateDev, thresholdDev, weightDev, stableDev);

    assert(cudaDeviceSynchronize() == cudaSuccess);

    assert(cudaMemcpy(&stable, stableDev, sizeof(bool),
                      cudaMemcpyHostToDevice) == cudaSuccess);
  } while (!stable);

  assert(cudaMemcpy(dataArray, stateDev, size * sizeof(bool),
                    cudaMemcpyDeviceToHost) == cudaSuccess);

  for (size_t i = 0; i < size; ++i) {
    state[i] = dataArray[i];
  }

  return state;
}
