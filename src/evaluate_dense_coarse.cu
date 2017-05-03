
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define COARSEN 4
#define BLOCK_SIZE 32

__global__ void gpu_dense_coarse_recall_kernel(size_t size,
					       bool * state,
					       float * thresholds,
					       float * weights,
					       bool * stable) {
  extern __shared__ bool localState[];

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    bool localStable = true;

    for (size_t j = 0; j < COARSEN; j++) {
      float value = 0.0f;
      for (size_t k = 0; k < size; ++k) {
	if (state[k])
	  value += weights[i * size + k];
	else
	  value -= weights[i * size + k];
      }

      bool update = value > thresholds[i];
      localStable &= (update == state[i]);
      state[i] = update;
    }
    
    if (!localStable)
      *stable = false;
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
  unsigned numBlocks = size / BLOCK_SIZE;

  if (size % BLOCK_SIZE) numBlocks++;

  cudaCheck(cudaMalloc((void**) &stateDev, sizeof(bool) * size));
  cudaCheck(cudaMalloc((void**) &stableDev, sizeof(bool)));

  copy(data.begin(), data.end(), dataArray);
  cudaCheck(cudaMemcpy(stateDev, dataArray, size * sizeof(bool),
                       cudaMemcpyHostToDevice));

  do {
    stable = true;
    cudaCheck(cudaMemcpy(stableDev, &stable, sizeof(bool),
                         cudaMemcpyHostToDevice));

    gpu_dense_coarse_recall_kernel<<< numBlocks, BLOCK_SIZE, size * sizeof(bool) >>>
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

