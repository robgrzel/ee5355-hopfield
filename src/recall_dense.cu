
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

__global__ void gpu_dense_recall_kernel(size_t size,
          bool *data,
          float *thresholds,
          float *weights) {
}

vector<bool> GPUDenseRecall::recall(const vector<bool> &data,
            const vector<float> &thresholds,
            const vector<vector<float> > &weights) {
  size_t size = data.size();
  bool dataArray[size];
  float thresholdArray[size];
  float weightArray[size][size];

  for (size_t i = 0; i < size; i++) {
    dataArray[size] = data[size];
    thresholdArray[size] = thresholds[i];

    for (size_t j = 0; j < size; j++) {
      weightArray[i][j] = weights[i][j];
    }
  }

  // TODO: Implement me!
  assert(false);
  return data;
}
