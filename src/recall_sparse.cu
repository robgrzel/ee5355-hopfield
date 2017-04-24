
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

__global__ void gpu_sparse_recall_kernel() {
  // TODO
}

vector<bool> GPUSparseRecall::recall(const vector<bool> &data,
                                     const vector<float> &thresholds,
                                     const vector<vector<float> > &weights) {
  // TODO: Implement me!
  assert(false);
  return data;
}
