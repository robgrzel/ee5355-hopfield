
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

__global__ void gpu_sparse_recall_kernel() {
  // TODO
}

GPUSparseHopfieldNetwork::GPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                                                   const std::vector<std::vector<float>> &weights) :
    HopfieldNetwork(thresholds, weights) {
  // TODO
}

GPUSparseHopfieldNetwork::~GPUSparseHopfieldNetwork() {
  // TODO
}

vector<bool> GPUSparseHopfieldNetwork::evaluate(const vector<bool> &data) {
  // TODO: Implement me!
  assert(false);
  return data;
}
