
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

void CPUHebbianTraining::train(const vector<bool> &data,
                               vector<vector<float> > &weights,
                               unsigned numDataSets) {
  size_t size = data.size();
  vector<int8_t> elems(size);
  for (size_t i = 0; i < size; i++) {
    if (data[i])
      elems[i] = 1;
    else
      elems[i] = -1;
  }

#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      float newWeight = elems[i] * elems[j];
      weights[i][j] = (weights[i][j] * numDataSets + newWeight) / (numDataSets + 1);
    }
  }
}
