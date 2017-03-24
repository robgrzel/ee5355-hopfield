
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

void CPUStorkeyTraining::train(const vector<bool> &data,
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

  float h[size][size];
  #pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      h[i][j] = 0;
      for (size_t k = 0; k < size; k++) {
	if (k != i && k != j)
	  h[i][j] += weights[i][k] * elems[k];
      }
    }
  }
  #pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      weights[i][j] += (elems[i] * elems[j] - elems[i] * h[j][i] - elems[j] * h[i][j]) / numDataSets;
    }
  }
}
