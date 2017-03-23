
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
      weights[i][j] += (elems[i] * data[j] - elems[i] * h[j][i] - elems[j] * h[i][j]) / numDataSets;
    }
  }
}


vector<bool> CPUDenseRecall::recall(const vector<bool> &data,
				    const vector<float> &thresholds,
				    const vector<vector<float> > &weights) {
  vector<bool> state = data;
  size_t size = data.size();
  bool stable = false;
  do {
    vector<bool> newState(size);
    stable = true;
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      float value = 0;
      for (size_t j = 0; j < size; j++) {
	if (state[j])
	  value += weights[i][j];
	else
	  value -= weights[i][j];
      }
      newState[i] = value > thresholds[i];
      #pragma omp atomic
      stable &= newState[i] == state[i];
    }
    state = newState;
  } while (!stable);
  return state;
}
