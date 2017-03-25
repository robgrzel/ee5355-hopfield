
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

vector<bool> CPUDenseRecall::recall(const vector<bool> &data,
				    const vector<float> &thresholds,
				    const vector<vector<float> > &weights) {
  vector<bool> state = data;
  size_t size = data.size();
  bool stable = false;
  do {
    vector<bool> newState(size);
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      float value = 0;
      for (size_t j = 0; j < size; j++) {
	if (state[j])
	  value += weights[i][j];
	else
	  value -= weights[i][j];
      }
      bool newStateVal = value > thresholds[i];
#pragma omp atomic
      stable |= newStateVal == state[i];
#pragma omp critical // Needed since STL isn't thread-safe for modification...
      newState[i] = newStateVal;
    }
    state = newState;
  } while (!stable);
  return state;
}
