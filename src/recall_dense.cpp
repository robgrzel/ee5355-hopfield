
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
