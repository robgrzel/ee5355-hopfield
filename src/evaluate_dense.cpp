#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

vector<bool> CPUDenseHopfieldNetwork::evaluate(const vector<bool> &data) {
  assert(data.size() == size);
  vector<bool> state = data;
  bool stable;
  do {
    stable = true;
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      float value = 0;
      for (size_t k = 0; k < size; k++) {
        if (state[k])
          value += weights[i][k];
        else
          value -= weights[i][k];
      }
      bool newStateVal = value > thresholds[i];
      bool updated = newStateVal != state[i];
      if (updated) {
	stable = false;
	state[i] = newStateVal;
      }
    }
  } while (!stable);
  return state;
}
