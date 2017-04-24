
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
  bool stable;
  do {
    stable = true;
    for (size_t i = 0; i < size; i += groupSize) {
      /*for (unsigned j = 0; j < size; j++) {
        cout << state[j] << " ";
        }
        cout << endl;*/
      bool updates[groupSize];
#pragma omp parallel for
      for (size_t j = 0; j < groupSize; j++) {
        if (i + j < size) {
          float value = 0;
          for (size_t k = 0; k < size; k++) {
            if (state[k])
              value += weights[i + j][k];
            else
              value -= weights[i + j][k];
          }
          updates[j] = value > thresholds[i + j];
#pragma omp atomic
          stable &= updates[j] == state[i + j];
        }
      }
      for (size_t j = 0; j < groupSize && i + j < size; j++) {
        state[i + j] = updates[j];
      }
    }
  } while (!stable);
  return state;
}
