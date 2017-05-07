#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <iostream>
using namespace std;

vector<bool> CPUDenseHopfieldNetwork::evaluate(const vector<bool> &data) {
  assert(data.size() == size);
  vector<bool> state = data;
  bool stable;

#ifdef DEBUG
  int iters = 0;
#endif

  do {
    stable = true;
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      /*for (size_t j = 0; j < size; j++) {
        cout << state[j] << " ";
        }
        cout << endl;*/
      float value = 0;
      for (size_t k = 0; k < size; k++) {
        if (state[k])
          value += weights[i][k];
        else
          value -= weights[i][k];
      }
      bool update = value > thresholds[i];
#pragma omp atomic
      stable &= update == state[i];
      state[i] = update;
    }

#ifdef DEBUG
    iters++;
    if((iters % 1000)==0) {
    printf("================================================\niteration %d\n", iters);
      for (int i = 0; i < sqrt(size); ++i)
      {
        for (int k = 0; k < sqrt(size); ++k)
        {
          printf("%s ", state[(i*sqrt(size))+k] ? "ja    " : "nein  ");
        }
        printf("\n");
      }
      printf("\n");
    }
    // usleep(1e4);
  } while ((!stable) && (iters < 1000000000));
#else
  } while (!stable);
#endif

  return state;
}
