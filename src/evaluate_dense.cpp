#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <iostream>
using namespace std;

#ifdef DEBUG
#define NEIN(s) (!(s) && (iters < 1000000000))
#else
#define NEIN(s) (!(s))
#endif

vector<bool> CPUDenseHopfieldNetwork::evaluate(const vector<bool> &data) {
  assert(data.size() == size);
  vector<bool> state = data;
  bool stable;
  vector<bool> last = data;

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
      last[i] = state[i];
      state[i] = update;
    }

#ifdef DEBUG
    iters++;
    printf("\033c");
    printf("================================================\niteration %d\n", iters);
    for (int i = 0; i < sqrt(size); ++i) {
      for (int k = 0; k < sqrt(size); ++k) {
        if (state[(i*sqrt(size))+k] == last[(i * sqrt(size)) + k])
          printf("%s", state[(i*sqrt(size))+k] ? "1" : "0");
        else
          printf("\x1b[43m\x1b[31m%s\x1b[0m",
                 state[(i*sqrt(size))+k] ? "1" : "0");

      }
      printf("\n");
    }
    printf("\n");
#endif

  } while (NEIN(stable));

  return state;
}
