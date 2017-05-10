#include "queens.hpp"
#include "hopfield.hpp"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>

using namespace std;

vector<float> Queens::getThresholds() {
  return vector<float>(num * num, threshold);
}

void Queens::printWeights(vector<vector<float>> &weights) {
  for (int i = 0; i < num * num; ++i) {
    for(int j = 0; j < num * num; ++j)
      cout << setw(5) << weights[i][j];
    cout << endl;
  }
}

/*
 * NOTE: I have just been playing around with this function and gamma/threshold.
 * Choosing parameters such that
 *      N * gamma < threshold < 2N * gamma
 * seems to work for some values pretty well (i.e. you actually get a
 * solution sometimes).
 */

bool isDiagonal(int i, int j, int k, int l) {
  int diffx = abs(i - k);
  int diffy = abs(j - l);
  return diffx == diffy;
}
 
vector<vector<float>> Queens::getWeights() {
  vector<vector<float>> weights(num * num, vector<float>(num * num, 0));

  for (int i = 0; i < num; ++i)
    for (int j = 0; j < num; ++j)
      for (int k = 0; k < num; ++k)
        for(int l = 0; l < num; ++l) {
          if (i == k && j == l)
            weights[i * num + j][k * num + l] = gamma;
          else if (i == k || j == l)
            weights[i * num + j][k * num + l] = -gamma;
          else if (isDiagonal(i, j, k, l))
            weights[i * num + j][k * num + l] = -gamma;
          else
            weights[i * num + j][k * num + l] = 0;
        }

#ifdef DEBUG
  printWeights(weights);
#endif

  return weights;
}

void Queens::solve() {
  vector<bool> board(num * num, false);

  CPUDenseHopfieldNetwork network(getThresholds(), getWeights());


  solution = network.evaluate(board);
}

void Queens::printSolution() {
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < num; ++j) {
      if ((i + j) % 2 == 0)
        cout << "\x1b[40m\x1b[37m";
      else
        cout << "\x1b[47m\x1b[30m";
      cout << (solution[i * num + j] ? "Q" : " ") << "\x1b[0m";
    }
    cout << endl;
  }
  cout << "\x1b[0m";
}

/**
 * @brief Verifies that solution is a valid n-queens solution
 * @retval false - solution is invalid
 * @retval true  - solution is valid
 */
bool Queens::verifySolution() {
  // Check each row
  for (int i = 0, count = 0; i < num; ++i) {
    count = 0;
    for (int j = 0; j < num; ++j) {
      count += solution[i * num + j];
    }
    if (count != 1) return false;
  }
  // Check each col
  for (int j = 0, count = 0; j < num; ++j) {
    count = 0;
    for (int i = 0; i < num; ++i) {
      count += solution[i * num + j];
    }
    if (count != 1) return false;
  }
  // TODO Check diagonals

  return true;
}

