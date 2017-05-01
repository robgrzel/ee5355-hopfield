
#include "hopfield.hpp"

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include "TSP_graph.hpp"
using namespace std;

#define GAMMA 1
#define THRESHOLD (-GAMMA/2)

int main() {
  srand(time(NULL)); // use current time to seed random number generator
  // Initialize TSP
  TSP_graph tsp;
  int numCities = 0;
  numCities = (rand() % 3) + 1;
  for (unsigned i = 0; i < numCities; i++) {
    int x = rand() % 100;
    int y = rand() % 100;
    tsp.add(x, y);
    printf("tsp: city %d added at (%d, %d)\n", i, x, y);
  }
  printf("tsp total %d cities\n", tsp.size());

  /* Network is represented as a tour matrix. Each node is an entry in the matrix

          | 1 2 ... n (Time)
    ------+-----------------
     S1   |
     S2   |
     ...  |
     Sn   |
    (City)

  */
  float E_tsp = 0.0;

  vector<float> thresholds(tsp.size(), THRESHOLD);
  HopfieldNetwork net(thresholds, tsp.get_weights());
  // for (unsigned i = 0; i < NUM_TESTS; i++) {
  //   net.train(data[i]);
  // }

  // cout << "Recall: " << endl;
  // Calculate L - The length of the trip

  // Calculate the constraint "one time in each city"

  // for (unsigned i = 0; i < NUM_TESTS; i++) {
  //   vector<bool> key = data[i];
  //   for (unsigned j = 0; j < numCities; j++) {
  //     if (!(rand() % CLEAR_KEY)) {
  //       key[j] = 0;
  //     }
  //   }
  //   for (unsigned j = 0; j < numCities; j++) {
  //     cout << key[j] << " ";
  //   }
  //   cout << "->" << endl;
  //   vector<bool> result = net.recall(key);
  //   for (unsigned j = 0; j < numCities; j++) {
  //     if (result[j] != data[i][j]) {
  //       cout << "\033[31m";
  //     }
  //     cout << result[j] << " ";
  //     if (result[j] != data[i][j]) {
  //       cout << "\033[39m";
  //     }
  //   }
  //   cout << endl;
  //   cout << endl;
  // }
}
