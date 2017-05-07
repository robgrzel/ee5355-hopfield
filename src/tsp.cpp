
#include "hopfield.hpp"

#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstdio>
#include "TSP_graph.hpp"
using namespace std;

int main(int argc, char ** argv) {
  srand(time(NULL)); // use current time to seed random number generator
  string cities = "cities";

  if (argc >= 2) {
    cities = string(argv[1]);
    printf("reading cities from file named %s\n", argv[1]);
  } else {
    printf("reading cities from file named 'cities'\n");
  }
  float gamma = 1.0f;
  if (argc >= 3) {
    gamma = atof(argv[2]);
  }

  // Initialize TSP
  TSP_graph tsp(gamma);
  ifstream in;

  in.open(cities);
  int x, y, i = 0;
  while (in >> x >> y) {
    printf("tsp: city %d added at (%d, %d)\n", i, x, y);
    tsp.add(x, y);
  }
  in.close();

  vector<bool> data(tsp.size()*tsp.size(), false);
  /* Network is represented as a tour matrix. Each node is an entry in the matrix

       x  | 1 2 ... n (Time)
    ------+-----------------
     S1   |
     S2   |
     ...  |
     Sn   |
    (City)

  x is stored row-major ie, x = {(S1, 1), (S1, 2)...}
  */

  CPUDenseHopfieldNetwork network(tsp.get_thresholds(), tsp.get_weights());
  data = network.evaluate(data);

  for (int i = 0; i < tsp.size(); ++i) {
    for (int k = 0; k < tsp.size(); ++k) {
      printf("%s ", data[(i*tsp.size())+k] ? "ja    " : "nein  ");
    }
    printf("\n");
  }
  printf("\n");

  // float E_tsp = 0.0;
  // float L = 0.0;
  // // Calculate the constraint "one time in each city"
  // float Condition = 
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
