
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
  double E_tsp = 0.0;

  vector<vector<double>> w;
  w.resize(pow(tsp.size(),2), vector<double>(pow(tsp.size(),2), 0));
  // calculate the weight matrix
  for (int k = 0; k < tsp.size(); ++k)
  {
    for (int i = 0; i < tsp.size(); ++i)
    {
      for (int k_plus_1 = 0; k_plus_1 < tsp.size(); ++k_plus_1)
      {
        for (int j = 0; j < tsp.size(); ++j)
        {
          double t = ((i == j) || (k == k_plus_1))?0:-GAMMA;

          w[(k * tsp.size()) + i][(k_plus_1 * tsp.size()) + j] = -tsp.dist_between(i, j) + t;
        }
      }
    }
  }

  // print the weight matrix
  cout<<"The weight matrix:"<<endl<<endl;
  for (int i = 0; i < tsp.size(); ++i)
  {
    for (int k = 0; k < tsp.size(); ++k)
    {
      printf("Node (city=%d, time=%d)\n", i, k);
      for (int j = 0; j < tsp.size(); ++j)
      {
        for (int k_plus_1 = 0; k_plus_1 < tsp.size(); ++k_plus_1)
        {
          printf("%2f\t", w[(k * tsp.size()) + i][(k_plus_1 * tsp.size()) + j]);
        }
        cout<<endl;
      }
      cout<<endl<<endl;
    }
  }
  // HopfieldNetwork net(numCities, 0, new CPUDenseRecall(), new CPUHebbianTraining());
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
