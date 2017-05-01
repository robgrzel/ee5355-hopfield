
#include "hopfield.hpp"

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include "TSP_graph.hpp"
using namespace std;

#define NUM_TESTS 5

int main() {
  printf("Training patterns: \n");
  
  srand(time(NULL)); // use current time to seed random number generator
  
  // Create a random pattern matrix to learn.
  // Each row is a separate pattern to learn (n bits each).

  // max capacity (number of patterns it can learn) of Hopfield network is 0.138N (N: number of neurons)
  // https://en.wikipedia.org/wiki/Hopfield_network#Capacity

  // calculate the weight matrix

  // print the weight matrix
  // cout<<"The weight matrix:"<<endl<<endl;
  // for(j=0;j<n;j++)
  // {
  //   for(i=0;i<n;i++)
  //     printf("%2d ",w[j*n+i]);
  //   cout<<endl;
  // }
  // cout<<endl;

  // Initialize TSP
  int numCities = 0;
  vector<TSP_graph> data(NUM_TESTS);
  for (unsigned i = 0; i < NUM_TESTS; i++) {
    numCities = (rand() % 10) + 1;
    for (unsigned j = 0; j < numCities; j++) {
      int x = rand() % 100;
      int y = rand() % 100;
      data[i].add(x, y);
      printf("data[%d]: city %d added at (%d, %d)\n", i, j, x, y);
    }
    printf("data[%d] total %d cities\n", i, data[i].size());
  }

  // HopfieldNetwork net(numCities, 0, new CPUDenseRecall(), new CPUHebbianTraining());
  // for (unsigned i = 0; i < NUM_TESTS; i++) {
  //   net.train(data[i]);
  // }

  // cout << "Recall: " << endl;

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
