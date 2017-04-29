
#include "hopfield.hpp"

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include "TSP_graph.hpp"
using namespace std;

#define NUM_TESTS 5
#define DATA_SIZE 50
#define CLEAR_KEY 2

int main() {
  printf("Training: \n");

  // Initialize TSP
  vector<TSP_graph> data(NUM_TESTS);
  // for (unsigned i = 0; i < NUM_TESTS; i++) {
  //   for (unsigned j = 0; j < DATA_SIZE; j++) {
  //     data[i][j] = rand() % 2;
  //     printf("%d ", data[i][j]);
  //   }
  //   printf("\n");
  // }

  // HopfieldNetwork net(DATA_SIZE, 0, new CPUDenseRecall(), new CPUHebbianTraining());
  // for (unsigned i = 0; i < NUM_TESTS; i++) {
  //   net.train(data[i]);
  // }

  // cout << "Recall: " << endl;

  // for (unsigned i = 0; i < NUM_TESTS; i++) {
  //   vector<bool> key = data[i];
  //   for (unsigned j = 0; j < DATA_SIZE; j++) {
  //     if (!(rand() % CLEAR_KEY)) {
  //       key[j] = 0;
  //     }
  //   }
  //   for (unsigned j = 0; j < DATA_SIZE; j++) {
  //     cout << key[j] << " ";
  //   }
  //   cout << "->" << endl;
  //   vector<bool> result = net.recall(key);
  //   for (unsigned j = 0; j < DATA_SIZE; j++) {
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
