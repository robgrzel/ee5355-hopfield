#include "hopfield.hpp"
#include "assoc_memory.hpp"

#include <vector>
#include <cstdlib>
#include <iostream>
using namespace std;

#define NUM_TESTS 5
#define DATA_SIZE 50
#define CLEAR_KEY 2

int main() {
  cout << "Training: " << endl;
  
  vector<vector<bool>> data(NUM_TESTS, vector<bool>(DATA_SIZE));
  for (unsigned i = 0; i < NUM_TESTS; i++) {
    for (unsigned j = 0; j < DATA_SIZE; j++) {
      data[i][j] = rand() % 2;
      cout << data[i][j] << " ";
    }
    cout << endl;
  }
  
  TrainedHopfieldNetwork net(DATA_SIZE, 0, new CPUDenseRecall(), new CPUHebbianTraining());
  for (unsigned i = 0; i < NUM_TESTS; i++) {
    net.train(data[i]);
  }
  
  cout << "Recall: " << endl;
  
  for (unsigned i = 0; i < NUM_TESTS; i++) {
    vector<bool> key = data[i];
    for (unsigned j = 0; j < DATA_SIZE; j++) {
      if (!(rand() % CLEAR_KEY))
        key[j] = 0;
    }
    for (unsigned j = 0; j < DATA_SIZE; j++) {
      cout << key[j] << " ";
    }
    cout << "->" << endl;
    vector<bool> result = net.recall(key);
    for (unsigned j = 0; j < DATA_SIZE; j++) {
      if (result[j] != data[i][j])
        cout << "\033[31m";
      cout << result[j] << " ";
      if (result[j] != data[i][j])
        cout << "\033[39m";
    }
    cout << endl;
    cout << endl;
  }
}
