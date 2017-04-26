#include "hopfield.hpp"

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>
using namespace std;

#define DEFAULT_SIZE 100
#define DEFAULT_NUM_VECTORS 10
#define DEFAULT_KEY_SIZE_PROP 0.1

int main(int argc, const char *argv[]) {
  if (argc < 2 || argc > 5) {
    cerr << "Usage: " << argv[0] << " <recall algorithm> <data size>(=" << DEFAULT_SIZE << ") <# of data vectors>(=" << DEFAULT_NUM_VECTORS << ") <fraction of vector included in key>(=" << DEFAULT_KEY_SIZE_PROP << ")" << endl;
                                                                                                                                                                          exit(1);
  }

  Recall *recall = getRecall(string(argv[1]));
  size_t size = DEFAULT_SIZE;
  if (argc >= 3)
    size = atoi(argv[2]);
  size_t num_vectors = DEFAULT_NUM_VECTORS;
  if (argc >= 4)
    num_vectors = atoi(argv[3]);
  float key_size_prop = DEFAULT_KEY_SIZE_PROP;
  if (argc >= 5)
    key_size_prop = atof(argv[4]);
  size_t key_size = size * key_size_prop;

  cout << "Generating test data... ";
  auto t1 = chrono::high_resolution_clock::now();
  vector<vector<bool>> data(num_vectors, vector<bool>(size));
  vector<vector<bool>> keys(num_vectors, vector<bool>(size, false));
  for (size_t i = 0; i < num_vectors; i++) {
    for (size_t j = 0; j < size; j++) {
      data[i][j] = rand() % 2;
      if ((i + j) % key_size == 0)
        keys[i][j] = data[i][j];
    }
  }
  auto t2 = chrono::high_resolution_clock::now();
  cout << (t2 - t1).count() << " sec" << endl;

  cout << "Training network... ";
  t1 = chrono::high_resolution_clock::now();
  TrainedHopfieldNetwork net(size, DEFAULT_THRESHOLD, recall, new CPUHebbianTraining());
  for (size_t i = 0; i < num_vectors; i++) {
    net.train(data[i]);
  }
  t2 = chrono::high_resolution_clock::now();
  cout << (t2 - t1).count() << " sec" << endl;

  cout << "Recalling data... ";
  t1 = chrono::high_resolution_clock::now();
  vector<vector<bool>> results(num_vectors);
  for (size_t i = 0; i < num_vectors; i++) {
    results[i] = net.recall(keys[i]);
  }
  t2 = chrono::high_resolution_clock::now();
  cout << (t2 - t1).count() << " sec" << endl;

  cout << "Checking accuracy... ";
  t1 = chrono::high_resolution_clock::now();
  size_t num_correct = 0;
  for (size_t i = 0; i < num_vectors; i++) {
    num_correct += (results[i] == data[i]);
  }
  t2 = chrono::high_resolution_clock::now();
  cout << (t2 - t1).count() << " sec" << endl;
  cout << num_correct * 100 / num_vectors << "% correct" << endl;
  
}
