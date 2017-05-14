#include "hopfield.hpp"
#include "assoc_memory.hpp"

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <limits>
#include <vector>
using namespace std;

#define SIZE 5000
#define MIN_PARALLEL 100
#define MAX_PARALLEL 5000
#define STEP_PARALLEL 400
#define MIN_NUM_VECTORS 150
#define MAX_NUM_VECTORS 500
#define STEP_NUM_VECTORS 10

#define KEY_SIZE_PROP 0.25

int main() {
  cout << "# of data vectors";
  for (unsigned parallel = MIN_PARALLEL; parallel <= MAX_PARALLEL; parallel += STEP_PARALLEL) {
    cout << "," << parallel << " Parallel Updates";
  }
  cout << endl;
  
  for (size_t num_vectors = MIN_NUM_VECTORS; num_vectors <= MAX_NUM_VECTORS; num_vectors += STEP_NUM_VECTORS) {
    vector<float> accuracies;
    for (unsigned parallel = MIN_PARALLEL; parallel <= MAX_PARALLEL; parallel += STEP_PARALLEL) {
      cerr << "# of parallel updates: " << parallel << endl;
      size_t size = SIZE;
      cerr << "            Data size: " << size << endl;
      cerr << "    # of data vectors: " << num_vectors << endl;
      float key_size_prop = KEY_SIZE_PROP;
      cerr << "             Key size: " << size * key_size_prop << endl;
      
      cerr << "Generating test data... " << flush;
      auto t1 = chrono::high_resolution_clock::now();
      vector<vector<bool>> data(num_vectors, vector<bool>(size));
      vector<vector<bool>> keys(num_vectors, vector<bool>(size, false));
      for (size_t j = 0; j < num_vectors; j++) {
        for (size_t k = 0; k < size; k++) {
          data[j][k] = rand() % 2;
          if ((j + k) % (size_t)(1 / key_size_prop) == 0)
            keys[j][k] = data[j][k];
        }
      }
      auto t2 = chrono::high_resolution_clock::now();
      chrono::duration<double> diff = t2 - t1;
      cerr << diff.count() << " sec" << endl;

      cerr << "Training network... " << flush;
      t1 = chrono::high_resolution_clock::now();
      AssociativeMemory mem(size, DEFAULT_THRESHOLD,
                            new CPUHebbianTraining(),
                            new GPUDenseBlockCoarseEvaluation(parallel));
      for (size_t j = 0; j < num_vectors; j++) {
        mem.store(data[j]);
      }
      t2 = chrono::high_resolution_clock::now();
      diff = t2 - t1;
      cerr << diff.count() << " sec" << endl;

      cerr << "Initializing network... " << flush;
      t1 = chrono::high_resolution_clock::now();
      mem.init();
      t2 = chrono::high_resolution_clock::now();
      diff = t2 - t1;
      cerr << diff.count() << " sec" << endl;

      cerr << "Recalling data... " << flush;
      t1 = chrono::high_resolution_clock::now();
      vector<vector<bool>> results(num_vectors);
      for (size_t k = 0; k < num_vectors; k++) {
        results[k] = mem.recall(keys[k]);
      }
      t2 = chrono::high_resolution_clock::now();
      diff = t2 - t1;
      cerr << diff.count() << " sec" << endl;

      cerr << "Checking accuracy... " << flush;
      t1 = chrono::high_resolution_clock::now();
      size_t num_correct = 0;
      for (size_t k = 0; k < num_vectors; k++) {
        num_correct += (results[k] == data[k]);
      }
      float accuracy = (float)num_correct * 100 / num_vectors;
      t2 = chrono::high_resolution_clock::now();
      diff = t2 - t1;
      cerr << diff.count() << " sec" << endl;
      cerr << accuracy << "% correct" << endl;
      accuracies.push_back(accuracy);
    }
    
    cout << num_vectors;
    for (double accuracy : accuracies) {
      cout << "," << accuracy;
    }
    cout << endl;
  }
}
