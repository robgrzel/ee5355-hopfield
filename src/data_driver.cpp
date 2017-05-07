#include "hopfield.hpp"
#include "assoc_memory.hpp"

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <limits>
#include <vector>
using namespace std;

#define MIN_SIZE 50
#define MAX_SIZE 10000
#define STEP_SIZE 50
#define TRIALS 5

#define EVALUATION {"cpu_dense", "gpu_dense", "gpu_dense_block", "cpu_sparse", "gpu_sparse", "gpu_sparse_queue"}

#define NUM_VECTORS 100
#define KEY_SIZE_PROP 0.25

int main() {
  string evaluationAlgorithms[] = EVALUATION;
  unsigned numEvaluationAlgorithms = sizeof(evaluationAlgorithms) / sizeof(string);

  cout << "Data Size";
  for (unsigned i = 0; i < numEvaluationAlgorithms; i++) {
    cout << "," << getEvaluation(evaluationAlgorithms[i])->getName();
  }
  cout << ",,Data Size";
  for (unsigned i = 0; i < numEvaluationAlgorithms; i++) {
    cout << "," << getEvaluation(evaluationAlgorithms[i])->getName();
  }
  cout << endl;
  
  for (size_t size = MIN_SIZE; size <= MAX_SIZE; size += STEP_SIZE) {
    vector<double> times(numEvaluationAlgorithms);
    vector<float> accuracies(numEvaluationAlgorithms);
    for (unsigned i = 0; i < numEvaluationAlgorithms; i++) {
      Evaluation *evaluation = getEvaluation(evaluationAlgorithms[i]);
      cerr << "Evaluation algorithm: " << evaluation->getName() << endl;
      cerr << "           Data size: " << size << endl;
      size_t num_vectors = NUM_VECTORS;
      cerr << "   # of data vectors: " << num_vectors << endl;
      float key_size_prop = KEY_SIZE_PROP;
      cerr << "            Key size: " << size * key_size_prop << endl;
      
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
      AssociativeMemory mem(size, DEFAULT_THRESHOLD, new CPUHebbianTraining(), evaluation);
      for (size_t j = 0; j < num_vectors; j++) {
        mem.store(data[j]);
      }
      t2 = chrono::high_resolution_clock::now();
      diff = t2 - t1;
      cerr << diff.count() << " sec" << endl;

      double time = numeric_limits<int>::max();
      float accuracy = 0;
      for (unsigned j = 0; j < TRIALS; j++) {
        cerr << "Recalling data... " << flush;
        t1 = chrono::high_resolution_clock::now();
        vector<vector<bool>> results(num_vectors);
        for (size_t k = 0; k < num_vectors; k++) {
          results[k] = mem.recall(keys[k]);
        }
        t2 = chrono::high_resolution_clock::now();
        diff = t2 - t1;
        cerr << diff.count() << " sec" << endl;
        if (diff.count() < time) {
          time = diff.count();

          cerr << "Checking accuracy... " << flush;
          t1 = chrono::high_resolution_clock::now();
          size_t num_correct = 0;
          for (size_t k = 0; k < num_vectors; k++) {
            num_correct += (results[k] == data[k]);
          }
          t2 = chrono::high_resolution_clock::now();
          diff = t2 - t1;
          cerr << diff.count() << " sec" << endl;
          cerr << (float)num_correct * 100 / num_vectors << "% correct" << endl;
          accuracy = (float)num_correct * 100  / num_vectors;
        }
      }
      cerr << "Best time:     " << time << " sec" << endl;
      cerr << "Best accuracy: " << accuracy << "% correct" << endl;
      times[i] = time;
      accuracies[i] = accuracy;
    }
    
    cout << size;
    for (double time : times) {
      cout << "," << time;
    }
    cout << ",," << size;
    for (double accuracy : accuracies) {
      cout << "," << accuracy;
    }
    cout << endl;
  }
}
