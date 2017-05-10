#include "hopfield.hpp"
#include "mincut.hpp"

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <limits>
#include <vector>
using namespace std;

#define MIN_SIZE 5000
#define MAX_SIZE 20000
#define STEP_SIZE 5000
#define TRIALS 4

#define EVALUATION {"cpu_dense", "gpu_dense_cutoff", "cpu_sparse"}

int main() {
  string evaluationAlgorithms[] = EVALUATION;
  unsigned numEvaluationAlgorithms = sizeof(evaluationAlgorithms) / sizeof(string);

  cout << "Data Size";
  for (unsigned i = 0; i < numEvaluationAlgorithms; i++) {
    cout << "," << getEvaluation(evaluationAlgorithms[i])->getName();
  }
  cout << endl;
  
  for (size_t size = MIN_SIZE; size <= MAX_SIZE; size += STEP_SIZE) {
      cerr << "           Data size: " << size << endl;
      
      cerr << "Generating test graph... " << flush;
      auto t1 = chrono::high_resolution_clock::now();
      MinCutGraph graph(size);
      auto t2 = chrono::high_resolution_clock::now();
      chrono::duration<double> diff = t2 - t1;
      cerr << diff.count() << " sec" << endl;


    for (unsigned i = 0; i < numEvaluationAlgorithms; i++) {
      Evaluation *evaluation = getEvaluation(evaluationAlgorithms[i]);
      cerr << "Evaluation algorithm: " << evaluation->getName() << endl;
      double time = numeric_limits<int>::max();
      for (unsigned j = 0; j < TRIALS; j++) {
        cerr << "Partitioning graph... " << flush;
        graph.partitionGraph(evaluation);
    }
  }
  }
}
