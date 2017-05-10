#include "hopfield.hpp"
#include "mincut.hpp"

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <limits>
#include <vector>
using namespace std;

#define MIN_SIZE 50
#define MAX_SIZE 15000
#define STEP_SIZE 50
#define TRIALS 5

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
    vector<double> times(numEvaluationAlgorithms);
    vector<float> accuracies(numEvaluationAlgorithms);
    for (unsigned i = 0; i < numEvaluationAlgorithms; i++) {
      Evaluation *evaluation = getEvaluation(evaluationAlgorithms[i]);
      cerr << "Evaluation algorithm: " << evaluation->getName() << endl;
      cerr << "           Data size: " << size << endl;
      
      cerr << "Generating test graph... " << flush;
      auto t1 = chrono::high_resolution_clock::now();
      MinCutGraph graph(size);
      auto t2 = chrono::high_resolution_clock::now();
      chrono::duration<double> diff = t2 - t1;
      cerr << diff.count() << " sec" << endl;

      double time = numeric_limits<int>::max();
      for (unsigned j = 0; j < TRIALS; j++) {
        cerr << "Partitioning graph... " << flush;
        t1 = chrono::high_resolution_clock::now();
        graph.partitionGraph(evaluation);
        t2 = chrono::high_resolution_clock::now();
        diff = t2 - t1;
        cerr << diff.count() << " sec" << endl;
        if (diff.count() < time) {
          time = diff.count();
	}
      }
      cerr << "Best time:     " << time << " sec" << endl;
      times[i] = time;
    }
    
    cout << size;
    for (double time : times) {
      cout << "," << time;
    }
    cout << endl;
  }
}
