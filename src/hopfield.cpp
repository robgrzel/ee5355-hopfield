#include "hopfield.hpp"

#include <iostream>
#include <string>
using namespace std;

Recall *getRecall(const string &name) {
  if (name == "cpu_dense")
    return new CPUDenseRecall;
  else if (name == "cpu_sparse")
    return new CPUSparseRecall;
  else if (name == "gpu_dense")
    return new GPUDenseRecall;
  else if (name == "gpu_sparse")
    return new GPUSparseRecall;
  else {
    cerr << "Unknown recall algorithm " << name << endl;
    exit(1);
  }
}
