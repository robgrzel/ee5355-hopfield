#include "hopfield.hpp"

#include <iostream>
#include <string>
using namespace std;

Training *getTraining(const std::string &name) {
  if (name == "hebbian")
    return new CPUHebbianTraining;
  else if (name == "storkey")
    return new CPUStorkeyTraining;
  else {
    cerr << "Unknown training algorithm " << name << endl;
    exit(1);
  }
}

Recall *getRecall(const std::string &name) {
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
