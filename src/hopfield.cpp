#include "hopfield.hpp"

#include <iostream>
#include <string>
using namespace std;

Evaluation *getEvaluation(const string &name) {
  if (name == "cpu_dense")
    return new CPUDenseEvaluation;
  else if (name == "cpu_sparse")
    return new CPUSparseEvaluation;
  else if (name == "gpu_dense")
    return new GPUDenseEvaluation;
  else if (name == "gpu_dense_bit")
    return new GPUDenseBitEvaluation;
  else if (name == "gpu_dense_block")
    return new GPUDenseBlockEvaluation;
  else if (name == "gpu_dense_cutoff")
    return new GPUDenseCutoffEvaluation;
  else if (name == "gpu_dense_coarse")
    return new GPUDenseCoarseEvaluation;
  else if (name == "gpu_sparse")
    return new GPUSparseEvaluation;
  else if (name == "gpu_sparse_queue")
    return new GPUSparseQueueEvaluation;
  else if (name == "gpu_sparse_gpp")
    return new GPUSparseGpuPreProEvaluation;
  else {
    cerr << "Unknown evaluation algorithm " << name << endl;
    exit(1);
  }
}
