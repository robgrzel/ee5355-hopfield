#pragma once

#include <vector>
#include <string>
#include <array>
#include <cassert>
#include <iostream>

#define DEFAULT_WEIGHT_THRESHOLD 0.21

// Some macros...
#define cudaCheck(stmt)                                                 \
  do {                                                                  \
    cudaError_t err = stmt;                                             \
    if (err != cudaSuccess) {                                           \
      std::cout << "Failed to run stmt " #stmt << std::endl;            \
      std::cout << "Got CUDA error ...  " << cudaGetErrorString(err) << std::endl; \
    exit(1);                                                            \
    }                                                                   \
  } while (0)

#define _unused(x) ((void)(x))

// Representation of a Hopfield network
class HopfieldNetwork {
public:
  HopfieldNetwork(const std::vector<float> &thresholds,
                  const std::vector<std::vector<float>> &weights) :
    size(thresholds.size()) {
    if(weights.size() != size) {
      printf("ASSERT FAIL\nweights.size=%lu\nthreshold.size=%lu\n", weights.size(), thresholds.size());
    };
    _unused(weights); // Suppress warning in release build
  }
  
  virtual ~HopfieldNetwork() {}
  
  virtual std::vector<bool> evaluate(const std::vector<bool> &data) = 0;

  const size_t size;
};

class CPUDenseHopfieldNetwork : public HopfieldNetwork {
public:
  CPUDenseHopfieldNetwork(const std::vector<float> &thresholds,
                          const std::vector<std::vector<float>> &weights) :
    HopfieldNetwork(thresholds, weights),
    thresholds(thresholds),
    weights(weights) {}
  ~CPUDenseHopfieldNetwork() {}
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  std::vector<float> thresholds;
  std::vector<std::vector<float> > weights;
};

class GPUDenseHopfieldNetwork : public HopfieldNetwork {
public:
  GPUDenseHopfieldNetwork(const std::vector<float> &thresholds,
                          const std::vector<std::vector<float>> &weights);
  ~GPUDenseHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  // Device memory
  float *thresholdsDev; // size
  float *weightsDev;    // size * size
};

class GPUDenseBitHopfieldNetwork : public HopfieldNetwork {
public:
  GPUDenseBitHopfieldNetwork(const std::vector<float> &thresholds,
                             const std::vector<std::vector<float>> &weights);
  ~GPUDenseBitHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  // Device memory
  float *thresholdsDev; // size
  float *weightsDev;    // size * size
};

class GPUDenseBlockHopfieldNetwork : public HopfieldNetwork {
public:
  GPUDenseBlockHopfieldNetwork(const std::vector<float> &thresholds,
			       const std::vector<std::vector<float>> &weights);
  ~GPUDenseBlockHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  // Device memory
  float *thresholdsDev; // size
  float *weightsDev;    // size * size
};

class GPUDenseBlockCoarseHopfieldNetwork : public HopfieldNetwork {
public:
  GPUDenseBlockCoarseHopfieldNetwork(const std::vector<float> &thresholds,
                               const std::vector<std::vector<float>> &weights,
                               size_t parallel=10);
  ~GPUDenseBlockCoarseHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

  const size_t parallel;
protected:
  // Device memory
  float *thresholdsDev; // size
  float *weightsDev;    // size * size
};

class GPUDenseCutoffHopfieldNetwork : public HopfieldNetwork {
public:
  GPUDenseCutoffHopfieldNetwork(const std::vector<float> &thresholds,
			       const std::vector<std::vector<float>> &weights);
  ~GPUDenseCutoffHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  CPUDenseHopfieldNetwork cpuNet;
  // Device memory
  float *thresholdsDev; // size
  float *weightsDev;    // size * size
};


class GPUDenseCoarseHopfieldNetwork : public HopfieldNetwork {
public:
  GPUDenseCoarseHopfieldNetwork(const std::vector<float> &thresholds,
				const std::vector<std::vector<float>> &weights);
  ~GPUDenseCoarseHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  // Device memory
  float *thresholdsDev; // size
  float *weightsDev;    // size * size
};

class SparseHopfieldNetwork : public HopfieldNetwork {
public:
  SparseHopfieldNetwork(const std::vector<float> &thresholds,
                        const std::vector<std::vector<float>> &weights,
                        float weightThreshold=DEFAULT_WEIGHT_THRESHOLD);
  virtual ~SparseHopfieldNetwork() {}
  void CSR_2_JDS();
  void CSR_2_ELL();
  std::vector<int> sort_indexes(std::vector<int> &v);
  
protected:
  const float weightThreshold;
  int w_size, w_col, w_row;
  int nnz, rowPtr, max_elements;
  std::vector<float> sW_nnz;
  std::vector<int> sW_colInd;
  std::vector<int> sW_rowPtr;
  std::vector<float> jds_w_nnz;
  std::vector<int> jds_w_colInd;
  std::vector<int> jds_w_rowPtr;
  std::vector<int> row;
  std::vector<int> nnzPerRow;
  std::vector<float> ell_w_nnz;
  std::vector<int> ell_w_colInd;
  std::vector<float> ell_w_nnzT;
  std::vector<int> ell_w_colIndT;
  //int *max_elements_d;
  //std::vector<int> sW_rowPtr;

};

class CPUSparseHopfieldNetwork : public SparseHopfieldNetwork {
public:
  CPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights,
                           float weightThreshold=DEFAULT_WEIGHT_THRESHOLD):
    SparseHopfieldNetwork(thresholds, weights, weightThreshold), 
    thresholds(thresholds) {}
  ~CPUSparseHopfieldNetwork() {}
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  std::vector<float> thresholds;
};

class GPUSparseHopfieldNetwork : public SparseHopfieldNetwork {
public:
  GPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights,
                           float weightThreshold=DEFAULT_WEIGHT_THRESHOLD);
  ~GPUSparseHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);
  
protected:
  bool *stable_d;
  bool *state_d;
  float *threshold_d;  // size
  float *sW_nnz_d;     // Number of Non zero elements
  int *sW_colInd_d;    // Number of Non zero elements
  int *sW_rowPtr_d;    // size+1

};


class GPUSparseELLHopfieldNetwork : public SparseHopfieldNetwork {
public:
  GPUSparseELLHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights,
                           float weightThreshold=DEFAULT_WEIGHT_THRESHOLD);
  ~GPUSparseELLHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);
  
protected:
  bool *stable_d;
  bool *state_d;
  float *threshold_d;  // size
  float *sW_nnz_d;     // Number of Non zero elements
  int *sW_colInd_d;    // Number of Non zero elements
  int *sW_rowPtr_d;    // size+1

  float *ell_w_nnz_d;     // Number of Non zero elements
  int *ell_w_colInd_d;    // Number of Non zero elements
  int max_elements_d;    // size+1
  
};


class GPUSparseELLCoalHopfieldNetwork : public SparseHopfieldNetwork {
public:
  GPUSparseELLCoalHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights,
                           float weightThreshold=DEFAULT_WEIGHT_THRESHOLD);
  ~GPUSparseELLCoalHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);
  
protected:
  bool *stable_d;
  bool *state_d;
  float *threshold_d;  // size
  float *sW_nnz_d;     // Number of Non zero elements
  int *sW_colInd_d;    // Number of Non zero elements
  int *sW_rowPtr_d;    // size+1

  float *ell_w_nnz_d;     // Number of Non zero elements
  int *ell_w_colInd_d;    // Number of Non zero elements
  int max_elements_d;    // size+1
  
};

class GPUSparseJDSHopfieldNetwork : public SparseHopfieldNetwork {
public:
  GPUSparseJDSHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights,
                           float weightThreshold=DEFAULT_WEIGHT_THRESHOLD);
  ~GPUSparseJDSHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);
  
protected:
  bool *stable_d;
  bool *state_d;
  float *threshold_d;  // size
  float *sW_nnz_d;     // Number of Non zero elements
  int *sW_colInd_d;    // Number of Non zero elements
  int *sW_rowPtr_d;    // size+1
  float *jds_w_nnz_d;     // Number of Non zero elements
  int *jds_w_colInd_d;    // Number of Non zero elements
  int *jds_w_rowPtr_d;    // size+1
  int *row_d;    // size
  float * jdsT_w_nnz_d;
  int  *jdsT_w_colInd_d, *jdsT_w_colStartIdx_d;

};

class GPUSparseQueueHopfieldNetwork : public SparseHopfieldNetwork {
public:
  GPUSparseQueueHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights,
                           float weightThreshold=DEFAULT_WEIGHT_THRESHOLD);
  ~GPUSparseQueueHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);
  
protected:
  bool *stable_d;
  bool *state_d;
  float *threshold_d;  // size
  float *sW_nnz_d;     // Number of Non zero elements
  int *sW_colInd_d;    // Number of Non zero elements
  int *sW_rowPtr_d;    // size+1
  int *nodePtr;
  

};

class GPUSparseWarpHopfieldNetwork : public SparseHopfieldNetwork {
public:
  GPUSparseWarpHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights,
                           float weightThreshold=DEFAULT_WEIGHT_THRESHOLD);
  ~GPUSparseWarpHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);
  
protected:
  bool *stable_d;
  bool *state_d;
  float *threshold_d;  // size
  float *sW_nnz_d;     // Number of Non zero elements
  int *sW_colInd_d;    // Number of Non zero elements
  int *sW_rowPtr_d;    // size+1
  int *d_nnzPerVector;
  float *d_w_dense;

  float *ell_w_nnz_d;     // Number of Non zero elements
  int *ell_w_colInd_d;    // Number of Non zero elements
  int max_elements_d;    // size+1

};

// Factory class for Hopfield networks
class Evaluation {
public:
  virtual ~Evaluation() {}

  virtual HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                               const std::vector<std::vector<float>> &weights) = 0;
  virtual std::string getName() const = 0;
};

// Implementation subclasses
class CPUDenseEvaluation : public Evaluation {
public:
  ~CPUDenseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new CPUDenseHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "CPU dense"; }
};

class GPUDenseEvaluation : public Evaluation {
public:
  ~GPUDenseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense"; }
};

class GPUDenseBitEvaluation : public Evaluation {
public:
  ~GPUDenseBitEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseBitHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense bit"; }
};

class GPUDenseBlockEvaluation : public Evaluation {
public:
  ~GPUDenseBlockEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseBlockHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense block"; }
};

class GPUDenseBlockCoarseEvaluation : public Evaluation {
public:
  GPUDenseBlockCoarseEvaluation(size_t parallel=10) : parallel(parallel) {}
  ~GPUDenseBlockCoarseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseBlockCoarseHopfieldNetwork(thresholds, weights, parallel);
  }
  std::string getName() const { return "GPU dense block coarse"; }
  
  const size_t parallel;
};

class GPUDenseCutoffEvaluation : public Evaluation {
public:
  ~GPUDenseCutoffEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseCutoffHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense cutoff"; }
};

class GPUDenseCoarseEvaluation : public Evaluation {
public:
  ~GPUDenseCoarseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseCoarseHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense coarse"; }
};

class SparseEvaluation : public Evaluation {
public:
  SparseEvaluation(float weightThreshold=DEFAULT_WEIGHT_THRESHOLD) :
    weightThreshold(weightThreshold) {}
  virtual ~SparseEvaluation() {}
  
protected:
  const float weightThreshold;
};

class CPUSparseEvaluation : public SparseEvaluation {
public:
  CPUSparseEvaluation(float weightThreshold=DEFAULT_WEIGHT_THRESHOLD) :
    SparseEvaluation(weightThreshold) {}
  ~CPUSparseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new CPUSparseHopfieldNetwork(thresholds, weights, weightThreshold);
  }
  std::string getName() const { return "CPU sparse"; }
};

class GPUSparseEvaluation : public SparseEvaluation {
public:
  GPUSparseEvaluation(float weightThreshold=DEFAULT_WEIGHT_THRESHOLD) :
    SparseEvaluation(weightThreshold) {}
  ~GPUSparseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUSparseHopfieldNetwork(thresholds, weights, weightThreshold);
  }
  std::string getName() const { return "GPU sparse - CSR"; }
};

class GPUSparseELLEvaluation : public SparseEvaluation {
public:
  GPUSparseELLEvaluation(float weightThreshold=DEFAULT_WEIGHT_THRESHOLD) :
    SparseEvaluation(weightThreshold) {}
  ~GPUSparseELLEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUSparseELLHopfieldNetwork(thresholds, weights, weightThreshold);
  }
  std::string getName() const { return "GPU sparse - ELL Non-Coalesced"; }
};


class GPUSparseELLCoalEvaluation : public SparseEvaluation {
public:
  GPUSparseELLCoalEvaluation(float weightThreshold=DEFAULT_WEIGHT_THRESHOLD) :
    SparseEvaluation(weightThreshold) {}
  ~GPUSparseELLCoalEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUSparseELLCoalHopfieldNetwork(thresholds, weights, weightThreshold);
  }
  std::string getName() const { return "GPU sparse - ELL Coalesced"; }
};

class GPUSparseJDSEvaluation : public SparseEvaluation {
public:
  GPUSparseJDSEvaluation(float weightThreshold=DEFAULT_WEIGHT_THRESHOLD) :
    SparseEvaluation(weightThreshold) {}
  ~GPUSparseJDSEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUSparseJDSHopfieldNetwork(thresholds, weights, weightThreshold);
  }
  std::string getName() const { return "GPU sparse - JDS"; }
};


class GPUSparseQueueEvaluation : public SparseEvaluation {
public:
  GPUSparseQueueEvaluation(float weightThreshold=DEFAULT_WEIGHT_THRESHOLD) :
    SparseEvaluation(weightThreshold) {}
  ~GPUSparseQueueEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUSparseQueueHopfieldNetwork(thresholds, weights, weightThreshold);
  }
  std::string getName() const { return "GPU sparse queued"; }
};


class GPUSparseWarpEvaluation : public SparseEvaluation {
public:
  GPUSparseWarpEvaluation(float weightThreshold=DEFAULT_WEIGHT_THRESHOLD) :
    SparseEvaluation(weightThreshold) {}
  ~GPUSparseWarpEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUSparseWarpHopfieldNetwork(thresholds, weights, weightThreshold);
  }
  std::string getName() const { return "GPU sparse warp_parallel"; }
};


Evaluation *getEvaluation(const std::string &name);
