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
    assert(weights.size() == size);
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

class GPUDenseCutoffHopfieldNetwork : public HopfieldNetwork {
public:
  GPUDenseCutoffHopfieldNetwork(const std::vector<float> &thresholds,
			       const std::vector<std::vector<float>> &weights);
  ~GPUDenseCutoffHopfieldNetwork();
  
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  std::vector<float> thresholds;
  std::vector<std::vector<float> > weights;
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
  
protected:
  const float weightThreshold;
  int w_size, w_col, w_row;
  int nnz, rowPtr;
  std::vector<float> sW_nnz;
  std::vector<int> sW_colInd;
  std::vector<int> sW_rowPtr;
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
  ~CPUDenseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new CPUDenseHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "CPU dense"; }
};

class GPUDenseEvaluation : public Evaluation {
  ~GPUDenseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense"; }
};

class GPUDenseBitEvaluation : public Evaluation {
  ~GPUDenseBitEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseBitHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense bit"; }
};

class GPUDenseBlockEvaluation : public Evaluation {
  ~GPUDenseBlockEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseBlockHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense block"; }
};

class GPUDenseCutoffEvaluation : public Evaluation {
  ~GPUDenseCutoffEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUDenseCutoffHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "GPU dense cutoff"; }
};

class GPUDenseCoarseEvaluation : public Evaluation {
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
  std::string getName() const { return "GPU sparse"; }
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
  std::string getName() const { return "GPU sparse with GPU pre processing"; }
};


Evaluation *getEvaluation(const std::string &name);
