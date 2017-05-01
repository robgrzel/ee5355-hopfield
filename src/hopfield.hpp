#pragma once

#include <vector>
#include <string>
#include <array>
#include <cassert>
#include <iostream>

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
  
  std::string getName() const { return "GPU dense"; }
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  // Device memory
  float *thresholdsDev; // size
  float *weightsDev;    // size * size
};

class CPUSparseHopfieldNetwork : public HopfieldNetwork {
public:
  CPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights);
  ~CPUSparseHopfieldNetwork();
  
  std::string getName() const { return "CPU sparse"; }
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  // TODO: Fill in representation of a sparse Hopfield network for the host
  //std::vector<float> thresholds;
  //std::vector<std::vector<float> > weights;
};

class GPUSparseHopfieldNetwork : public HopfieldNetwork {
public:
  GPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights);
  ~GPUSparseHopfieldNetwork();
  
  std::string getName() const { return "GPU sparse"; }
  std::vector<bool> evaluate(const std::vector<bool> &data);
  
protected:
  // TODO: Fill in representation of a sparse Hopfield network for the device
  //float *thresholds; // size
  //float *weights;    // size * size
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
  std::string getName() const { return "CPU dense"; }
};

class CPUSparseEvaluation : public Evaluation {
  ~CPUSparseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new CPUSparseHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "CPU sparse"; }
};

class GPUSparseEvaluation : public Evaluation {
  ~GPUSparseEvaluation() {}
  
  HopfieldNetwork *makeHopfieldNetwork(const std::vector<float> &thresholds,
                                       const std::vector<std::vector<float>> &weights) {
    return new GPUSparseHopfieldNetwork(thresholds, weights);
  }
  std::string getName() const { return "CPU sparse"; }
};

Evaluation *getEvaluation(const std::string &name);
