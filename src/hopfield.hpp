#pragma once

#include <vector>
#include <string>
#include <array>
#include <cassert>
#include <iostream>

// Representation of a Hopfield networl
class HopfieldNetwork {
public:
  HopfieldNetwork(const std::vector<float> &thresholds,
                  const std::vector<std::vector<float>> &weights) :
    size(thresholds.size()) {
    assert(weights.size() == size);
  }
  
  virtual ~HopfieldNetwork() {}
  
  virtual std::string getName() const = 0;
  virtual std::vector<bool> evaluate(const std::vector<bool> &data) = 0;

  const size_t size;
};

class CPUDenseHopfieldNetwork : public HopfieldNetwork {
public:
  HopfieldNetwork(const std::vector<float> &thresholds,
                  const std::vector<std::vector<float>> &weights) :
    HopfieldNetwork(thresholds, weights),
    thresholds(thresholds),
    weights(weights) {}
  
  std::string getName() const { return "CPU dense"; }
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  std::vector<float> thresholds;
  std::vector<std::vector<float> > weights;
};

class GPUDenseHopfieldNetwork : public HopfieldNetwork {
public:
  DenseHopfieldNetwork(const std::vector<float> &thresholds,
                       const std::vector<std::vector<float>> &weights) :
    HopfieldNetwork(thresholds, weights)
  ~DenseHopfieldNetwork();
  
  std::string getName() const { return "GPU dense"; }
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  // Device memory
  float *thresholds; // size
  float *weights;    // size * size
};

class CPUSparseHopfieldNetwork : public SparseHopfieldNetwork {
public:
  CPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights) :
    SparseHopfieldNetwork(thresholds, weights) {}
  
  std::string getName() const { return "CPU sparse"; }
  std::vector<bool> evaluate(const std::vector<bool> &data);

protected:
  // TODO: Fill in representation of a sparse Hopfield network for the host
  //std::vector<float> thresholds;
  //std::vector<std::vector<float> > weights;
};

class GPUSparseHopfieldNetwork : public SparseHopfieldNetwork {
public:
  CPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                           const std::vector<std::vector<float>> &weights) :
    SparseHopfieldNetwork(thresholds, weights);
  ~SparseHopfieldNetwork();
  
  std::string getName() const { return "GPU sparse"; }
  std::vector<bool> evaluate(const std::vector<bool> &data);
  
protected:
  // TODO: Fill in representation of a sparse Hopfield network for the device
  //float *thresholds; // size
  //float *weights;    // size * size
};

// Factory function for Hopfield networks
HopfieldNetwork *makeHopfieldNetwork(std::string name,
                                     const std::vector<float> &thresholds,
                                     const std::vector<std::vector<float>> &weights);
