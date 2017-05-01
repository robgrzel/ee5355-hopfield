#pragma once

#include <vector>
#include <string>
#include <array>
#include <cassert>
#include <iostream>

// Strategy abstract class for various implementations of recall
class Recall {
public:
  virtual ~Recall() {}
  
  virtual std::vector<bool> recall(const std::vector<bool> &data,
                                   const std::vector<float> &thresholds,
                                   const std::vector<std::vector<float> > &weights) = 0;
  virtual std::string getName() const = 0;
};

Recall *getRecall(const std::string &name);

// Implementation subclasses
class CPUDenseRecall : public Recall {
public:
  CPUDenseRecall(unsigned groupSize=8) : groupSize(groupSize) {};
  ~CPUDenseRecall() {};

  std::vector<bool> recall(const std::vector<bool> &data,
                           const std::vector<float> &thresholds,
                           const std::vector<std::vector<float> > &weights);
  std::string getName() const { return "CPU dense"; }

  const unsigned groupSize;
};

class CPUSparseRecall : public Recall {
public:
  ~CPUSparseRecall() {};

  std::vector<bool> recall(const std::vector<bool> &data,
                           const std::vector<float> &thresholds,
                           const std::vector<std::vector<float> > &weights);
  std::string getName() const { return "CPU sparse"; }
};

class GPUDenseRecall : public Recall {
public:
  ~GPUDenseRecall() {};

  std::vector<bool> recall(const std::vector<bool> &data,
                           const std::vector<float> &thresholds,
                           const std::vector<std::vector<float> > &weights);
  std::string getName() const { return "GPU dense"; }
};

class GPUSparseRecall : public Recall {
public:
  ~GPUSparseRecall() {};

  std::vector<bool> recall(const std::vector<bool> &data,
                           const std::vector<float> &thresholds,
                           const std::vector<std::vector<float> > &weights);
  std::string getName() const { return "GPU sparse"; }
};

// Representation of a Hopfield network
class HopfieldNetwork {
public:
  HopfieldNetwork(const std::vector<float> thresholds,
                  const std::vector<std::vector<float>> weights,
                  Recall *recallImpl = new CPUDenseRecall()) :
    size(thresholds.size()),
    thresholds(thresholds),
    weights(weights),
    recallImpl(recallImpl) {
    assert(weights.size() == size);
    assert(thresholds.size() == size);
  }
  
  virtual ~HopfieldNetwork() {
    delete recallImpl;
  }

  std::vector<bool> recall(const std::vector<bool> &data) {
    assert(data.size() == size);
    return recallImpl->recall(data, thresholds, weights);
  }

  const size_t size;

protected:
  std::vector<float> thresholds;
  std::vector<std::vector<float> > weights;
  
private:
  Recall *recallImpl;
};
