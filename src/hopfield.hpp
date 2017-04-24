#pragma once

// Note: __CUDACC__ macro tests are used to exclude __device__ and __host__ specifiers if compiling with gcc

#include <vector>
#include <array>
#include <cassert>
#include <iostream>

#define DEFAULT_THRESHOLD 0.1

// Strategy abstract classes for various implementations of training and recall
class Training {
public:
  virtual ~Training() {}

  virtual void train(const std::vector<bool> &data,
         std::vector<std::vector<float> > &weights,
         unsigned numDataSets) = 0;
  virtual std::string getName() const = 0;
};

class Recall {
public:
  virtual ~Recall() {}

  virtual std::vector<bool> recall(const std::vector<bool> &data,
           const std::vector<float> &thresholds,
           const std::vector<std::vector<float> > &weights) = 0;
  virtual std::string getName() const = 0;
};

// Implementation subclasses
class CPUHebbianTraining : public Training {
public:
  ~CPUHebbianTraining() {};

  void train(const std::vector<bool> &data,
       std::vector<std::vector<float> > &weights,
       unsigned numDataSets);
  std::string getName() const { return "CPU Hebbian"; }
};

class CPUStorkeyTraining : public Training {
public:
  ~CPUStorkeyTraining() {};

  void train(const std::vector<bool> &data,
       std::vector<std::vector<float> > &weights,
       unsigned numDataSets);
  std::string getName() const { return "CPU Storkey"; }
};

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
      Recall *recallImpl = new CPUDenseRecall(),
      Training *trainingImpl = new CPUHebbianTraining()) :
    size(thresholds.size()),
    trainingImpl(trainingImpl),
    recallImpl(recallImpl),
    weights(size, std::vector<float>(size, 0)),
    thresholds(thresholds),
    numDataSets(0) {}

  HopfieldNetwork(size_t size,
      float threshold = DEFAULT_THRESHOLD,
      Recall *recallImpl = new CPUDenseRecall(),
      Training *trainingImpl = new CPUHebbianTraining()) :
    HopfieldNetwork(std::vector<float>(size, threshold), recallImpl, trainingImpl) {}

  ~HopfieldNetwork() {
    delete recallImpl;
    delete trainingImpl;
  }

  void train(const std::vector<bool> &data) {
    assert(data.size() == size);
    trainingImpl->train(data, weights, numDataSets++);
  }

  std::vector<bool> recall(const std::vector<bool> &data) {
    assert(data.size() == size);
    return recallImpl->recall(data, thresholds, weights);
  }

  const size_t size;
private:
  Training *trainingImpl;
  Recall *recallImpl;

  std::vector<std::vector<float> > weights;
  const std::vector<float> thresholds;

  unsigned numDataSets;
};
