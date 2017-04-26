#pragma once

// Note: __CUDACC__ macro tests are used to exclude __device__ and __host__ specifiers if compiling with gcc

#include <vector>
#include <string>
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

Training *getTraining(const std::string &name);

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

// Representation of a Hopfield network that is trained from data vectors
class TrainedHopfieldNetwork : public HopfieldNetwork {
public:
  TrainedHopfieldNetwork(const std::vector<float> thresholds,
                         Recall *recallImpl = new CPUDenseRecall(),
                         Training *trainingImpl = new CPUHebbianTraining()) :
    HopfieldNetwork(thresholds, std::vector<std::vector<float>>(thresholds.size(), std::vector<float>(thresholds.size(), 0)), recallImpl),
    trainingImpl(trainingImpl),
    numDataSets(0) {}

  TrainedHopfieldNetwork(size_t size,
                         float threshold = DEFAULT_THRESHOLD,
                         Recall *recallImpl = new CPUDenseRecall(),
                         Training *trainingImpl = new CPUHebbianTraining()) :
    TrainedHopfieldNetwork(std::vector<float>(size, threshold), recallImpl, trainingImpl) {}

  ~TrainedHopfieldNetwork() {
    delete trainingImpl;
  }

  void train(const std::vector<bool> &data) {
    assert(data.size() == size);
    trainingImpl->train(data, weights, numDataSets++);
  }
  
private:
  Training *trainingImpl;
  unsigned numDataSets;
};
