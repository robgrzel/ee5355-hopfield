#pragma once

#include "hopfield.hpp"

#include <vector>
#include <string>
#include <array>
#include <cassert>
#include <iostream>

#define DEFAULT_THRESHOLD 0.1

// Strategy abstract class for various implementations of training
class Training {
public:
  virtual ~Training() {}
  
  virtual void train(const std::vector<bool> &data,
                     std::vector<std::vector<float> > &weights,
                     unsigned numDataSets) = 0;
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

Training *getTraining(const std::string &name);

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

