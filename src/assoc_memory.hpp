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
class AssociativeMemory {
public:
  AssociativeMemory(const std::vector<float> &thresholds,
                    Training *trainingImpl = new CPUHebbianTraining,
                    Evaluation *evaluationImpl = new CPUDenseEvaluation) :
    thresholds(thresholds),
    weights(thresholds.size(), std::vector<float>(thresholds.size())),
    trainingImpl(trainingImpl),
    evaluationImpl(evaluationImpl),
    network(NULL),
    numDataSets(0) {}

  AssociativeMemory(size_t size,
                    float threshold = DEFAULT_THRESHOLD,
                    Training *trainingImpl = new CPUHebbianTraining,
                    Evaluation *evaluationImpl = new CPUDenseEvaluation) :
    AssociativeMemory(std::vector<float>(size, threshold), trainingImpl, evaluationImpl) {}

  ~AssociativeMemory() {
    delete trainingImpl;
    delete evaluationImpl;
    if (network != NULL)
      delete network;
  }

  void store(const std::vector<bool> &data) {
    assert(data.size() == size);
    
    if (network != NULL)
      delete network;
    network = NULL;
    
    trainingImpl->train(data, weights, numDataSets++);
  }

  std::vector<bool> recall(const std::vector<bool> &data) {
    assert(data.size() == size);
    
    if (network == NULL)
      network = evaluationImpl->makeHopfieldNetwork(thresholds, weights);
    
    return network->evaluate(data);
  }
  
private:
  std::vector<float> thresholds;
  std::vector<std::vector<float> > weights;
    
  Training *trainingImpl;
  Evaluation *evaluationImpl;
  HopfieldNetwork *network;

  size_t numDataSets;
};

