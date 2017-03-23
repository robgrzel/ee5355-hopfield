#pragma once

// Note: __CUDACC__ macro tests are used to exclude __device__ and __host__ specifiers if compiling with gcc

#include <stdint.h>
//#include <stdbool.h>
#include <vector>
#include <array>
#include <cassert>
#include <iostream>

#define DEFAULT_THRESHOLD 0.5

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
  ~CPUDenseRecall() {};

  std::vector<bool> recall(const std::vector<bool> &data,
			   const std::vector<float> &thresholds,
			   const std::vector<std::vector<float> > &weights);
  std::string getName() const { return "CPU dense"; }
};

// Representation of a Hopfield network
class HopfieldNetwork {
public:
  HopfieldNetwork(const std::vector<float> thresholds,
		  Recall *recallImpl = new CPUDenseRecall(),
		  Training *trainingImpl = new CPUStorkeyTraining()) :
    size(thresholds.size()),
    trainingImpl(trainingImpl),
    recallImpl(recallImpl),
    weights(size, std::vector<float>(size, 0)),
    thresholds(thresholds),
    numDataSets(0) {}

  HopfieldNetwork(size_t size,
		  Recall *recallImpl = new CPUDenseRecall(),
		  Training *trainingImpl = new CPUStorkeyTraining()) :
    HopfieldNetwork(std::vector<float>(size, DEFAULT_THRESHOLD), recallImpl, trainingImpl) {}

  void train(const std::vector<bool> &data) {
    assert(data.size() == size());
    trainingImpl->train(data, weights, numDataSets++);
  }
  std::vector<bool> recall(const std::vector<bool> &data) {
    assert(data.size() == size());
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
