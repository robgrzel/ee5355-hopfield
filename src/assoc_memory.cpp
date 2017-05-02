#include "assoc_memory.hpp"

#include <iostream>
#include <string>
using namespace std;

void AssociativeMemory::store(const vector<bool> &data) {
  assert(data.size() == size);
    
  if (network != NULL)
    delete network;
  network = NULL;
    
  trainingImpl->train(data, weights, numDataSets++);
}

vector<bool> AssociativeMemory::recall(const vector<bool> &data) {
  assert(data.size() == size);
    
  if (network == NULL)
    network = evaluationImpl->makeHopfieldNetwork(thresholds, weights);
    
  return network->evaluate(data);
}

Training *getTraining(const string &name) {
  if (name == "hebbian")
    return new CPUHebbianTraining;
  else if (name == "storkey")
    return new CPUStorkeyTraining;
  else {
    cerr << "Unknown training algorithm " << name << endl;
    exit(1);
  }
}
