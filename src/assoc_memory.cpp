#include "assoc_memory.hpp"

#include <iostream>
#include <string>
using namespace std;

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
