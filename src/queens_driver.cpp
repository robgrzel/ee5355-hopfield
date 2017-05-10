#include "queens.hpp"

#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char ** argv) {
  srand(time(0));
  int num = 8;
  float gamma;
  float threshold;

  if (argc != 4) {
    cerr << "usage: " << argv[0] << " N gamma threshold" << endl;
    exit(0);
  }

  num = atoi(argv[1]);
  gamma = atof(argv[2]);
  threshold = atof(argv[3]);

  unsigned repeats = 0;
  do {
    Queens nQueens(num, gamma, threshold);

    nQueens.solve();
    nQueens.printSolution();
  } while (!nQueens.verifySolution());

  printf("%d,%f,%f,%u\n", num, gamma, threshold, repeats);

  return 0;
}
