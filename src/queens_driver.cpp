#include "queens.hpp"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
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

  Queens nQueens(num, gamma, threshold);
  do {
    nQueens.solve();
    system("clear");
    nQueens.verifySolution();
    nQueens.printSolution();
    printf("%f queens on average for %u iterations\n", nQueens.getAverage(), nQueens.getIterations());
  } while (!nQueens.verifySolution());

  printf("%d,%.2f,%.2f,%u\n", num, gamma, threshold, nQueens.getIterations());

  return 0;
}
