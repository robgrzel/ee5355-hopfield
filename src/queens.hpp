#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <stdlib.h>

class Queen {
  private:
    int x;
    int y;
  public:
    Queen() : x(0), y(0) {}
    Queen(int a, int b) : x(a), y(b) {}
    Queen(const Queen& q) : x(q.x), y(q.y) {
    }
    bool equals(const Queen& q) {
      return abs(q.x - x) == 0 && abs(q.y - y) == 0;
    }
    bool operator==(const Queen& q) {
      return equals(q);
    }
    bool operator!=(const Queen& q) {
      return !equals(q);
    }
    bool conflict(const Queen& q) {
      // return (x == q.x) || (y == q.y) || ((x+y) == (q.x+q.y));
      return (x == q.x) || (y == q.y);
    }
    void print() {
      std::cout << "(" << x << ", " << y << ")";
    }
    void println() {
      print();
      std::cout << std::endl;
    }
    ~Queen() {}
};

class Queens {
  private:
    int num;
    float gamma;
    float threshold;
    std::vector<bool> solution;

    void printWeights(std::vector<std::vector<float>> &weights);

  public:
    Queens(int n, float g, float t) : num(n), gamma(g), threshold(t) {}

    std::vector<float> getThresholds();
    std::vector<std::vector<float>> getWeights();

    void solve();
    void printSolution();
    bool verifySolution();

};
