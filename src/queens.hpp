#pragma once

#include <vector>

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
