// Taken from http://www.cs.sfu.ca/CourseCentral/125/tjd/tsp_example.html
#ifndef TSP_GRAPH_H
#define TSP_GRAPH_H

#include "Point.h"
#include <vector>
#include <cstdio>
#include <iostream>

using namespace std;
class TSP_graph {
private:
  vector<Point> cities;
  vector<vector<float> > w;
  float gamma;
  float threshold;

public:

  TSP_graph(float g) : gamma(g), threshold(-g / 2.0) {}

  float get_threshold() const {
    return threshold;
  }

  // add a city at point p
  void add(const Point& p) {
    cities.push_back(p);
  }

  // add a city at (x, y)
  void add(double x, double y) {
    Point p(x, y);
    add(p);
  }

  // # of cities
  int size() const {
    return cities.size();
  }

  // calculate the distance between cities i and j
  double dist_between(int i, int j) const {
    return dist(cities[i], cities[j]);  // dist is from Point.h
  }

  // calculate the score of the given tour
  double score(const vector<int>& tour) const {
    double result = dist_between(tour[0], tour[size() - 1]);
    for(int i = 1; i < size(); ++i) {                // i starts at 1, not 0!
      result += dist_between(tour[i - 1], tour[i]);
    }
    return result;
  }

  // return a starting tour of all the cities in the order 0, 1, 2, 3,
  // ...
  vector<int> get_default_tour() const {
    vector<int> tour(size());
    for(int i = 0; i < size(); ++i) {
      tour[i] = i;
    }
    return tour;
  }

  vector<vector<float> > calculate_weights() {
    w = vector<vector<float> > (size() * size(), vector<float>(size() * size(), 0));
    printf("weights size = %lu\n", w.size());
    // calculate the weight matrix
    for (int i = 0; i < size(); ++i) {
      for (int k1 = 0; k1 < size(); ++k1) {
        for (int j = 0; j < size(); ++j) {
          for (int k2 = 0; k2 < size(); ++k2) {
            // printf("weight calculated for (i=%d, k1=%d) to (j=%d, k1+1=%d\n", i,k1,j,k2);
            float t = ((i == j) || (k1 == k2))?-gamma:0;

            w[(i * size()) + k1][(j * size()) + k2] = -dist_between(i, j) + t;
          }
        }
      }
    }
    return w;
  }

  void print_weights() {
    // print the weight matrix
    cout<<"The weight matrix:"<<endl<<endl;
    for (int i = 0; i < size(); ++i) {
      for (int k1 = 0; k1 < size(); ++k1) {
        printf("Node (city=%d, time=%d)\n", i, k1);
        for (int j = 0; j < size(); ++j) {
          for (int k2 = 0; k2 < size(); ++k2) {
            printf("%4.4f\t", w[(i * size()) + k1][(j * size()) + k2]);
          }
          cout<<endl;
        }
        cout<<endl<<endl;
      }
    }
  }

  vector<vector<float> > get_weights() {
    calculate_weights();
#ifdef DEBUG
    printf("Returning the following weights matrix\n");
    print_weights();
#endif
    return w;
  }

  vector<float> get_thresholds() {
    return vector<float>(size() * size(), threshold);
  }
}; // class TSP_graph
#endif
