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
    vector<vector<float> > w(pow(size(),2), vector<float>(pow(size(),2), 0));
      
    // calculate the weight matrix
    for (int k = 0; k < size(); ++k)
    {
      for (int i = 0; i < size(); ++i)
      {
        for (int k_plus_1 = 0; k_plus_1 < size(); ++k_plus_1)
        {
          for (int j = 0; j < size(); ++j)
          {
            float t = ((i == j) || (k == k_plus_1))?0:-gamma;

            w[(k * size()) + i][(k_plus_1 * size()) + j] = -dist_between(i, j) + t;
          }
        }
      }
    }
    return w;
  }

  void print_weights() {
    // print the weight matrix
    cout<<"The weight matrix:"<<endl<<endl;
    for (int i = 0; i < size(); ++i)
    {
      for (int k = 0; k < size(); ++k)
      {
        printf("Node (city=%d, time=%d)\n", i, k);
        for (int j = 0; j < size(); ++j)
        {
          for (int k_plus_1 = 0; k_plus_1 < size(); ++k_plus_1)
          {
            printf("%2f\t", w[(k * size()) + i][(k_plus_1 * size()) + j]);
          }
          cout<<endl;
        }
        cout<<endl<<endl;
      }
    }
  }

  vector<vector<float> > get_weights() {
    return w;
  }
}; // class TSP_graph
#endif
