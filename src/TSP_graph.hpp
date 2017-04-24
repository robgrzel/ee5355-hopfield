// Taken from http://www.cs.sfu.ca/CourseCentral/125/tjd/tsp_example.html
#include "Point.h"

class TSP_graph {
private:
  vector<Point> cities;

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

}; // class TSP_graph