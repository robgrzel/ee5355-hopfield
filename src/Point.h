// Taken from http://www.cs.sfu.ca/CourseCentral/125/tjd/oop_intro
// which is down. Use the Google cached version
// http://webcache.googleusercontent.com/search?q=cache:vH-4b0fhndYJ:www.cs.sfu.ca/CourseCentral/125/tjd/oop_intro.html+&cd=1&hl=en&ct=clnk&gl=us

#ifndef POINT_H
#define POINT_H

// If the absolute value of the difference of two doubles is the less
// than min_diff, then they will be considered equal.
const double min_diff = 0.00000000001;

struct Point {
  double x;
  double y;

  Point() : x(0), y(0) {
  }

  Point(int a, int b) : x(a), y(b) {
  }

  Point(const Point& p) : x(p.x), y(p.y) {
  }

  bool equals(const Point& p) {
    return abs(p.x - x) < min_diff && abs(p.y - y) < min_diff;
  }

  bool operator==(const Point& p) {
    return equals(p);
  }

  bool operator!=(const Point& p) {
    return !equals(p);
  }

  void print() {
    cout << "(" << x << ", " << y << ")";
  }

  void println() {
    print();
    cout << endl;
  }

  ~Point() {
  }
}; // Point

ostream& operator<<(ostream& out, const Point& p) {
  out << "(" << p.x << ", " << p.y << ")";
  return out;
}

double dist(const Point& p, const Point& q) {
  double dx = p.x - q.x;
  double dy = p.y - q.y;
  return sqrt(dx * dx + dy * dy);
}

#endif