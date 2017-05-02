#include <iostream>
#include <vector>
#include "mincut.hpp"

template<typename T>
void printVector(std::vector<std::vector<T> > v) {
    for (unsigned i = 0; i < v.size(); i++) {
        for (unsigned j = 0; j < v[i].size(); j++)
            std::cout << v[i][j] << " ";
        std::cout << std::endl;
    }
}

int main(void) {
	std::vector<std::vector<float> > v { { 1, 1, 1}, { 2, 2, 2 } };
	MinCutGraph graph(v);
}
