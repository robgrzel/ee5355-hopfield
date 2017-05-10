#ifndef UTILS_INCLUDE
#define UTILS_INCLUDE

#include <vector>
#include <iostream>
#include <iomanip>

template<typename T>    
void printVector(std::vector<std::vector<T> > v) {
    for (unsigned i = 0; i < v.size(); i++) {
        for (unsigned j = 0; j < v[i].size(); j++) {
            std::cout << std::fixed << std::setw(3) << std::setprecision(2) << v[i][j];
            std::cout << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>    
void printVector(std::vector<T> v) {
    for (unsigned i = 0; i < v.size(); i++) {
            
        std::cout << std::fixed << std::setw(3) << std::setprecision(2) << v[i];    
        std::cout << " ";
        }
        std::cout << std::endl;
}

#endif
