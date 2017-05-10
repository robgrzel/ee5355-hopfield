#include "hopfield.hpp"
#include "assoc_memory.hpp"
#include "mincut.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << "<num vertices> <evaluation method>" << std::endl;
        exit(1);
    }

    unsigned numVertices = atoi(argv[1]);
    Evaluation *evaluation = getEvaluation(std::string(argv[2]));

    MinCutGraph graph(numVertices);
    //printVector<float>(graph.getWeights());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<unsigned> > partitions = graph.partitionGraph(evaluation);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    std::cout << diff.count() << " sec" << std::endl;

    std::cout << "\nPartitions, one partition per line: \n";
    printVector(partitions[0]);
    std::cout << "=====================================================\n";
    printVector(partitions[1]);

    return 0;
}
