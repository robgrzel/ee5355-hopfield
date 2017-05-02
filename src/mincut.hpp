#ifndef MIN_CUT_GRAPH
#define MIN_CUT_GRAPH

#include <vector>
#include "hopfield.hpp"

/***************************************************
 * defines a class to marshall graph specifications
 * to a corresponding hopfield network.
 *
 * weights between nodes and stationary node aka station (see proposal)
 * are decided
 *
 * stationary node is the one that is permanently classified
 * as 1
 *
 *
 *
 * all other nodes are assigned arbitrary states, and
 * updates are performed to finalize labels and hence,
 * the cut
 *
 *
 ***************************************************/

class MinCutGraph {
    private:
        /*
         * vertices are labelled from 0 to weights.size()-1
         * 
         *
         */
        std::vector<std::vector<float> > weights;

        /*
         * Method to pick station
         * 
         * Only uses state of the graph to decide
         * no need for external inputs
         */
        unsigned pickStation();

        /*
         * Generate threshold vector given station
         * for hopfield network
         *
         */
        std::vector<float> generateThresholds(unsigned station);

        /*
         * Generate weight matrix given station
         * for hopfield network
         *
         */
        std::vector<std::vector<float> > generateWeights(unsigned station);

        /*
         *
         * I think there is no initialization that guarantees 
         * quick / better convergence, if that were the case,
         * wouldn't need to solve in the first place
         *
         */
        std::vector<bool> generateInitialStates();

        /*
         * Returns graph index given hopfield index
         * and choice of station
         */
        static unsigned mapToGraphIndex(unsigned station, unsigned hopfieldIndex);

        /*
         *
         */
        static unsigned mapToHopfieldIndex(unsigned station, unsigned minCutIndex);



    public:
        MinCutGraph(std::vector<std::vector<float> > &weights) : 
            weights(weights) {}

        /*
         *
         */
        std::vector<std::vector<unsigned> > partitionGraph(Evaluation *const evaluationImpl);
};

#endif
