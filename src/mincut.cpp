#include <vector>
#include "mincut.hpp"
#include "hopfield.hpp"
#include <assert.h>
/*
 * Just returns the last node of the graph
 *
 *
 * Need to revisit to see if any particular choice
 * would lead to better convergence
 *
 */
unsigned MinCutGraph::pickStation() {
   return weights.size()-1;
}

/*
 * The thresholds would simply be the edge weights
 * from the station to all other nodes
 *
 * TODO: Is a deepcopy required?
 */
std::vector<float> MinCutGraph::generateThresholds(unsigned station) {
    return weights[station];
}


/*
 * This method generates weight matrix of the hopfield network
 * given station
 *
 * The result is generated by omitting row and column corresponding to
 * station
 *
 * station between 
 */
std::vector<std::vector<float> > MinCutGraph::generateWeights(unsigned station) {
    /*
     * Must deepcopy
     * Assignment does deepcopy by itself
     */
    std::vector<std::vector<float> > hopfieldW = weights;
    hopfieldW.erase(hopfieldW.begin() + station);

    for (unsigned i = 0; i < hopfieldW.size(); i++)
        hopfieldW[i].erase(hopfieldW[i].begin() + station);

    return hopfieldW;
}

/*
 *
 *
 *
 */
unsigned MinCutGraph::mapToGraphIndex(unsigned station, unsigned hopfieldIndex) {
    return (hopfieldIndex < station) ? hopfieldIndex : hopfieldIndex+1;
}

/*
 * Make sure that minCutIndex != station
 *
 */
unsigned MinCutGraph::mapToHopfieldIndex(unsigned station, unsigned graphIndex) {
    return (graphIndex < station) ? graphIndex : graphIndex-1;
}

/*
 * the hopfield network omits station
 *
 */
std::vector<bool> MinCutGraph::generateInitialStates() {
    unsigned size = weights.size()-1;
    std::vector<bool> states(size);
    for (unsigned i = 0; i < size; i++) {
        states.push_back(false);
    }
}

/*
 * 
 * 
 */
HopfieldNetwork MinCutGraph::generateHopfieldNetwork() {
	HopfieldNetwork network();
}
