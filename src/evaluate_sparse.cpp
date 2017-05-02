#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

CPUSparseHopfieldNetwork::CPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                                                   const std::vector<std::vector<float>> &weights,
                                                   float weightThreshold) :
  SparseHopfieldNetwork(thresholds, weights, weightThreshold), 
    thresholds(thresholds) {
  // Converting dense weight matrix to sparse matrix
  size_t size = thresholds.size(); //size threshold = size data
  int w_size = (int)size;
  int w_col = w_size;
  int w_row = w_size;


  int nnz=0;
  int rowPtr = 0;
  for(int i=0; i < w_row; ++i)
  {
	sW_rowPtr.push_back(rowPtr);
	for(int j=0; j < w_col; ++j)
	{
		if(weights[i][j]*weights[i][j]>weightThreshold*weightThreshold)
		{
			sW_nnz.push_back(weights[i][j]);
			sW_colInd.push_back(j);
			++nnz;	
		}
	}
	rowPtr=nnz;

  }
  
  sW_rowPtr.push_back(rowPtr); // Last pointer equal number of NNZ elements
 

  //Sparse matrix Debuging code

  printf("Percentage of NNZ elements in weight matrix using threshold %f = %f%%\n", weightThreshold,(100.00*nnz/(w_size*w_size)));
/*
   for (int f=0; f<nnz;++f)
      printf("%f  ",sW_nnz[f]);
   printf("\n");
   for (int f=0; f<nnz;++f)
      printf("%d  ",sW_colInd[f]);
   printf("\n");
   for (int f=0; f<w_row+1;++f)
      printf("%d  ",sW_rowPtr[f]);
   printf("\n");
*/
  
}

CPUSparseHopfieldNetwork::~CPUSparseHopfieldNetwork() {

}

vector<bool> CPUSparseHopfieldNetwork::evaluate(const vector<bool> &data) {
  vector<bool> state = data;
  size_t size = data.size();
  bool stable;

  do {
    stable = true;
#pragma omp parallel for

    for (size_t i = 0; i < size; i++)
    { 
	float value = 0;     
     	for(int k=sW_rowPtr[i]; k<sW_rowPtr[i+1];k++)
	{			
		if (state[sW_colInd[k]])
         		 value += sW_nnz[k];
       		else
     			value -= sW_nnz[k];
	}

      bool update = value > thresholds[i];
#pragma omp atomic
      stable &= update == state[i];
      state[i] = update;
    }
  } while (!stable);

  return state;
}

