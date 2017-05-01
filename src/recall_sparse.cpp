#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;
#define WEIGHT_THRESHOLD 0.05


vector<bool> CPUSparseRecall::recall(const vector<bool> &data,
                                     const vector<float> &thresholds,
                                     const vector<vector<float> > &weights) {
  // Converting dense weight matrix to sparse matrix
  vector<bool> state = data;
  size_t size = data.size();
  int w_size = (int)size;
  int w_col = w_size;
  int w_row = w_size;
  float *sW_nnz;
  int *sW_colInd;
  int *sW_rowPtr;

  sW_nnz = (float*)malloc(sizeof(float)*w_row*w_col);
  sW_colInd = (int*)malloc(sizeof(int)*w_row*w_col);
  sW_rowPtr = (int*)malloc(sizeof(int)*w_row+1);

  int nnz=0;
  int rowPtr = 0;
  for(int i=0; i < w_row; ++i)
  {
	sW_rowPtr[i]=rowPtr;
	for(int j=0; j < w_col; ++j)
	{
		if(weights[i][j]*weights[i][j]>WEIGHT_THRESHOLD*WEIGHT_THRESHOLD)
		{
			sW_nnz[nnz]=weights[i][j];
			sW_colInd[nnz] = j;
			++nnz;	
		}
	}
	rowPtr=nnz;

  }
  
  sW_rowPtr[w_row]=rowPtr; // Last pointer equal number of NNZ elements
 

 	//Sparse matrix Debuging code
   printf("Percentage of NNZ elements in weight matrix using threshold %f = %f\n", WEIGHT_THRESHOLD,(100.00*nnz/(w_size*w_size)));
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

