#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
using namespace std;



std::vector<int> SparseHopfieldNetwork::sort_indexes( std::vector<int> &v) {

  // initialize original index locations
  std::vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  
  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  
  return idx;
}


void SparseHopfieldNetwork::CSR_2_ELL( )
{
  //Converting CSR to ELL
   max_elements = *std::max_element(nnzPerRow.begin(),nnzPerRow.end());
   for(int i=0; i <w_row;++i)
   {
     for(int j =0; j<max_elements;++j){
        if(sW_rowPtr[i+1]-sW_rowPtr[i] > j) {
        	ell_w_nnz.push_back(sW_nnz[sW_rowPtr[i]+j]);
        	ell_w_colInd.push_back(sW_colInd[sW_rowPtr[i]+j]);
        }
        else {
        	ell_w_nnz.push_back(0);
        	ell_w_colInd.push_back(0);
        }
     }
    
   }

   //Performing transpose
   for(int i =0; i <max_elements;++i)
     for(int j=0; j <w_row;++j) {
         ell_w_nnzT.push_back(ell_w_nnz[j*max_elements+i]);
         ell_w_colIndT.push_back(ell_w_colInd[j*max_elements+i]);
      }
}


void SparseHopfieldNetwork::CSR_2_JDS( )
{
  //Converting CSR to JDS
   row=sort_indexes(nnzPerRow);

   int jds_nnz=0;
   for(int i=0; i <w_row;++i)
   {
     jds_w_rowPtr.push_back(jds_nnz);
     for(int j =sW_rowPtr[row[i]]; j<sW_rowPtr[row[i]+1];++j){
        jds_w_nnz.push_back(sW_nnz[j]);
        jds_w_colInd.push_back(sW_colInd[j]);
        jds_nnz++;
     }
    
   }
   
   jds_w_rowPtr.push_back(jds_nnz);
}

SparseHopfieldNetwork::SparseHopfieldNetwork(const std::vector<float> &thresholds,
                                             const std::vector<std::vector<float>> &weights,
                                             float weightThreshold) :
    HopfieldNetwork(thresholds, weights),
    weightThreshold(weightThreshold),
    w_size(size), w_col(w_size), w_row(w_size),
    nnz(0), rowPtr(0),
    sW_rowPtr(w_row + 1) {

  // Converting dense weight matrix to sparse matrix
  for(int i=0; i < w_row; ++i)
  {
	sW_rowPtr[i] = rowPtr;
	for(int j=0; j < w_col; ++j)
	{
		if(weights[i][j]*weights[i][j]>weightThreshold*weightThreshold)
		{
                  //cout << "Entering critical" << endl;
			sW_nnz.push_back(weights[i][j]);
			sW_colInd.push_back(j);
			++nnz;	
                        //cout << "Exiting critical" << endl;
		}
	}
        nnzPerRow.push_back(nnz-rowPtr);
	rowPtr=nnz;

  }
  
  sW_rowPtr[w_row] = rowPtr; // Last pointer equal number of NNZ elements

  //Sparse matrix Debuging code
#ifndef NDEBUG
  printf("Percentage of NNZ elements in weight matrix using threshold %f = %f%%\n", weightThreshold,(100.00*nnz/(w_size*w_size)));
#endif
/*
printf("\n   CSR   \n");
   for(int i=0; i < w_row; ++i) {
       for (int f=sW_rowPtr[i]; f<sW_rowPtr[i+1];++f)
           printf("%.2f  ",sW_nnz[f]);
       cout <<endl;
    }
   printf("\n");
   for(int i=0; i < w_row; ++i) {
       for (int f=sW_rowPtr[i]; f<sW_rowPtr[i+1];++f)
           printf("%d  ",sW_colInd[f]);
       cout <<endl;
    }
   for (int f=0; f<w_row+1;++f)
      printf("%d  ",sW_rowPtr[f]);
   printf("\n");


printf("\n   JDS   \n");
   for (int f=0; f<nnz;++f)
      printf("%.2f  ",jds_w_nnz[f]);
   printf("\n");
   for (int f=0; f<nnz;++f)
      printf("%d  ",jds_w_colInd[f]);
   printf("\n");
   for (int f=0; f<w_row+1;++f)
      printf("%d  ",jds_w_rowPtr[f]);
   printf("\n");



printf("\n   ELL   \n");
 
   for(int i=0; i < max_elements; ++i) {
      for (int f=i*w_row; f<(i*w_row+w_row);++f)
          printf("%.2f  ",ell_w_nnzT[f]);
      cout <<endl;
   }
   printf("\n");
   for(int i=0; i < max_elements; ++i) {
      for (int f=i*w_row; f<(i*w_row+w_row);++f)
          printf("%d  ",ell_w_colIndT[f]);
      cout <<endl;
   }
   printf("%d  ",max_elements);
   printf("\n");
*/


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
      if (update != state[i])
        stable = false;
      state[i] = update;
    }
  } while (!stable);

  return state;
}

