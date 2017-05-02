
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

#define WEIGHT_THRESHOLD 0.05
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void gpu_sparse_recall_kernel() {
  // TODO
}

GPUSparseHopfieldNetwork::GPUSparseHopfieldNetwork(const std::vector<float> &thresholds,
                                                   const std::vector<std::vector<float>> &weights,
                                                   float weightThreshold) :
  SparseHopfieldNetwork(thresholds, weights, weightThreshold) {
  // TODO
  //Variables
  bool stable;
  size_t size;
  bool *data_h;
  float *threshold_h;
  float *weight_h;

  bool *stable_d;
  bool *state_d;
  float *threshold_d;
  float *sW_nnz_d;
  int *sW_colInd_d;
  int *sW_rowPtr_d;


  size=data.size();

  //Allocating host memory
  data_h = (bool*)malloc(sizeof(bool) * size);
  //state_h = (bool*)malloc(sizeof(bool) * size);
  threshold_h = (float*)malloc(sizeof(float) * size);
  weight_h = (float*)malloc(sizeof(float) * size * size);

  //Transering Values
  //TODO: Find a better way
  for (size_t i = 0; i < size; ++i) {
    data_h[i] = data[i];
    threshold_h[i] = thresholds[i];

    for (size_t j = 0; j < size; ++j) {
      weight_h[i*size+j] = weights[i][j];
    }
  }

  //   Convering dense   //
  //   weight matrix to  //
  //    Sparse matrix    //

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
		if(weights[i][j]*weights[i][j]>WEIGHT_THRESHOLD*WEIGHT_THRESHOLD)
		{
			sW_nnz.push_back(weights[i][j]);
			sW_colInd.push_back(j);
			++nnz;	
		}
	}
	rowPtr=nnz;
  }
  
  sW_rowPtr.push_back(rowPtr); // Last pointer equal number of NNZ elements
 

  //Allocating device memory
  cudaMalloc((void**)&state_d,sizeof(bool) * size);
  cudaMalloc((void**)&stable_d,sizeof(bool));
  cudaMalloc((void**)&threshold_d,sizeof(float) * size);
  cudaMalloc((void**)&sW_nnz_d,sizeof(float) * nnz);
  cudaMalloc((void**)&sW_colInd_d,sizeof(int) * nnz);
  cudaMalloc((void**)&sW_rowPtr_d,sizeof(int) * (w_row+1));
  

  // Copying data to device
  gpuErrchk(cudaMemcpy(state_d, data_h, size * sizeof(bool),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(threshold_d, threshold_h, size * sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_nnz_d, sW_nnz, nnz*sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_colInd_d, sW_colInd, nnz*sizeof(int),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_rowPtr_d, sW_rowPtr,(w_row+1)*sizeof(int),cudaMemcpyHostToDevice));


}

GPUSparseHopfieldNetwork::~GPUSparseHopfieldNetwork() {

  //Free Device memory
  cudaFree(state_d);
  cudaFree(threshold_d);
  cudaFree(sW_nnz_d);
  cudaFree(sW_colInd_d);
  cudaFree(sW_rowPtr_d);
  cudaFree(stable_d);

}

vector<bool> GPUSparseHopfieldNetwork::evaluate(const vector<bool> &data) {
  // TODO: Implement me!
  assert(false);
  return data;
}
