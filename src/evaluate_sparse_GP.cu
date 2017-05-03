
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <time.h>
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}





__global__ void gpu_sparse_gpu_prePro_recall_kernel(size_t size,
                                        bool * state,
                                        float * thresholds,
                                        float * sW_nnz,
                                        int * sW_colInd,
                                        int * sW_rowPtr,
                                        bool * stable) 
{
  // TODO
   size_t node = blockIdx.x * blockDim.x + threadIdx.x;
   if (node < size) {
    float value = 0.0f;
    for (size_t k = sW_rowPtr[node]; k < sW_rowPtr[node+1]; ++k) 
    {	
	if (state[sW_colInd[k]])
		value += sW_nnz[k];
       	else
     		value -= sW_nnz[k];
    }

    bool update = value > thresholds[node];
    if (update != state[node]) {
      *stable = false;
      state[node] = update;
    }
  }
  
}

GPUSparseGpuPreProHopfieldNetwork::GPUSparseGpuPreProHopfieldNetwork(const std::vector<float> &thresholds,
                                                   const std::vector<std::vector<float>> &weights,
                                                   float weightThreshold) :
  SparseHopfieldNetwork(thresholds, weights, weightThreshold) {

  //   Convering dense   //
  //   weight matrix to  //
  //    Sparse matrix    //
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  int w_size = (int)size;
  float *h_w_dense = (float*)malloc(w_size*w_size*sizeof(*h_w_dense));
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      //Make loose connections -> No connection
      h_w_dense[i+j*size] = weights[i][j]*weights[i][j]>weightThreshold*weightThreshold ? weights[i][j] : 0;
    }
  }

  
  gpuErrchk(cudaMalloc(&d_w_dense,w_size*w_size*sizeof(float)));
  gpuErrchk(cudaMemcpy(d_w_dense,h_w_dense,w_size*w_size*sizeof(float),cudaMemcpyHostToDevice));

  cusparseMatDescr_t descrW;
  cusparseCreateMatDescr(&descrW);
  cusparseSetMatType (descrW, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase (descrW, CUSPARSE_INDEX_BASE_ZERO);
  int nnz = 0;
  const int lda = w_size;
  
  gpuErrchk(cudaMalloc(&d_nnzPerVector, w_size*sizeof(int)));

  cusparseSnnz(handle,CUSPARSE_DIRECTION_ROW,w_size,w_size,descrW,d_w_dense,lda,d_nnzPerVector,&nnz);

  int *h_nnzPerVector = (int *) malloc(w_size*sizeof(int));
  gpuErrchk(cudaMemcpy(h_nnzPerVector,d_nnzPerVector,w_size*sizeof(int),cudaMemcpyDeviceToHost));


  printf("Percentage of NNZ elements in weight matrix using threshold %f = %f%%\n", weightThreshold,(100.00*nnz/(w_size*w_size)));
 

  //Allocating device memory
  gpuErrchk(cudaMalloc((void**)&state_d,sizeof(bool) * size));
  gpuErrchk(cudaMalloc((void**)&stable_d,sizeof(bool)));
  gpuErrchk(cudaMalloc((void**)&threshold_d,sizeof(float) * size));
  gpuErrchk(cudaMalloc((void**)&sW_nnz_d,sizeof(float) * nnz));
  gpuErrchk(cudaMalloc((void**)&sW_colInd_d,sizeof(int) * nnz));
  gpuErrchk(cudaMalloc((void**)&sW_rowPtr_d,sizeof(int) * (w_size+1)));
  
  cusparseSdense2csr(handle,w_size,w_size,descrW,d_w_dense,lda,d_nnzPerVector,sW_nnz_d,sW_rowPtr_d,sW_colInd_d);

  // Copying data to device
  gpuErrchk(cudaMemcpy(threshold_d, thresholds.data(), size * sizeof(float),cudaMemcpyHostToDevice));


}

GPUSparseGpuPreProHopfieldNetwork::~GPUSparseGpuPreProHopfieldNetwork() {

  //Free Device memory
  cudaFree(state_d);
  cudaFree(threshold_d);
  cudaFree(sW_nnz_d);
  cudaFree(sW_colInd_d);
  cudaFree(sW_rowPtr_d);
  cudaFree(stable_d);

}

vector<bool> GPUSparseGpuPreProHopfieldNetwork::evaluate(const vector<bool> &data) {
  // TODO: Implement me!

  bool stable_h;
  bool data_h[size];

  unsigned numThreads = 256;
  unsigned numBlocks = (size-1)/numThreads+1;
  copy(data.begin(), data.end(), data_h);
  gpuErrchk(cudaMemcpy(state_d, data_h, size * sizeof(bool),cudaMemcpyHostToDevice));
  do {
    stable_h = true;
    gpuErrchk(cudaMemcpy(stable_d, &stable_h, sizeof(bool),
                         cudaMemcpyHostToDevice));

    gpu_sparse_gpu_prePro_recall_kernel<<< numBlocks, numThreads >>> 
    (size, state_d, threshold_d, sW_nnz_d, sW_colInd_d, sW_rowPtr_d, stable_d);


    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(&stable_h, stable_d, sizeof(bool),
                         cudaMemcpyDeviceToHost));
  } while (!stable_h);

  gpuErrchk(cudaMemcpy(data_h, state_d, size * sizeof(bool),
                       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaDeviceSynchronize());
  
  vector<bool> state(data_h, data_h + size);


  return state;

}
