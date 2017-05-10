
#include "hopfield.hpp"

#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
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
__global__ void gpu_sparse_jds_recall_kernel(size_t size,
                                        bool * state,
                                        float * thresholds,
                                        float * jds_w_nnz,
                                        int * jds_w_colInd,
                                        int * jds_w_rowPtr,
					int * row,
                                        bool * stable) 
{
   size_t node = blockIdx.x * blockDim.x + threadIdx.x;
   int org_node = row[node];
   if (node < size) {
    float value = 0.0f;

   for (size_t k = jds_w_rowPtr[node]; k < jds_w_rowPtr[node+1]; ++k) 
    {			
	if (state[jds_w_colInd[k]])
		value += jds_w_nnz[k];
       	else
     		value -= jds_w_nnz[k];
    }

    bool update = value > thresholds[org_node];
    if (update != state[org_node]) {
      *stable = false;
      state[org_node] = update;
    }
  }
  
}

GPUSparseJDSHopfieldNetwork::GPUSparseJDSHopfieldNetwork(const std::vector<float> &thresholds,
                                                   const std::vector<std::vector<float>> &weights,
                                                   float weightThreshold) :
  SparseHopfieldNetwork(thresholds, weights, weightThreshold) {


  //Converting CSR to JDS sparse matrix
  CSR_2_JDS();


  //Allocating device memory
  gpuErrchk(cudaMalloc((void**)&state_d,sizeof(bool) * size));
  gpuErrchk(cudaMalloc((void**)&stable_d,sizeof(bool)));
  gpuErrchk(cudaMalloc((void**)&threshold_d,sizeof(float) * size));
  gpuErrchk(cudaMalloc((void**)&jds_w_nnz_d,sizeof(float) * nnz));
  gpuErrchk(cudaMalloc((void**)&jds_w_colInd_d,sizeof(int) * nnz));
  gpuErrchk(cudaMalloc((void**)&jds_w_rowPtr_d,sizeof(int) * (w_row+1)));
  gpuErrchk(cudaMalloc((void**)&row_d,sizeof(int) * (w_row)));
  

  // Copying data to device
  gpuErrchk(cudaMemcpy(threshold_d, thresholds.data(), size * sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(row_d, row.data(),w_row*sizeof(int),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(jds_w_nnz_d, jds_w_nnz.data(), nnz*sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(jds_w_colInd_d, jds_w_colInd.data(), nnz*sizeof(int),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(jds_w_rowPtr_d, jds_w_rowPtr.data(),(w_row+1)*sizeof(int),cudaMemcpyHostToDevice));


}

GPUSparseJDSHopfieldNetwork::~GPUSparseJDSHopfieldNetwork() {

  //Free Device memory
  cudaFree(state_d);
  cudaFree(threshold_d);
  cudaFree(jds_w_nnz_d);
  cudaFree(jds_w_colInd_d);
  cudaFree(jds_w_rowPtr_d);
  cudaFree(stable_d);

}

vector<bool> GPUSparseJDSHopfieldNetwork::evaluate(const vector<bool> &data) {

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

    gpu_sparse_jds_recall_kernel<<< numBlocks, numThreads >>> 
    (size, state_d, threshold_d, jds_w_nnz_d, jds_w_colInd_d, jds_w_rowPtr_d,row_d, stable_d);


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
