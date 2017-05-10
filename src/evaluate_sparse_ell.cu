
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
__global__ void gpu_sparse_ell_recall_kernel(size_t size,
                                        bool * state,
                                        float * thresholds,
                                        float * ell_w_nnz,
                                        int * ell_w_colInd,
                                        int max_elements ,
                                        bool * stable) 
{
   size_t node = blockIdx.x * blockDim.x + threadIdx.x;
   if (node < size) {
    float value = 0.0f;

    for (size_t k = node*max_elements; k <node*max_elements+max_elements ; ++k)
    {
        if (state[ell_w_colInd[k]])
                value += ell_w_nnz[k];
        else
                value -= ell_w_nnz[k];
    }

    bool update = value > thresholds[node];
    if (update != state[node]) {
      *stable = false;
      state[node] = update;
    }
  }
 
}

GPUSparseELLHopfieldNetwork::GPUSparseELLHopfieldNetwork(const std::vector<float> &thresholds,
                                                   const std::vector<std::vector<float>> &weights,
                                                   float weightThreshold) :
  SparseHopfieldNetwork(thresholds, weights, weightThreshold) {
  //Converting CSR to ELL
  CSR_2_ELL();
 
  //Allocating device memory
  gpuErrchk(cudaMalloc((void**)&state_d,sizeof(bool) * size));
  gpuErrchk(cudaMalloc((void**)&stable_d,sizeof(bool)));
  gpuErrchk(cudaMalloc((void**)&threshold_d,sizeof(float) * size));
  gpuErrchk(cudaMalloc((void**)&ell_w_nnz_d,sizeof(float) * (max_elements*w_row)));
  gpuErrchk(cudaMalloc((void**)&ell_w_colInd_d,sizeof(int) * (max_elements*w_row)));
  

  // Copying data to device
  gpuErrchk(cudaMemcpy(threshold_d, thresholds.data(), size * sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(ell_w_nnz_d, ell_w_nnz.data(), max_elements*w_row*sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(ell_w_colInd_d, ell_w_colInd.data(), max_elements*w_row*sizeof(int),cudaMemcpyHostToDevice));


}

GPUSparseELLHopfieldNetwork::~GPUSparseELLHopfieldNetwork() {

  //Free Device memory
  cudaFree(state_d);
  cudaFree(threshold_d);
  cudaFree(ell_w_nnz_d);
  cudaFree(ell_w_colInd_d);
  cudaFree(stable_d);

}

vector<bool> GPUSparseELLHopfieldNetwork::evaluate(const vector<bool> &data) {

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

    gpu_sparse_ell_recall_kernel<<< numBlocks, numThreads >>> 
    (size, state_d, threshold_d, ell_w_nnz_d, ell_w_colInd_d, max_elements, stable_d);

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
