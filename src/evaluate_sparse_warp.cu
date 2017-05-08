
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
   __shared__ float sh_state[256];
   int tx=threadIdx.x;
   int tID= blockIdx.x*blockDim.x + tx; // Thread ID
   int wID = tID/32;   			// Warp ID
   int lane = tID & (32-1); 		// Lane ID
   int node=wID;
   float value;
   if (node < size) {
    int start_node = sW_rowPtr[node];
    int end_node = sW_rowPtr[node+1];
    sh_state[tx] = 0.0f;
    value =0;
    for (size_t k = start_node+lane; k < end_node; k +=32) 
    {	
	if (state[sW_colInd[k]])
		sh_state[tx] += sW_nnz[k];
       	else
     		sh_state[tx] -= sW_nnz[k];
    }

    //Reduction of addition
/*    if (lane <16) sh_state[tx] += sh_state[tx+16];
    if (lane <8) sh_state[tx] += sh_state[tx+8];
    if (lane <4) sh_state[tx] += sh_state[tx+4];
    if (lane <2) sh_state[tx] += sh_state[tx+2];
    if (lane <1) sh_state[tx] += sh_state[tx+1];
*/
    __syncthreads();
    if (lane ==0) {
       for(int s =0; s<32; ++s)  
         value += sh_state[tx+s];

    bool update = value > thresholds[node];
    if (update != state[node]) {
      *stable = false;
      state[node] = update;
    }
  
   }
  } 
}

GPUSparseGpuPreProHopfieldNetwork::GPUSparseGpuPreProHopfieldNetwork(const std::vector<float> &thresholds,
                                                   const std::vector<std::vector<float>> &weights,
                                                   float weightThreshold) :
  SparseHopfieldNetwork(thresholds, weights, weightThreshold) {
  //Allocating device memory
  gpuErrchk(cudaMalloc((void**)&state_d,sizeof(bool) * size));
  gpuErrchk(cudaMalloc((void**)&stable_d,sizeof(bool)));
  gpuErrchk(cudaMalloc((void**)&threshold_d,sizeof(float) * size));
  gpuErrchk(cudaMalloc((void**)&sW_nnz_d,sizeof(float) * nnz));
  gpuErrchk(cudaMalloc((void**)&sW_colInd_d,sizeof(int) * nnz));
  gpuErrchk(cudaMalloc((void**)&sW_rowPtr_d,sizeof(int) * (w_row+1)));
  

  // Copying data to device
  gpuErrchk(cudaMemcpy(threshold_d, thresholds.data(), size * sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_nnz_d, sW_nnz.data(), nnz*sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_colInd_d, sW_colInd.data(), nnz*sizeof(int),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_rowPtr_d, sW_rowPtr.data(),(w_row+1)*sizeof(int),cudaMemcpyHostToDevice));


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
  unsigned numBlocks = (size*32-1)/numThreads+1;
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
