
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
__global__ void gpu_sparse_recall_kernel(size_t size,
                                        bool * state,
                                        float * thresholds,
                                        float * sW_nnz,
                                        int * sW_colInd,
                                        int * sW_rowPtr,
                                        bool * stable,
					int * nodePtr) 
{
  
   int node;
   do {
     node = atomicAdd(nodePtr,1);
     if(node < size) {
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
  } while(node<(int)size);
  
}

GPUSparseQueueHopfieldNetwork::GPUSparseQueueHopfieldNetwork(const std::vector<float> &thresholds,
                                                   const std::vector<std::vector<float>> &weights,
                                                   float weightThreshold) :
  SparseHopfieldNetwork(thresholds, weights, weightThreshold) {

  //   Convering dense   //
  //   weight matrix to  //
  //    Sparse matrix    //
  
  gpuErrchk(cudaMalloc((void**)&nodePtr,sizeof(int)));

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
  printf("Percentage of NNZ elements in weight matrix using threshold %f = %f%%\n", weightThreshold,(100.00*nnz/(w_size*w_size)));
 

  //Allocating device memory
  cudaMalloc((void**)&state_d,sizeof(bool) * size);
  cudaMalloc((void**)&stable_d,sizeof(bool));
  cudaMalloc((void**)&threshold_d,sizeof(float) * size);
  cudaMalloc((void**)&sW_nnz_d,sizeof(float) * nnz);
  cudaMalloc((void**)&sW_colInd_d,sizeof(int) * nnz);
  cudaMalloc((void**)&sW_rowPtr_d,sizeof(int) * (w_row+1));
  

  // Copying data to device
  gpuErrchk(cudaMemcpy(threshold_d, thresholds.data(), size * sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_nnz_d, sW_nnz.data(), nnz*sizeof(float),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_colInd_d, sW_colInd.data(), nnz*sizeof(int),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sW_rowPtr_d, sW_rowPtr.data(),(w_row+1)*sizeof(int),cudaMemcpyHostToDevice));


}

GPUSparseQueueHopfieldNetwork::~GPUSparseQueueHopfieldNetwork() {

  //Free Device memory
  cudaFree(state_d);
  cudaFree(threshold_d);
  cudaFree(sW_nnz_d);
  cudaFree(sW_colInd_d);
  cudaFree(sW_rowPtr_d);
  cudaFree(stable_d); 
  cudaFree(nodePtr);

}

vector<bool> GPUSparseQueueHopfieldNetwork::evaluate(const vector<bool> &data) {
  // TODO: Implement me!

  bool stable_h;
  int nodePtr_h;
  bool data_h[size];

  unsigned numThreads = 256;
  unsigned numBlocks = (size-1)/numThreads+1;

  copy(data.begin(), data.end(), data_h);
  gpuErrchk(cudaMemcpy(state_d, data_h, size * sizeof(bool),cudaMemcpyHostToDevice));

  do {
    stable_h = true;
    gpuErrchk(cudaMemcpy(stable_d, &stable_h, sizeof(bool),
                         cudaMemcpyHostToDevice));


    nodePtr_h=0;
    gpuErrchk(cudaMemcpy(nodePtr,&nodePtr_h, sizeof(int),cudaMemcpyHostToDevice));



    gpu_sparse_recall_kernel<<< numBlocks, numThreads >>> (size, state_d, threshold_d, sW_nnz_d, sW_colInd_d, sW_rowPtr_d, stable_d,nodePtr);

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
