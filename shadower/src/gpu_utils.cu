#include "shadower/hdr/gpu_utils.cuh"
#include <cuda_runtime_api.h>
#include <cufft.h>

// Define CUDA kernel
__global__ void gpu_vec_sc_prod_ccc(cufftComplex* d_signal, cufftComplex* d_phase_list, int fft_size, int symbols_per_slot) {
  int idx        = threadIdx.x + blockIdx.x * blockDim.x;
  int symbol_idx = blockIdx.y;

  if (idx < fft_size) {
    int          index = symbol_idx * fft_size + idx;
    cufftComplex phase = d_phase_list[symbol_idx];

    // Complex multiplication: result = d_signal * phase
    cufftComplex result;
    result.x = d_signal[index].x * phase.x - d_signal[index].y * phase.y;
    result.y = d_signal[index].x * phase.y + d_signal[index].y * phase.x;
    d_signal[index] = result;
  }
}

// Function to launch the kernel
void launch_gpu_vec_sc_prod_ccc(cufftComplex* d_signal, cufftComplex* d_phase_list, int fft_size, int symbols_per_slot) {
  dim3 threadsPerBlock(256);
  dim3 numBlocks((fft_size + threadsPerBlock.x - 1) / threadsPerBlock.x, symbols_per_slot);
  gpu_vec_sc_prod_ccc<<<numBlocks, threadsPerBlock>>>(d_signal, d_phase_list, fft_size, symbols_per_slot);
  cudaDeviceSynchronize();
}
