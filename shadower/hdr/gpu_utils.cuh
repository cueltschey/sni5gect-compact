#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cufft.h>

void launch_gpu_vec_sc_prod_ccc(cufftComplex* d_signal, cufftComplex* d_phase_list, int fft_size, int symbols_per_slot);

#endif // GPU_UTILS_H