#define THREADS_PER_BLOCK 256
#include "shadower/hdr/ssb_cuda.cuh"
#include <complex>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <vector>

// Kernel for complex conjugate multiplication in the frequency domain
__global__ void complex_conj_mult(cufftComplex* input, cufftComplex* pss_seq, cufftComplex* output, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    cufftComplex a = input[idx];
    cufftComplex b = pss_seq[idx]; // Assume pre-conjugated
    b.y            = -b.y;
    output[idx].x  = a.x * b.x - a.y * b.y;
    output[idx].y  = a.x * b.y + a.y * b.x;
  }
}

// Kernel to compute the absolute squared magnitude (correlation power)
__global__ void compute_power(cufftComplex* input, float* power, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    power[idx] = input[idx].x * input[idx].x + input[idx].y * input[idx].y; // |z|^2
  }
}

// Kernel to normalize the correlation
__global__ void normalize_correlation(float* corr, float* power, int N, float scale_factor)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N && power[idx] > 0) {
    corr[idx] /= (power[idx] * scale_factor);
  }
}

SSBCuda::SSBCuda(double                      srate_,
                 double                      dl_freq_,
                 double                      ssb_freq_,
                 srsran_subcarrier_spacing_t scs_,
                 srsran_ssb_pattern_t        pattern_,
                 srsran_duplex_mode_t        duplex_mode_) :
  srate(srate_), dl_freq(dl_freq_), ssb_freq(ssb_freq_), scs(scs_), pattern(pattern_), duplex_mode(duplex_mode_)
{
}

SSBCuda::~SSBCuda() {}

void SSBCuda::cleanup()
{
  if (d_freq) {
    cudaFree(d_freq);
  }
  if (d_corr) {
    cudaFree(d_corr);
  }
  if (d_pss_seq) {
    cudaFree(d_pss_seq);
  }
  if (d_corr_mag) {
    cudaFree(d_corr_mag);
  }
  if (d_power) {
    cudaFree(d_power);
  }
  if (fft_plan) {
    cufftDestroy(fft_plan);
  }
}

bool SSBCuda::init(uint32_t N_id_2)
{
  srsran_ssb_args_t ssb_args = {};
  ssb_args.max_srate_hz      = srate;
  ssb_args.min_scs           = scs;
  ssb_args.enable_search     = true;
  ssb_args.enable_measure    = true;
  ssb_args.enable_decode     = true;
  if (srsran_ssb_init(&ssb, &ssb_args) != 0) {
    printf("Error initialize ssb\n");
    return false;
  }
  srsran_ssb_cfg_t ssb_cfg = {};
  ssb_cfg.srate_hz         = srate;
  ssb_cfg.center_freq_hz   = dl_freq;
  ssb_cfg.ssb_freq_hz      = ssb_freq;
  ssb_cfg.scs              = scs;
  ssb_cfg.pattern          = pattern;
  ssb_cfg.duplex_mode      = duplex_mode;
  ssb_cfg.periodicity_ms   = 10;
  if (srsran_ssb_set_cfg(&ssb, &ssb_cfg) < SRSRAN_SUCCESS) {
    printf("Error set srsran_ssb_set_cfg\n");
    return false;
  }

  cufftPlan1d(&fft_plan, ssb.corr_sz, CUFFT_C2C, 1);
  cudaMallocHost((void**)&h_pin_time, (ssb.sf_sz + ssb.ssb_sz) * sizeof(cufftComplex));
  cudaMalloc((void**)&d_freq, ssb.corr_sz * sizeof(cufftComplex));
  cudaMalloc((void**)&d_time, ssb.corr_sz * sizeof(cufftComplex));
  cudaMalloc((void**)&d_corr, ssb.corr_sz * sizeof(cufftComplex));
  cudaMalloc((void**)&d_pss_seq, ssb.corr_sz * sizeof(cufftComplex));
  cudaMalloc((void**)&d_corr_mag, ssb.corr_sz * sizeof(float));
  cudaMalloc((void**)&d_power, ssb.corr_sz * sizeof(float));
  cudaMemcpy(d_pss_seq, ssb.pss_seq[N_id_2], ssb.corr_sz * sizeof(cufftComplex), cudaMemcpyHostToDevice);
  cudaStreamCreate(&stream);
  cufftSetStream(fft_plan, stream);
  return true;
}

int SSBCuda::ssb_pss_find_cuda(cf_t* in, uint32_t nof_samples, uint32_t* found_delay)
{
  if (ssb.corr_sz == 0) {
    return -1;
  }
  uint32_t best_delay = 0;
  float    best_corr  = 0;
  uint32_t t_offset   = 0;
  uint32_t total_len  = nof_samples + ssb.ssb_sz;
  memcpy(h_pin_time, h_pin_time + ssb.sf_sz, sizeof(cufftComplex) * ssb.ssb_sz);
  memcpy(h_pin_time + ssb.ssb_sz, in, sizeof(cufftComplex) * nof_samples);
  while ((t_offset + ssb.symbol_sz) < total_len) {
    // Number of samples taken in this iteration
    uint32_t chunk_size = ssb.corr_sz;

    // Detect if the correlation input exceeds the input length, take the maximum amount of samples
    if (t_offset + ssb.corr_sz > total_len) {
      chunk_size = total_len - t_offset;
    }

    // Copy the amount of samples
    cudaMemcpyAsync(d_time, h_pin_time + t_offset, sizeof(cufftComplex) * chunk_size, cudaMemcpyHostToDevice, stream);

    // Append zeros if there's space left
    if (chunk_size < ssb.corr_sz) {
      cudaMemsetAsync(d_time + chunk_size, 0, sizeof(cufftComplex) * (ssb.corr_sz - chunk_size), stream);
    }

    // Perform the FFT covnert to frequncy domain
    cufftExecC2C(fft_plan, d_time, d_freq, CUFFT_FORWARD);

    // Perform correlation between frequency domain and PSS sequence
    int blocks = (ssb.corr_sz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // clang-format off
    complex_conj_mult<<<blocks, THREADS_PER_BLOCK>>>(d_freq, d_pss_seq, d_corr, ssb.corr_sz);
    // clang-format on

    cudaStreamSynchronize(stream);

    cufftExecC2C(fft_plan, d_corr, d_corr, CUFFT_INVERSE);

    cudaStreamSynchronize(stream);

    // clang-format off
    compute_power<<<blocks, THREADS_PER_BLOCK>>>(d_corr, d_corr_mag, ssb.corr_window);
    cudaStreamSynchronize(stream);
    // clang-format on 
    // Find the maximum correlation peak index
    thrust::device_ptr<float> dev_corr_mag(d_corr_mag);
    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_corr_mag, dev_corr_mag +ssb.corr_window);
    uint32_t peak_idx = max_ptr - dev_corr_mag;
    float peak_val;
    cudaMemcpy(&peak_val, d_corr_mag + peak_idx, sizeof(float), cudaMemcpyDeviceToHost);

    if (best_corr < peak_val) {
      best_corr  = peak_val;
      best_delay = peak_idx + t_offset;
    }
    t_offset += ssb.corr_window;
  }
  *found_delay = best_delay;
  return 0;
}