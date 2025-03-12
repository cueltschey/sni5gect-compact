#ifndef SSB_CUDA_H
#define SSB_CUDA_H
#include "srsran/phy/sync/ssb.h"
#include <cufft.h>

class SSBCuda
{
public:
  SSBCuda(double                      srate_,
          double                      dl_freq_,
          double                      ssb_freq_,
          srsran_subcarrier_spacing_t scs_,
          srsran_ssb_pattern_t        pattern_,
          srsran_duplex_mode_t        duplex_mode_);
  ~SSBCuda();
  bool init(uint32_t N_id_2);
  void cleanup();
  int  ssb_pss_find_cuda(cf_t* in, uint32_t nof_samples, uint32_t* found_delay);

  //   bool ssb_run_sync_find(cf_t* buffer);
  //   bool ssb_run_sync_track(cf_t* buffer);

private:
  srsran_ssb_t                ssb = {};
  double                      srate;
  double                      dl_freq;
  double                      ssb_freq;
  srsran_subcarrier_spacing_t scs;
  srsran_ssb_pattern_t        pattern;
  srsran_duplex_mode_t        duplex_mode;

  cufftHandle   fft_plan = {};
  cudaStream_t  stream; // CUDA stream for asynchronous data transfer
  cufftComplex *h_pin_time = nullptr, *d_freq = nullptr, *d_time = nullptr, *d_corr = nullptr, *d_pss_seq = nullptr;

  float *d_corr_mag = nullptr, *d_power = nullptr;
};
#endif // SSB_CUDA_H