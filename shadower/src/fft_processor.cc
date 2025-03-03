#include "shadower/hdr/fft_processor.h"
#include "shadower/hdr/constants.h"
#include <chrono>
#include <cmath>
FFTProcessor::FFTProcessor(double sample_rate_, srsran_subcarrier_spacing_t scs_, uint32_t num_prbs_) :
  sample_rate(sample_rate_),
  scs(scs_),
  two_pow_numerology(1 << scs),
  scs_khz((1 << scs) * 15),
  nof_re(num_prbs_ * SRSRAN_NRE),
  slots_per_subframe(1 << scs),
  sf_len(sample_rate * SF_DURATION),
  symbols_per_subframe(symbols_per_slot * slots_per_subframe)
{
  fft_size  = sf_len / scs_khz;
  half_fft  = fft_size / 2;
  half_subc = nof_re / 2;
  /* Calculate the duration in the unit of Tc */
  ofdm_units    = 2048.0 * K * 1.0 / two_pow_numerology;
  cp_units      = 144.0 * K * 1.0 / two_pow_numerology;
  long_cp_units = (144.0 + 16.0) * K * 1.0 / two_pow_numerology;

  /* Calculate the duration in seconds */
  ofdm_duration      = ofdm_units * Tc;
  normal_cp_duration = cp_units * Tc;
  long_cp_duration   = long_cp_units * Tc;

  /* Calculate the cyclic prefix length */
  ofdm_length    = std::floor(ofdm_duration * sample_rate);
  cp_length      = std::floor(normal_cp_duration * sample_rate);
  long_cp_length = std::floor(long_cp_duration * sample_rate);

  /* Initialize the cyclic prefix */
  cp_length_list.resize(symbols_per_subframe, cp_length);
  // Long CP list for 0 and 7 * 2^(miu - 1)
  cp_length_list[0]                      = long_cp_length;
  cp_length_list[7 * two_pow_numerology] = long_cp_length;
  cudaError_t error = cudaMalloc((void**)&d_signal, symbols_per_slot * fft_size * sizeof(cufftComplex));
  if (error != cudaError::cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }

  cudaError_t error2 = cudaMallocHost((void**)&h_pinned_buffer, symbols_per_slot * fft_size * sizeof(cufftComplex));
  if (error2 != cudaError::cudaSuccess) {
    throw std::runtime_error("cudaMallocHost failed");
  }
  cufftResult result = cufftPlan1d(&plan, fft_size, CUFFT_C2C, symbols_per_slot);
  if (result != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT error: Plan creation failed");
  }

  cudaStreamCreate(&stream);
  cufftSetStream(plan, stream);
}

/* Process the samples from a slot at a time */
void FFTProcessor::process_samples(cf_t* buffer, cf_t* ofdm_symbols, uint32_t slot_idx)
{
  uint32_t start_idx      = slot_idx % slots_per_subframe * symbols_per_slot;
  uint32_t current_offset = 0;
  uint32_t idx, cyclic_prefix_length;

  // Use regular memcpy instead of cudaMemcpyHostToHost
  for (uint32_t i = 0; i < symbols_per_slot; i++) {
    cyclic_prefix_length = cp_length_list[start_idx + i];
    current_offset += cyclic_prefix_length; // remove the cyclic prefix
    memcpy(h_pinned_buffer + i * fft_size, buffer + current_offset, sizeof(cufftComplex) * fft_size);
    current_offset += ofdm_length; // proceeds after processing the OFDM symbol
  }

  // Asynchronous transfer to GPU
  cudaMemcpyAsync(
      d_signal, h_pinned_buffer, symbols_per_slot * fft_size * sizeof(cufftComplex), cudaMemcpyHostToDevice, stream);

  // Run fft
  cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);

  // Copy result back asynchronously
  cudaMemcpyAsync(
      h_pinned_buffer, d_signal, symbols_per_slot * fft_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost, stream);

  // Wait for all operations to complete
  cudaStreamSynchronize(stream);

  // Copy final output back to host
  for (uint32_t i = 0; i < symbols_per_slot; i++) {
    // Copy the result to OFDM symbols
    memcpy(ofdm_symbols + i * nof_re, h_pinned_buffer + i * fft_size + fft_size - half_subc, half_subc * sizeof(cf_t));
    memcpy(ofdm_symbols + i * nof_re + half_subc, h_pinned_buffer + i * fft_size, half_subc * sizeof(cf_t));
  }
}