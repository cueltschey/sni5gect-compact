#include "shadower/hdr/fft_processor.h"
#include "shadower/hdr/constants.h"
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

  /* Initiaize the input and output buffer */
  input  = srsran_vec_cf_malloc(fft_size);
  output = srsran_vec_cf_malloc(fft_size);

  plan = fft_create_plan(fft_size,
                         reinterpret_cast<liquid_float_complex*>(input),
                         reinterpret_cast<liquid_float_complex*>(output),
                         LIQUID_FFT_FORWARD,
                         0);
}

/* Process the samples from a slot at a time */
void FFTProcessor::process_samples(cf_t* buffer, cf_t* ofdm_symbols, uint32_t slot_idx)
{
  uint32_t start_idx      = slot_idx % slots_per_subframe * symbols_per_slot;
  uint32_t current_offset = 0;
  for (uint32_t i = 0; i < symbols_per_slot; i++) {
    uint32_t idx                  = start_idx + i;
    uint32_t cyclic_prefix_length = cp_length_list[idx];

    /* Copy the data in to input buffer */
    current_offset += cyclic_prefix_length;
    memcpy(input, buffer + current_offset, ofdm_length * sizeof(cf_t));
    // printf("Symbol %u\n", i);
    // for (uint32_t j = 0; j < ofdm_length; j++) {
    //   std::complex<float> srsran_c = reinterpret_cast<std::complex<float>&>(input[j]);
    //   printf("%f + %fi\n", srsran_c.real(), srsran_c.imag());
    // }
    current_offset += ofdm_length;

    /* Execute the FFT */
    fft_execute(plan);

    /* Copy the output to the ofdm_symbols */
    // memcpy(ofdm_symbols + i * fft_size, output, fft_size * sizeof(cf_t));
    memcpy(ofdm_symbols + i * nof_re, output + fft_size - half_subc, half_subc * sizeof(cf_t));
    memcpy(ofdm_symbols + i * nof_re + half_subc, output, half_subc * sizeof(cf_t));
  }
}