#include "shadower/hdr/fft_processor.h"
#include "shadower/hdr/utils.h"
#include "test_variables.h"
#include <complex>
#include <fstream>
#include <iostream>
#include <vector>

#if TEST_TYPE == 1
std::string sample_file = "shadower/test/data/srsran/sib.fc32";
#elif TEST_TYPE == 2
std::string sample_file = "shadower/test/data/ssb.fc32";
#endif // TEST_TYPE

int main()
{
  srslog::basic_logger& logger = srslog_init();
  logger.set_level(srslog::basic_levels::debug);
  std::vector<cf_t> samples(sf_len);
  if (!load_samples(sample_file, samples.data(), sf_len)) {
    logger.error("Failed to load data from %s", sample_file.c_str());
    return -1;
  }
  std::cout << "Loaded samples from " << sample_file << std::endl;

  FFTProcessor fft_processor(config.sample_rate, scs, config.nof_prb);
  cf_t*        output_ofdm_symbols = srsran_vec_cf_malloc(fft_processor.fft_size * 14);
  fft_processor.process_samples(samples.data() + slot_len, output_ofdm_symbols, 1);

  char filename[64];
  sprintf(filename, "raw");
  write_record_to_file(samples.data() + slot_len, slot_len, filename);

  // sprintf(filename, "ofdm_output_fft%u", fft_processor.fft_size);
  // write_record_to_file(output_ofdm_symbols, fft_processor.fft_size * 14, filename);

  sprintf(filename, "ofdm_output_fft%u", fft_processor.nof_re);
  write_record_to_file(output_ofdm_symbols, fft_processor.nof_re * 14, filename);
  return 0;
}
