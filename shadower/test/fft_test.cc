#include "shadower/hdr/fft_processor.h"
#include "shadower/hdr/utils.h"
#include "srsran/common/phy_cfg_nr.h"
#include "test_variables.h"
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <vector>

#if TEST_TYPE == 1
std::string sample_file = "shadower/test/data/srsran/sib.fc32";
#elif TEST_TYPE == 2
std::string sample_file = "shadower/test/data/ssb.fc32";
#elif TEST_TYPE == 3
std::string sample_file = "shadower/test/data/srsran/sib.fc32";
#endif // TEST_TYPE
uint32_t test_round = 1;

int main()
{
  srslog::basic_logger& logger = srslog_init();
  logger.set_level(srslog::basic_levels::debug);

  /* initialize phy cfg */
  srsran::phy_cfg_nr_t phy_cfg = {};
  init_phy_cfg(phy_cfg, config);
  srsran_slot_cfg_t slot_cfg = {.idx = 1};

  /* UE DL init with configuration from phy_cfg */
  srsran_ue_dl_nr_t ue_dl  = {};
  cf_t*             buffer = srsran_vec_cf_malloc(sf_len);
  if (!init_ue_dl(ue_dl, buffer, phy_cfg)) {
    logger.error("Failed to init UE DL");
    return -1;
  }

  std::vector<cf_t> samples(sf_len);
  uint32_t          filesize = std::min(sf_len, (uint32_t)23040);
  if (!load_samples(sample_file, samples.data(), filesize)) {
    logger.error("Failed to load data from %s", sample_file.c_str());
    return -1;
  }
  std::cout << "Loaded samples from " << sample_file << std::endl;

  /* CPU processing of samples */
  auto start_cpu = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < 10000; i++) {
    srsran_ue_dl_nr_estimate_fft(&ue_dl, &slot_cfg);
  }
  auto                          end_cpu             = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds_cpu = end_cpu - start_cpu;
  std::cout << "CPU time: " << elapsed_seconds_cpu.count() << "s\n";

  FFTProcessor fft_processor(config.sample_rate, scs, config.nof_prb);
  cf_t*        output_ofdm_symbols = srsran_vec_cf_malloc(fft_processor.fft_size * 14);

  auto start_gpu = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < 10000; i++) {
    fft_processor.process_samples(samples.data() + slot_len, output_ofdm_symbols, 1);
  }
  auto                          end_gpu             = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds_gpu = end_gpu - start_gpu;
  std::cout << "GPU time: " << elapsed_seconds_gpu.count() << "s\n";

  char filename[64];
  sprintf(filename, "raw");
  write_record_to_file(samples.data() + slot_len, slot_len, filename);

  // sprintf(filename, "ofdm_output_fft%u", fft_processor.fft_size);
  // write_record_to_file(output_ofdm_symbols, fft_processor.fft_size * 14, filename);

  sprintf(filename, "ofdm_output_fft%u", fft_processor.nof_re);
  write_record_to_file(output_ofdm_symbols, fft_processor.nof_re * 14, filename);
  return 0;
}
