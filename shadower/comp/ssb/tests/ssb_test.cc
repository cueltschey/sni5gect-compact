#if ENABLE_CUDA
#include "shadower/comp/ssb/ssb_cuda.cuh"
#endif // ENABLE_CUDA
#include "shadower/utils/arg_parser.h"
#include "shadower/utils/constants.h"
#include "shadower/utils/ssb_utils.h"
#include "shadower/utils/utils.h"
#include "srsran/phy/phch/pbch_msg_nr.h"
#include "srsran/phy/sync/ssb.h"
#include "srsran/srslog/srslog.h"
#include "srsran/support/srsran_test.h"
#include <getopt.h>

ShadowerConfig config = {};
std::string    sample_file;
uint32_t       cell_id = 1;
int            band    = 78;
uint32_t       rounds  = 1000;

srsran_subcarrier_spacing_t scs = srsran_subcarrier_spacing_30kHz;

void parse_args(int argc, char* argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "sfbFIrBS")) != -1) {
    switch (opt) {
      case 's': {
        double srateMHz    = atof(argv[optind]);
        config.sample_rate = srateMHz * 1e6;
        printf("Using sample rate: %f MHz\n", srateMHz);
        break;
      }
      case 'f': {
        double freqMHz = atof(argv[optind]);
        config.dl_freq = freqMHz * 1e6;
        config.ul_freq = config.dl_freq;
        printf("Using center frequency: %f MHz\n", config.dl_freq);
        break;
      }
      case 'b': {
        double ssbFreqMHz = atof(argv[optind]);
        config.ssb_freq   = ssbFreqMHz * 1e6;
        printf("Using SSB Frequency: %f MHz\n", config.ssb_freq);
        break;
      }
      case 'B': {
        band = atoi(argv[optind]);
        printf("Using band: %d\n", band);
        config.band = band;
        break;
      }
      case 'S': {
        scs = srsran_subcarrier_spacing_from_str(argv[optind]);
        break;
      }
      case 'F': {
        sample_file = argv[optind];
        printf("Using sample file: %s\n", sample_file.c_str());
        break;
      }
      case 'I': {
        cell_id = atoi(argv[optind]);
        printf("Using cell id: %u\n", cell_id);
        break;
      }
      case 'r': {
        rounds = atoi(argv[optind]);
        printf("Round each test for: %u rounds\n", rounds);
        break;
      }
      default:
        fprintf(stderr, "Unknown option or missing argument.\n");
        exit(EXIT_FAILURE);
    }
  }
  srsran::srsran_band_helper helper;
  config.ssb_pattern = helper.get_ssb_pattern(band, scs);
  config.duplex_mode = helper.get_duplex_mode(band);
  config.scs_ssb     = scs;
  if (sample_file.empty()) {
    fprintf(stderr, "Sample file is required.\n");
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
  parse_args(argc, argv);
  /* initialize logger */
  config.log_level             = srslog::basic_levels::debug;
  srslog::basic_logger& logger = srslog_init(&config);
  uint32_t              sf_len = config.sample_rate * SF_DURATION;

  /* load IQ samples from file */
  std::vector<cf_t> samples(sf_len);
  if (!load_samples(sample_file, samples.data(), sf_len)) {
    logger.error("Failed to load data from %s", sample_file.c_str());
    return -1;
  }

  srsran_ssb_t                  ssb         = {};
  srsran_csi_trs_measurements_t measurement = {};
  srsran_pbch_msg_nr_t          pbch_msg    = {};

  if (!init_ssb(ssb,
                config.sample_rate,
                config.dl_freq,
                config.ssb_freq,
                config.scs_ssb,
                config.ssb_pattern,
                config.duplex_mode)) {
    logger.error("Failed to initialize SSB");
    return -1;
  }

  /* srsran_ssb_find evaluation */
  auto t_start = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < rounds; i++) {
    if (srsran_ssb_find(&ssb, samples.data(), cell_id, &measurement, &pbch_msg) != SRSRAN_SUCCESS) {
      logger.error("Error running srsran_ssb_find");
      return -1;
    }
  }
  auto t_end      = std::chrono::high_resolution_clock::now();
  auto t_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / rounds;
  logger.info("srsran_ssb_find: %ld us", t_duration);
  TESTASSERT(pbch_msg.crc == true);

  /* run SSB track test */
  t_start = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < rounds; i++) {
    if (srsran_ssb_track(&ssb, samples.data(), cell_id, 0, 0, &measurement, &pbch_msg) != SRSRAN_SUCCESS) {
      logger.error("Error running srsran_ssb_track");
      return -1;
    }
  }
  t_end      = std::chrono::high_resolution_clock::now();
  t_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / rounds;
  logger.info("srsran_ssb_track: %ld us", t_duration);
  TESTASSERT(pbch_msg.crc == true);

#if ENABLE_CUDA
  /* run SSB cuda find test */
  SSBCuda ssb_cuda(
      config.sample_rate, config.dl_freq, config.ssb_freq, config.scs_ssb, config.ssb_pattern, config.duplex_mode);
  if (!ssb_cuda.init(SRSRAN_NID_2_NR(cell_id))) {
    logger.error("Failed to initialize SSB CUDA");
    return -1;
  }

  t_start = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < rounds; i++) {
    if (ssb_cuda.ssb_run_sync_find((cf_t*)samples.data(), cell_id, &measurement, &pbch_msg) != SRSRAN_SUCCESS) {
      logger.error("Error running ssb_cuda.ssb_run_sync_find");
      return -1;
    }
  }
  t_end      = std::chrono::high_resolution_clock::now();
  t_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / rounds;
  logger.info("ssb_cuda.ssb_run_sync_find: %ld us", t_duration);
  ssb_cuda.cleanup();
#endif // ENABLE_CUDA
  return 0;
}