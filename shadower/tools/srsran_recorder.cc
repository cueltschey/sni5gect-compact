#include "srsran/phy/utils/vector.h"
#include "srsran/radio/radio.h"
#include "srsran/radio/rf_buffer.h"
#include "srsran/srslog/logger.h"
#include <fstream>
#include <iostream>
#include <string>
double      center_freq   = 3424.5e6;
double      sample_rate   = 46.08e6;
double      subframe_time = 1e-3;
double      gain          = 40;
std::string output_file   = "output.fc32";
int         num_frames    = 1200000;

int main(int argc, char* argv[])
{
  if (argc > 1) {
    // first argument is center frequency in MHz
    double centerFreqMHz = atof(argv[1]);
    center_freq          = centerFreqMHz * 1e6;
  }
  if (argc > 2) {
    num_frames = atoi(argv[2]);
  }
  if (argc > 3) {
    output_file = argv[3];
  }
  if (argc > 4) {
    // fourth argument is sample rate in MHz
    double sampleRateMHz = atof(argv[4]);
    sample_rate          = sampleRateMHz * 1e6;
  }
  srslog::init();
  static srslog::sink* log_sink = srslog::create_stdout_sink();
  srslog::log_channel* chan     = srslog::create_log_channel("main_channel", *log_sink);
  srslog::set_default_sink(*log_sink);
  srsran::radio     rf;
  srsran::rf_args_t rf_args = {};
  rf_args.type              = "UHD";
  rf_args.log_level         = "DEBUG";
  rf_args.srate_hz          = sample_rate;
  rf_args.dl_freq           = center_freq;
  rf_args.ul_freq           = center_freq;
  rf_args.freq_offset       = 0;
  rf_args.rx_gain           = gain;
  rf_args.tx_gain           = 1;
  rf_args.nof_carriers      = 1;
  rf_args.nof_antennas      = 1;
  rf_args.device_args       = "type=b200,master_clock_rate=46.08e6";
  if (rf.init(rf_args, nullptr) != 0) {
    printf("Failed to init radio\n");
    return -1;
  }
  rf.set_rx_srate(sample_rate);
  rf.set_rx_freq(0, center_freq);
  rf.set_rx_gain(gain);
  rf.set_tx_srate(sample_rate);
  rf.set_tx_freq(0, center_freq);
  rf.set_tx_gain(1);

  std::ofstream out(output_file, std::ios::binary);

  if (!out.is_open()) {
    printf("Failed to open output file\n");
    return -1;
  }

  uint32_t               sf_len    = rf_args.srate_hz * subframe_time;
  cf_t*                  buffer    = srsran_vec_cf_malloc(sf_len);
  srsran::rf_timestamp_t timestamp = {};
  srsran::rf_buffer_t    rf_buffer = {};
  rf_buffer.set(0, buffer);
  rf_buffer.set_nof_samples(sf_len);
  for (uint32_t i = 0; i < num_frames * 10; i++) {
    try {
      if (!rf.rx_now(rf_buffer, timestamp)) {
        printf("Failed to receive samples\n");
        return -1;
      }
      out.write(reinterpret_cast<char*>(buffer), sf_len * sizeof(cf_t));
      printf(".");
      fflush(stdout);
    } catch (const std::exception& e) {
      printf("Exception: %s\n", e.what());
      out.close();
      return -1;
    }
  }
}