#include "shadower/hdr/source.h"

double      sample_rate     = 61.44e6;
double      center_freq     = 3424.5e6;
std::string source_filename = "shadower/test/data/sib1.fc32";
std::string sdr_args        = "type=b200,master_clock_rate=61.44e6,clock=internal";
uint32_t    rx_gain         = 40;
uint32_t    tx_gain         = 80;
float       send_delay      = 2e-3;

int main()
{
  /* initialize logger */
  srslog::basic_logger& logger = srslog_init();
  logger.set_level(srslog::basic_levels::debug);

  /* Test the file source */
  Source*  source;
  uint32_t sf_len                = sample_rate * SF_DURATION;
  uint32_t slot_len              = sf_len / 2;
  source                         = new FileSource(source_filename.c_str(), sample_rate);
  cf_t*              test_buffer = srsran_vec_cf_malloc(sf_len);
  srsran_timestamp_t ts          = {};
  source->receive(test_buffer, sf_len, &ts);
  source->send(test_buffer, sf_len, ts);
  source->close();

  /* Test the SDR source */
  cf_t* tx_buffer = srsran_vec_cf_malloc(sf_len);
  if (!load_samples(source_filename, tx_buffer, sf_len)) {
    logger.error("Error loading samples\n");
    return -1;
  }
  char filename[64];
  source = new SDRSource(sdr_args, sample_rate, center_freq, center_freq, rx_gain, tx_gain);
  for (uint32_t i = 0; i < 50; i++) {
    sprintf(filename, "received_data_%u", i);
    /* Receive the samples again */
    source->receive(test_buffer, sf_len, &ts);
    srsran_timestamp_add(&ts, 0, send_delay);
    /* Send the samples out */
    source->send(tx_buffer + slot_len, slot_len, ts);
    write_record_to_file(test_buffer, sf_len, filename);
  }
  source->close();
}