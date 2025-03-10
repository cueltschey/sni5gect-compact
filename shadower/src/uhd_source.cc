#include "shadower/hdr/uhd_source.h"
void UHDSource::close()
{
  srsran_rf_close(&rf);
}

/* Initialize the radio object and apply the configurations */
UHDSource::UHDSource(std::string device_args,
                     double      srate_,
                     double      rx_freq,
                     double      tx_freq,
                     double      rx_gain,
                     double      tx_gain) :
  srate(srate_)
{
  if (srsran_rf_open(&rf, (char*)device_args.c_str()) != 0) {
    throw std::runtime_error("Failed to open radio");
  }

  srsran_rf_set_rx_srate(&rf, srate);
  srsran_rf_set_rx_freq(&rf, 0, rx_freq);
  srsran_rf_set_rx_gain(&rf, rx_gain);

  srsran_rf_set_tx_srate(&rf, srate);
  srsran_rf_set_tx_freq(&rf, 0, tx_freq);
  srsran_rf_set_tx_gain(&rf, tx_gain);
}

int UHDSource::receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts)
{
  try {
    int samples_received = srsran_rf_recv_with_time(&rf, buffer, nof_samples, true, &ts->full_secs, &ts->frac_secs);
    return samples_received;
  } catch (const std::exception& e) {
    return -1;
  }
}

/* TODO implement send the IQ samples at the scheduled time */
int UHDSource::send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot)
{
  std::lock_guard<std::mutex> lock(mutex);
  try {
    int samples_sent = srsran_rf_send_timed2(&rf, samples, length, tx_time.full_secs, tx_time.frac_secs, true, true);
    return samples_sent;
  } catch (const std::exception& e) {
    return -1;
  }
}