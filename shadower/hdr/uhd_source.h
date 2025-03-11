#ifndef UHD_SOURCE_H
#define UHD_SOURCE_H
#include "shadower/hdr/source.h"
#include "srsran/radio/radio.h"
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/thread.hpp>

class UHDSource final : public Source
{
public:
  UHDSource(std::string device_args, double srate_, double rx_freq, double tx_freq, double rx_gain, double tx_gain);
  bool is_sdr() const override { return true; }
  int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) override;
  int  receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts) override;
  void close() override;
  void set_srate(double srate_) { srate = srate_; }
  void set_tx_gain(double gain) override { srsran_rf_set_tx_gain(&rf, gain); }
  void set_rx_gain(double gain) override { srsran_rf_set_rx_gain(&rf, gain); }
  void set_tx_srate(double sample_rate) override { srsran_rf_set_tx_srate(&rf, sample_rate); }
  void set_rx_srate(double sample_rate) override { srsran_rf_set_rx_srate(&rf, sample_rate); }
  void set_tx_freq(double freq) override { srsran_rf_set_tx_freq(&rf, 0, freq); }
  void set_rx_freq(double freq) override { srsran_rf_set_rx_freq(&rf, 0, freq); }

private:
  srsran_rf_t rf{};
  std::mutex  mutex;
  double      srate;
};

extern "C" {
__attribute__((visibility("default"))) Source* create_source(ShadowerConfig& config)
{
  return new UHDSource(
      config.source_params, config.sample_rate, config.dl_freq, config.ul_freq, config.rx_gain, config.tx_gain);
}
}
#endif // UHD_SOURCE_H