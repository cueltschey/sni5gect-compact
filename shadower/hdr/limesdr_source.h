#ifndef LIMESDR_SOURCE_H
#define LIMESDR_SOURCE_H
#include "limesuiteng/LimePlugin.h"
#include "limesuiteng/limesuiteng.hpp"
#include "shadower/hdr/lime_parameter_provider.h"
#include "shadower/hdr/ring_buffer.h"
#include "shadower/hdr/source.h"
class LimeSDRSource final : public Source
{
public:
  LimeSDRSource(std::string device_args, double srate_, double rx_freq, double tx_freq, double rx_gain, double tx_gain);
  bool is_sdr() const override { return true; }
  int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) override;
  int  receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts) override;
  void close() override;
  void set_tx_gain(double gain) override;
  void set_rx_gain(double gain) override;
  void set_tx_srate(double sample_rate) override { set_srate(sample_rate); }
  void set_rx_srate(double sample_rate) override { set_srate(sample_rate); }
  void set_tx_freq(double freq) override;
  void set_rx_freq(double freq) override;

private:
  double                srate;
  srslog::basic_logger& logger = srslog::fetch_basic_logger("LimeSDR");
  LimePluginContext*    lime;
  LimeRuntimeParameters state = {};

  int number_of_channels = 1;

  void set_srate(double sample_rate);

  std::function<void(lime::LogLevel, const std::string&)> log_callback =
      [](lime::LogLevel level, const std::string& msg) { puts(msg.c_str()); };
};

extern "C" {
__attribute__((visibility("default"))) Source* create_source(ShadowerConfig& config)
{
  return new LimeSDRSource(
      config.source_params, config.sample_rate, config.dl_freq, config.ul_freq, config.rx_gain, config.tx_gain);
}
}
#endif // LIMESDR_SOURCE_H