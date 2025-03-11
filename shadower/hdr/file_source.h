#ifndef FILE_SOURCE_H
#define FILE_SOURCE_H
#include "shadower/hdr/source.h"

class FileSource final : public Source
{
public:
  FileSource(const char* file_name, double sample_rate);
  bool is_sdr() const override { return false; }
  int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) override;
  int  receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts) override;
  void close() override;
  void set_tx_gain(double gain) override {};
  void set_rx_gain(double gain) override {};
  void set_tx_srate(double sample_rate) override {};
  void set_rx_srate(double sample_rate) override {};
  void set_tx_freq(double freq) override {};
  void set_rx_freq(double freq) override {};

private:
  std::ifstream      ifile;
  double             srate;
  srsran_timestamp_t timestamp_prev{};
};

extern "C" {
__attribute__((visibility("default"))) Source* create_source(ShadowerConfig& config)
{
  return new FileSource(config.source_params.c_str(), config.sample_rate);
}
}
#endif // FILE_SOURCE_H