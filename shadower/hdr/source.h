#ifndef SOURCE_H
#define SOURCE_H
#include "shadower/hdr/utils.h"
#include "srsran/radio/radio.h"
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/thread.hpp>
#include <vector>
class Source
{
public:
  virtual bool is_sdr() const { return false; }
  virtual int  receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts)                  = 0;
  virtual int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) = 0;
  virtual void close()                                                                              = 0;
  virtual void set_tx_gain(double gain)                                                             = 0;
  virtual void set_rx_gain(double gain)                                                             = 0;
};

class FileSource final : public Source
{
public:
  FileSource(const char* file_name, double sample_rate);
  bool is_sdr() const override { return false; }
  int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) override;
  int  receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts) override;
  void close() override;
  void set_tx_gain(double gain) override {}
  void set_rx_gain(double gain) override {}

private:
  std::ifstream      ifile;
  double             srate;
  srsran_timestamp_t timestamp_prev{};
};

class SDRSource final : public Source
{
public:
  SDRSource(const std::string& device_args,
            double             srate_,
            double             rx_freq,
            double             tx_freq,
            double             rx_gain,
            double             tx_gain);
  bool is_sdr() const override { return true; }
  int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) override;
  int  receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts) override;
  void close() override;
  void set_tx_gain(double gain) override { srsran_rf_set_tx_gain(&rf, gain); }
  void set_rx_gain(double gain) override { srsran_rf_set_rx_gain(&rf, gain); }

private:
  srsran_rf_t rf{};
  std::mutex  mutex;
  double      srate;
};

#endif // SOURCE_H