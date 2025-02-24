#ifndef SOURCE_H
#define SOURCE_H
#include "limesuiteng/LimePlugin.h"
#include "limesuiteng/limesuiteng.hpp"
#include "shadower/hdr/constants.h"
#include "shadower/hdr/lime_parameter_provider.h"
#include "shadower/hdr/ring_buffer.h"
#include "shadower/hdr/utils.h"
#include "srsran/radio/radio.h"
#include "srsran/srslog/srslog.h"
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/thread.hpp>
#include <vector>
class Source
{
public:
  virtual ~Source() = default;
  virtual bool is_sdr() const { return false; }
  virtual int  receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts)                  = 0;
  virtual int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) = 0;
  virtual void close()                                                                              = 0;
  virtual void set_tx_gain(double gain)                                                             = 0;
  virtual void set_rx_gain(double gain)                                                             = 0;
  virtual void set_tx_srate(double sample_rate)                                                     = 0;
  virtual void set_rx_srate(double sample_rate)                                                     = 0;
  virtual void set_tx_freq(double freq)                                                             = 0;
  virtual void set_rx_freq(double freq)                                                             = 0;
};

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

class UHDSource final : public Source
{
public:
  UHDSource(std::string device_args, double srate_, double rx_freq, double tx_freq, double rx_gain, double tx_gain);
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

class LimeSDRSource final : public Source
{
public:
  LimeSDRSource(std::string device_args, double srate_, double rx_freq, double tx_freq, double rx_gain, double tx_gain);
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
  srslog::basic_logger& logger    = srslog::fetch_basic_logger("LimeSDR");
  LimePluginContext*    lime;
  LimeRuntimeParameters state = {};

  int number_of_channels = 1;

  void set_srate(double sample_rate);

  std::function<void(lime::LogLevel, const std::string&)> log_callback =
      [this](lime::LogLevel level, const std::string& msg) { logger.info("%s", msg.c_str()); };
};

class SDRSource final : public Source
{
public:
  SDRSource(const std::string& device_args,
            double             srate_,
            double             rx_freq,
            double             tx_freq,
            double             rx_gain,
            double             tx_gain,
            std::string        device_name = "UHD")
  {
    if (device_name == "UHD") {
      source = std::make_unique<UHDSource>(device_args, srate_, rx_freq, tx_freq, rx_gain, tx_gain);
    } else if (device_name == "LimeSDR") {
      source = std::make_unique<LimeSDRSource>(device_args, srate_, rx_freq, tx_freq, rx_gain, tx_gain);
    }
  }
  bool is_sdr() const override { return true; }
  int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) override
  {
    return source->send(samples, length, tx_time, slot);
  };
  int receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts) override
  {
    return source->receive(buffer, nof_samples, ts);
  }
  void close() override { source->close(); };
  void set_tx_gain(double gain) override { source->set_tx_gain(gain); };
  void set_rx_gain(double gain) override { source->set_rx_gain(gain); };
  void set_tx_srate(double sample_rate) override { source->set_tx_srate(sample_rate); };
  void set_rx_srate(double sample_rate) override { source->set_rx_srate(sample_rate); };
  void set_tx_freq(double freq) override { source->set_tx_freq(freq); };
  void set_rx_freq(double freq) override { source->set_rx_freq(freq); };

private:
  std::unique_ptr<Source> source;
};

#endif // SOURCE_H