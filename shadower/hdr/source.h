#ifndef SOURCE_H
#define SOURCE_H
#include "limesuiteng/LimePlugin.h"
#include "limesuiteng/limesuiteng.hpp"
#include "shadower/hdr/constants.h"
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

class UHDSource final : public Source
{
public:
  UHDSource(std::string device_args, double srate_, double rx_freq, double tx_freq, double rx_gain, double tx_gain);
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

class LimeSDRSource final : public Source
{
public:
  LimeSDRSource(std::string device_args, double srate_, double rx_freq, double tx_freq, double rx_gain, double tx_gain);
  int  send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot = 0) override;
  int  receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts) override;
  void close() override;
  void set_tx_gain(double gain) override {}
  void set_rx_gain(double gain) override {}

private:
  double                             srate;
  srslog::basic_logger&              logger = srslog::fetch_basic_logger("LimeSDR");
  std::map<std::string, std::string> parse_device_args(const std::string& device_args);
  void                               ConnectToFilteredOrDefaultDevice(std::map<std::string, std::string>& args);
  lime::SDRDevice*                   device;
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

private:
  std::unique_ptr<Source> source;
};

class srsRAN_ParamProvider : public LimeSettingsProvider
{
private:
  static std::string trim(const std::string& s)
  {
    std::string out = s;
    while (!out.empty() && std::isspace(out[0]))
      out = out.substr(1);
    while (!out.empty() && std::isspace(out[out.size() - 1]))
      out = out.substr(0, out.size() - 1);
    return out;
  }

  void argsToMap(const std::string& args)
  {
    bool        inKey = true;
    std::string key, val;
    for (size_t i = 0; i < args.size(); i++) {
      const char ch = args[i];
      if (inKey) {
        if (ch == ':')
          inKey = false;
        else if (ch == ',')
          inKey = true;
        else
          key += ch;
      } else {
        if (ch == ',')
          inKey = true;
        else
          val += ch;
      }
      if ((inKey && !val.empty()) || ((i + 1) == args.size())) {
        key = trim(key);
        val = trim(val);
        printf("Key:Value{ %s:%s }\n", key.c_str(), val.c_str());
        if (!key.empty()) {
          if (val[0] == '"')
            strings[key] = val.substr(1, val.size() - 2);
          else
            numbers[key] = stod(val);
        }
        key = "";
        val = "";
      }
    }
  }

public:
  srsRAN_ParamProvider(const char* args) : mArgs(args) { argsToMap(mArgs); }

  bool GetString(std::string& dest, const char* varname) override
  {
    auto iter = strings.find(std::string(varname));
    if (iter == strings.end())
      return false;

    printf("provided: %s\n", varname);

    dest = iter->second;
    return true;
  }

  bool GetDouble(double& dest, const char* varname) override
  {
    auto iter = numbers.find(varname);
    if (iter == numbers.end())
      return false;

    dest = iter->second;
    return true;
  }

private:
  std::string                                  mArgs;
  std::unordered_map<std::string, double>      numbers;
  std::unordered_map<std::string, std::string> strings;
};

#endif // SOURCE_H