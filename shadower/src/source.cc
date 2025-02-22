#include "shadower/hdr/source.h"
#include <sstream>
FileSource::FileSource(const char* file_name, double sample_rate) :
  ifile{file_name, std::ifstream::binary}, srate(sample_rate)
{
  if (!ifile.is_open()) {
    throw std::runtime_error("Error opening file");
  }
  timestamp_prev = {0, 0};
}

void FileSource::close()
{
  if (ifile.is_open()) {
    ifile.close();
  }
}

/* Fake send write the samples to send into file */
int FileSource::send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot)
{
  char filename[256];
  sprintf(filename, "tx_slot_%u", slot);
  write_record_to_file(samples, length, filename, "records");
  return length;
}

/* Read the IQ samples from the file, and proceed the timestamp with number of samples / sample rate */
int FileSource::receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts)
{
  ifile.read(reinterpret_cast<char*>(buffer), nof_samples * sizeof(cf_t));
  srsran_timestamp_add(&timestamp_prev, 0, nof_samples / srate);
  srsran_timestamp_copy(ts, &timestamp_prev);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (ifile.eof()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return -1;
  }
  return nof_samples;
}

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
  // Add quirks for USRPB210
  std::string n_device_args = device_args;
  if (device_args.find("b200") != std::string::npos) {
    n_device_args += ",master_clock_rate=" + std::to_string(srate_);
  }

  if (srsran_rf_open(&rf, (char*)n_device_args.c_str()) != 0) {
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

static void LimeLogCallback(lime::LogLevel level, const std::string& msg)
{
  // if (level > lime::LogLevel::Info) {
  //   return;
  // }
  printf(GREEN "[LimeSDR]" RESET " %s\n", msg.c_str());
}

/* Initialize the radio object and apply the configurations */
LimeSDRSource::LimeSDRSource(std::string device_args,
                             double      srate_,
                             double      rx_freq,
                             double      tx_freq,
                             double      rx_gain,
                             double      tx_gain) :
  srate(srate_)
{
  std::map<std::string, std::string> args = parse_device_args(device_args);
  for (auto& arg : args) {
    LimeLogCallback(lime::LogLevel::Info, "Device argument " + arg.first + ": " + arg.second);
  }
  lime::registerLogHandler(LimeLogCallback);
  ConnectToFilteredOrDefaultDevice(args);
  if (!device) {
    throw std::runtime_error("Failed to connect to device");
  }
  LimeLogCallback(lime::LogLevel::Info, "Connected to device: " + device->GetDescriptor().name);
  device->SetMessageLogCallback(LimeLogCallback);
  device->Init();
  printf("srate: %f\n", srate);
}

void LimeSDRSource::ConnectToFilteredOrDefaultDevice(std::map<std::string, std::string>& args)
{
  // Enumerate available devices
  auto handles = lime::DeviceRegistry::enumerate();
  if (handles.empty()) {
    LimeLogCallback(lime::LogLevel::Error, "No LimeSDR devices found");
    device = nullptr;
    return;
  }

  // Show info about all available devices
  for (size_t i = 0; i < handles.size(); i++) {
    LimeLogCallback(lime::LogLevel::Info, "Device " + std::to_string(i) + " available: " + handles[i].Serialize());
  }

  // Find the device based on the serial
  lime::SDRDevice* dev          = nullptr;
  bool             found_target = false;
  for (size_t i = 0; i < handles.size(); i++) {
    // If serial number matches
    if (args.find("serial") != args.end()) {
      if (handles[i].serial == args["serial"]) {
        dev          = lime::DeviceRegistry::makeDevice(handles[i]);
        found_target = true;
        break;
      }
    }
    // If device name matches
    if (args.find("name") != args.end()) {
      if (handles[i].name == args["name"]) {
        dev          = lime::DeviceRegistry::makeDevice(handles[i]);
        found_target = true;
        break;
      }
    }
  }

  if (!found_target) {
    dev = lime::DeviceRegistry::makeDevice(handles.at(0));
  }
  if (!dev) {
    LimeLogCallback(lime::LogLevel::Error, "Failed to create device");
    device = nullptr;
    return;
  }
  device = dev;
}

void LimeSDRSource::close() {}

int LimeSDRSource::receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts)
{
  return 0;
}

/* TODO implement send the IQ samples at the scheduled time */
int LimeSDRSource::send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot)
{
  return 0;
}

std::map<std::string, std::string> LimeSDRSource::parse_device_args(const std::string& device_args)
{
  std::map<std::string, std::string> args;
  std::istringstream                 iss(device_args);
  std::string                        token;
  while (std::getline(iss, token, ',')) {
    std::istringstream is(token);
    std::string        key;
    std::string        value;
    std::getline(is, key, ':');
    std::getline(is, value);
    args[key] = value;
  }
  return args;
}