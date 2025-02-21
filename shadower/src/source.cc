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

/* Initialize the radio object and apply the configurations */
LimeSDRSource::LimeSDRSource(std::string device_args,
                             double      srate_,
                             double      rx_freq,
                             double      tx_freq,
                             double      rx_gain,
                             double      tx_gain) :
  srate(srate_)
{
  std::string device_name = "limesuiteng";
  logger.info(YELLOW "SDR Name: \"%s\", Args: \"%s\"" RESET, device_name, device_args);
  if (srsran_rf_open_devname(&rf, (const char*)device_name.c_str(), (char*)device_args.c_str(), 1) != 0) {
    throw std::runtime_error("Failed to open radio");
  }

  srsran_rf_set_rx_srate(&rf, srate);
  srsran_rf_set_rx_freq(&rf, 0, rx_freq);
  srsran_rf_set_rx_gain(&rf, rx_gain);

  srsran_rf_set_tx_srate(&rf, srate);
  srsran_rf_set_tx_freq(&rf, 0, tx_freq);
  srsran_rf_set_tx_gain(&rf, tx_gain);
}

void LimeSDRSource::close()
{
  srsran_rf_close(&rf);
}

void LimeSDRSource::thread_recv()
{
  cf_t        buffer[512];
  static bool overflow_indication = false;

  enable_rt_scheduler(0);
  srsran_rf_start_rx_stream(&rf, true);

  while (true) {
    int nsamples = srsran_rf_recv(&rf, buffer, sizeof(buffer) / sizeof(cf_t), true);
    if (nsamples < 0)
      continue;

    // ring_buffer.put(buffer, nsamples);

    if (!ring_buffer.put(buffer, nsamples)) {
      if (!overflow_indication) {
        overflow_indication = true;
        logger.error("RX Overflow - Ring buffer full (%d MB)", ring_buffer.capacity() / 1024 / 1024);
      }
    }
  }
}

int LimeSDRSource::receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts)
{
  static bool rx_started = false;

  if (!rx_started) {
    rx_started = true;
    std::thread t(&LimeSDRSource::thread_recv, this);
    t.detach();
  }

  while (ring_buffer.size_used() < nof_samples)
    usleep(10);

  return ring_buffer.get(buffer, nof_samples) ? nof_samples : 0;
}

/* TODO implement send the IQ samples at the scheduled time */
int LimeSDRSource::send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot)
{
  std::lock_guard<std::mutex> lock(mutex);
  try {
    int samples_sent = srsran_rf_send_timed2(&rf, samples, length, tx_time.full_secs, tx_time.frac_secs, true, true);
    return samples_sent;
  } catch (const std::exception& e) {
    return -1;
  }
}