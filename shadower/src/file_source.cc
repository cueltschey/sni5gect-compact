#include "shadower/hdr/file_source.h"
#include "shadower/hdr/utils.h"

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