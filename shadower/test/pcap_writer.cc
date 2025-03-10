#include "srsran/common/mac_pcap.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <hex_data> (space-separated, e.g., '80 48 00 38 16 53')" << std::endl;
    return 1;
  }

  // Parse hex data from command line
  std::vector<uint8_t> data;
  for (int i = 1; i < argc; ++i) {
    std::stringstream ss;
    ss << std::hex << argv[i];
    int byte;
    ss >> byte;
    data.push_back(static_cast<uint8_t>(byte));
  }

  uint32_t          len    = data.size();
  srsran::mac_pcap* writer = new srsran::mac_pcap();
  if (writer->open("logs/debug.pcap")) {
    printf("Failed to open pcap file\n");
  }
  writer->write_dl_crnti_nr(data.data(), data.size(), 0x10b, 4, 10);
  writer->close();
}