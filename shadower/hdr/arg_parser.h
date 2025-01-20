#ifndef SNIFFER_HDR_CONFIG_H_
#define SNIFFER_HDR_CONFIG_H_
#include "srsran/common/band_helper.h"
#include "srsran/phy/common/phy_common_nr.h"
#include "srsran/srslog/srslog.h"
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <fstream>
#include <iostream>

namespace bpo = boost::program_options;

struct ShadowerConfig {
  uint16_t                    band;
  uint32_t                    nof_prb;
  srsran_subcarrier_spacing_t scs_common;
  srsran_subcarrier_spacing_t scs_ssb;
  uint16_t                    ra_rnti;

  uint32_t freq_offset;
  uint32_t tx_gain;
  uint32_t rx_gain;

  uint32_t dl_arfcn;
  double   dl_freq;

  uint32_t ul_arfcn;
  double   ul_freq;

  uint32_t ssb_arfcn;
  double   ssb_freq;

  double   sample_rate;
  double   uplink_cfo;
  uint32_t slots_to_delay;
  int32_t  send_advance_samples;
  uint32_t max_flooding_epoch;
  float    tx_cfo_correction;
  int32_t  ul_sample_offset;
  uint32_t n_ue_dl_worker;
  uint32_t n_ue_ul_worker;
  uint32_t n_gnb_ul_worker;
  uint32_t n_gnb_dl_worker;
  uint32_t pdsch_mcs;
  uint32_t pdsch_prbs;
  uint32_t close_timeout;
  bool     parse_messages;

  srsran_ssb_pattern_t ssb_pattern;
  srsran_duplex_mode_t duplex_mode;

  bool        enable_recorder = false;
  std::string recorder_file;

  bool        use_sdr;
  std::string device_args;
  std::string record_file;

  srslog::basic_levels log_level           = srslog::basic_levels::info;
  srslog::basic_levels bc_worker_log_level = srslog::basic_levels::info;
  srslog::basic_levels worker_log_level    = srslog::basic_levels::info;
  srslog::basic_levels syncer_log_level    = srslog::basic_levels::info;
  std::string          log_file;

  bool        write_to_pcap = false;
  std::string pcap_folder;
  size_t      pool_size = 12;
  uint32_t    num_ues   = 10;

  std::string exploit_module;
};

inline int parse_args(ShadowerConfig& config, int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
    return SRSRAN_ERROR;
  }
  char*                    config_file = argv[1];
  bpo::options_description common("Config options");
  std::string              log_level;
  common.add_options()
      // clang-format off
      // Cell config section
      ("cell.band", bpo::value<uint16_t>(&config.band)->default_value(78), "NR band number by default it is 1")
      ("cell.nof_prb", bpo::value<uint32_t>(&config.nof_prb)->default_value(51), "Number of PRBs")
      ("cell.scs_common", bpo::value<std::string>()->default_value("30"), "Common subcarrier spacing")
      ("cell.scs_ssb", bpo::value<std::string>()->default_value("30"), "SSB subcarrier spacing")
      ("cell.ra_rnti", bpo::value<uint16_t>(&config.ra_rnti)->default_value(0), "SSB subcarrier spacing")
      // RF config section
      ("rf.freq_offset", bpo::value<uint32_t>(&config.freq_offset)->default_value(0), "Frequency offset")
      ("rf.tx_gain", bpo::value<uint32_t>(&config.tx_gain)->default_value(0), "TX gain")
      ("rf.rx_gain", bpo::value<uint32_t>(&config.rx_gain)->default_value(40), "RX gain")
      ("rf.dl_arfcn", bpo::value<uint32_t>(&config.dl_arfcn)->default_value(628300), "DL ARFCN")
      ("rf.ssb_arfcn", bpo::value<uint32_t>(&config.ssb_arfcn)->default_value(628320), "SSB ARFCN")
      ("rf.sample_rate", bpo::value<double>(&config.sample_rate)->default_value(23.04e6), "Sample rate")
      ("rf.uplink_cfo", bpo::value<double>(&config.uplink_cfo)->default_value(0), "Uplink CFO to apply")
      ("rf.ul_sample_offset", bpo::value<int32_t>(&config.ul_sample_offset)->default_value(0), "Uplink Number of samples in last slot")
      // task configuration section
      ("task.slots_to_delay", bpo::value<uint32_t>(&config.slots_to_delay)->default_value(2), "Number of slots to delay before sending out")
      ("task.max_flooding_epoch", bpo::value<uint32_t>(&config.max_flooding_epoch)->default_value(100), "Number of slots to delay before sending out")
      ("task.tx_cfo_correction", bpo::value<float>(&config.tx_cfo_correction)->default_value(0), "CFO correction before sending out")
      ("task.send_advance_samples", bpo::value<int32_t>(&config.send_advance_samples)->default_value(174), "Number of samples to send in advance")
      ("task.n_ue_dl_worker", bpo::value<uint32_t>(&config.n_ue_dl_worker)->default_value(4), "Number of concurrent ue dl workers")
      ("task.n_ue_ul_worker", bpo::value<uint32_t>(&config.n_ue_ul_worker)->default_value(4), "Number of concurrent ue ul workers")
      ("task.n_gnb_ul_worker", bpo::value<uint32_t>(&config.n_gnb_ul_worker)->default_value(4), "Number of concurrent gnb ul workers")
      ("task.n_gnb_dl_worker", bpo::value<uint32_t>(&config.n_gnb_dl_worker)->default_value(4), "Number of concurrent gnb dl workers")
      ("task.pdsch_mcs", bpo::value<uint32_t>(&config.pdsch_mcs)->default_value(0), "modulation and coding scheme")
      ("task.pdsch_prbs", bpo::value<uint32_t>(&config.pdsch_prbs)->default_value(24), "Number of physical resource block for PDSCH")
      ("task.close_timeout", bpo::value<uint32_t>(&config.close_timeout)->default_value(1000), "Number of milliseconds to delay before closing the connection")
      ("task.parse_messages", bpo::value<bool>(&config.parse_messages)->default_value(true), "Prevent parsing messages if the flag is false")
      // source config section
      ("source.use_sdr", bpo::value<bool>(&config.use_sdr)->default_value(false), "Use SDR")
      ("source.device_args", bpo::value<std::string>(&config.device_args)->default_value(""), "Device args for downlink")
      ("source.record_file", bpo::value<std::string>(&config.record_file)->default_value(""), "Record file for downlink")
      // Pcap settings
      ("pcap.write_to_pcap", bpo::value<bool>(&config.write_to_pcap)->default_value(false), "Use SDR")
      ("pcap.pcap_folder", bpo::value<std::string>(&config.pcap_folder)->default_value("/tmp/"), "Log level")
      // Recorder settings
      ("recorder.enable", bpo::value<bool>(&config.enable_recorder)->default_value(false), "Enable recorder")
      ("recorder.filename", bpo::value<std::string>(&config.recorder_file)->default_value("output.fc32"), "output FC32 file path")
      // log config section
      ("log.log_level", bpo::value<std::string>()->default_value("INFO"), "Log level")
      ("log.bc_worker_log_level", bpo::value<std::string>()->default_value("INFO"), "Broadcast Worker Log level")
      ("log.worker_log_level", bpo::value<std::string>()->default_value("INFO"), "Worker Log level")
      ("log.syncer_log_level", bpo::value<std::string>()->default_value("INFO"), "Syncer Log level")
      ("log.log_file", bpo::value<std::string>(&config.log_file)->default_value(""), "Log level")
      // pool section
      ("worker.pool_size", bpo::value<size_t>(&config.pool_size)->default_value(20), "Pool size")
      ("worker.num_ues", bpo::value<uint32_t>(&config.num_ues)->default_value(10), "Number of UEs to pre-initialize")
      // exploit section
      ("exploit.module", bpo::value<std::string>(&config.exploit_module)->default_value(""), "Exploit module");
  // clang-format on

  std::ifstream conf(config_file, std::ios::in);
  if (conf.fail()) {
    std::cerr << "Error: failed to open config file" << std::endl;
    return SRSRAN_ERROR;
  }

  // parse config file and handle errors gracefully
  bpo::variables_map vm;
  try {
    bpo::store(bpo::parse_config_file(conf, common), vm);
    bpo::notify(vm);
  } catch (const boost::program_options::error& e) {
    std::cerr << e.what() << std::endl;
    return SRSRAN_ERROR;
  }

  srsran::srsran_band_helper bands;
  config.scs_common = srsran_subcarrier_spacing_from_str(vm["cell.scs_common"].as<std::string>().c_str());
  config.scs_ssb    = srsran_subcarrier_spacing_from_str(vm["cell.scs_ssb"].as<std::string>().c_str());

  config.ul_arfcn = bands.get_ul_arfcn_from_dl_arfcn(config.dl_arfcn);
  config.dl_freq  = bands.nr_arfcn_to_freq(config.dl_arfcn);
  config.ul_freq  = bands.nr_arfcn_to_freq(config.ul_arfcn);
  config.ssb_freq = bands.nr_arfcn_to_freq(config.ssb_arfcn);

  config.ssb_pattern = srsran::srsran_band_helper::get_ssb_pattern(config.band, config.scs_ssb);
  config.duplex_mode = bands.get_duplex_mode(config.band);

  config.log_level           = srslog::str_to_basic_level(vm["log.log_level"].as<std::string>());
  config.syncer_log_level    = srslog::str_to_basic_level(vm["log.syncer_log_level"].as<std::string>());
  config.bc_worker_log_level = srslog::str_to_basic_level(vm["log.bc_worker_log_level"].as<std::string>());
  config.worker_log_level    = srslog::str_to_basic_level(vm["log.worker_log_level"].as<std::string>());
  return SRSRAN_SUCCESS;
}
#endif // SNIFFER_HDR_CONFIG_H_