#ifndef TEST_VARIABLES_H
#define TEST_VARIABLES_H
#include "shadower/utils/arg_parser.h"
#include "shadower/utils/constants.h"
#include "srsran/phy/common/phy_common_nr.h"
#include <iomanip>
#include <sstream>
struct test_args_t {
  ShadowerConfig config;
  uint32_t       slot_per_sf;
  uint32_t       sf_len;
  uint32_t       slot_len;
  uint32_t       nof_sc;
  uint32_t       nof_re;
  std::string    mib_config_raw;
  std::string    sib_config_raw;
  uint32_t       sib_size;
  std::string    rrc_setup_raw;
  uint32_t       rrc_setup_size;
  std::string    rar_ul_grant_file;
  uint16_t       c_rnti;
  uint16_t       ra_rnti;
  uint32_t       ncellid;
};

/* Shadower config for srsran n78 20MHz*/
ShadowerConfig srsran_n78_20MHz_config = {
    .band              = 78,
    .nof_prb           = 51,
    .scs_common        = srsran_subcarrier_spacing_30kHz,
    .scs_ssb           = srsran_subcarrier_spacing_30kHz,
    .freq_offset       = 0,
    .tx_gain           = 80,
    .rx_gain           = 40,
    .dl_freq           = 3427.5e6,
    .ul_freq           = 3427.5e6,
    .ssb_freq          = 3421.92e6,
    .sample_rate       = 23.04e6,
    .nof_channels      = 1,
    .uplink_cfo        = -0.00054,
    .ssb_pattern       = SRSRAN_SSB_PATTERN_C,
    .duplex_mode       = SRSRAN_DUPLEX_MODE_TDD,
    .delay_n_slots     = 5,
    .duplications      = 4,
    .tx_cfo_correction = 0.0,
    .tx_advancement    = 160,
    .pdsch_mcs         = 3,
    .pdsch_prbs        = 24,
    .n_ue_dl_worker    = 4,
    .n_ue_ul_worker    = 4,
    .n_gnb_ul_worker   = 4,
    .n_gnb_dl_worker   = 4,
    .close_timeout     = 5000,
    .parse_messages    = true,
    .enable_gpu        = false,
    .enable_recorder   = false,
    .source_type       = "file",
    .source_params     = "/root/records/example.fc32",
    .source_module     = file_source_module_path,
    .source_srate      = 23.04e6,
    .log_level         = srslog::basic_levels::debug,
    .bc_worker_level   = srslog::basic_levels::debug,
    .worker_log_level  = srslog::basic_levels::debug,
    .syncer_log_level  = srslog::basic_levels::debug,
    .pcap_folder       = "/tmp/",
};

/* Shadower config for srsran n78 40MHz*/
ShadowerConfig srsran_n78_40MHz_config = {
    .band              = 78,
    .nof_prb           = 106,
    .scs_common        = srsran_subcarrier_spacing_30kHz,
    .scs_ssb           = srsran_subcarrier_spacing_30kHz,
    .freq_offset       = 0,
    .tx_gain           = 80,
    .rx_gain           = 40,
    .dl_freq           = 3427.5e6,
    .ul_freq           = 3427.5e6,
    .ssb_freq          = 3413.28e6,
    .sample_rate       = 46.08e6,
    .nof_channels      = 1,
    .uplink_cfo        = -0.00054,
    .ssb_pattern       = SRSRAN_SSB_PATTERN_C,
    .duplex_mode       = SRSRAN_DUPLEX_MODE_TDD,
    .delay_n_slots     = 5,
    .duplications      = 4,
    .tx_cfo_correction = 0.0,
    .tx_advancement    = 160,
    .pdsch_mcs         = 3,
    .pdsch_prbs        = 24,
    .n_ue_dl_worker    = 4,
    .n_ue_ul_worker    = 4,
    .n_gnb_ul_worker   = 4,
    .n_gnb_dl_worker   = 4,
    .close_timeout     = 5000,
    .parse_messages    = true,
    .enable_gpu        = false,
    .enable_recorder   = false,
    .source_type       = "file",
    .source_params     = "/root/records/example.fc32",
    .source_module     = file_source_module_path,
    .bc_worker_level   = srslog::basic_levels::debug,
    .source_srate      = 46.08e6,
    .log_level         = srslog::basic_levels::debug,
    .worker_log_level  = srslog::basic_levels::debug,
    .syncer_log_level  = srslog::basic_levels::debug,
    .pcap_folder       = "/tmp/",
};

/* Shadower config for Effnet 20MHz*/
ShadowerConfig effnet_n78_20MHz = {
    .band              = 78,
    .nof_prb           = 51,
    .scs_common        = srsran_subcarrier_spacing_30kHz,
    .scs_ssb           = srsran_subcarrier_spacing_30kHz,
    .freq_offset       = 0,
    .tx_gain           = 80,
    .rx_gain           = 40,
    .dl_freq           = 3619.2e6,
    .ul_freq           = 3619.2e6,
    .ssb_freq          = 3619.2e6,
    .sample_rate       = 23.04e6,
    .nof_channels      = 1,
    .uplink_cfo        = 0.0,
    .ssb_pattern       = SRSRAN_SSB_PATTERN_C,
    .duplex_mode       = SRSRAN_DUPLEX_MODE_TDD,
    .delay_n_slots     = 5,
    .duplications      = 4,
    .tx_cfo_correction = 0.0,
    .tx_advancement    = 160,
    .pdsch_mcs         = 3,
    .pdsch_prbs        = 24,
    .n_ue_dl_worker    = 4,
    .n_ue_ul_worker    = 4,
    .n_gnb_ul_worker   = 4,
    .n_gnb_dl_worker   = 4,
    .close_timeout     = 5000,
    .parse_messages    = true,
    .enable_gpu        = false,
    .enable_recorder   = false,
    .source_type       = "file",
    .source_params     = "/root/records/example.fc32",
    .source_module     = file_source_module_path,
    .source_srate      = 23.04e6,
    .log_level         = srslog::basic_levels::debug,
    .bc_worker_level   = srslog::basic_levels::debug,
    .worker_log_level  = srslog::basic_levels::debug,
    .syncer_log_level  = srslog::basic_levels::debug,
    .pcap_folder       = "/tmp/",
};

test_args_t init_test_args(int test_number)
{
  test_args_t test_args = {};
  if (test_number == 0) {
    test_args.config            = srsran_n78_20MHz_config;
    test_args.mib_config_raw    = "shadower/test/data/srsran-n78-20MHz/mib.raw";
    test_args.sib_config_raw    = "shadower/test/data/srsran-n78-20MHz/sib1.raw";
    test_args.sib_size          = 101;
    test_args.rrc_setup_raw     = "shadower/test/data/srsran-n78-20MHz/rrc_setup.raw";
    test_args.rrc_setup_size    = 316;
    test_args.rar_ul_grant_file = "shadower/test/data/srsran-n78-20MHz/rach_msg2_ul_grant.raw";
    test_args.c_rnti            = 17921;
    test_args.ra_rnti           = 267;
    test_args.ncellid           = 1;
  } else if (test_number == 1) {
    test_args.config            = srsran_n78_40MHz_config;
    test_args.mib_config_raw    = "shadower/test/data/srsran-n78-40MHz/mib.raw";
    test_args.sib_config_raw    = "shadower/test/data/srsran-n78-40MHz/sib1.raw";
    test_args.sib_size          = 101;
    test_args.rrc_setup_raw     = "shadower/test/data/srsran-n78-40MHz/rrc_setup.raw";
    test_args.rrc_setup_size    = 316;
    test_args.rar_ul_grant_file = "shadower/test/data/srsran-n78-40MHz/rach_msg2_ul_grant.raw";
    test_args.c_rnti            = 17921;
    test_args.ra_rnti           = 267;
    test_args.ncellid           = 1;
  } else if (test_number == 2) {
    test_args.config            = effnet_n78_20MHz;
    test_args.mib_config_raw    = "shadower/test/data/effnet/mib.raw";
    test_args.sib_config_raw    = "shadower/test/data/effnet/sib1.raw";
    test_args.sib_size          = 106;
    test_args.rrc_setup_raw     = "shadower/test/data/effnet/rrc_setup.raw";
    test_args.rrc_setup_size    = 176;
    test_args.rar_ul_grant_file = "shadower/test/data/effnet/rach_msg2_ul_grant.raw";
    test_args.c_rnti            = 42000;
    test_args.ra_rnti           = 0x0113;
    test_args.ncellid           = 1;
  } else {
    throw std::invalid_argument("Invalid test number");
  }

  test_args.slot_per_sf = 1 << (uint32_t)test_args.config.scs_common;
  test_args.sf_len      = test_args.config.sample_rate * SF_DURATION;
  test_args.slot_len    = test_args.sf_len / test_args.slot_per_sf;
  test_args.nof_sc      = test_args.config.nof_prb * SRSRAN_NRE;
  test_args.nof_re      = test_args.nof_sc * SRSRAN_NSYMB_PER_SLOT_NR;
  return test_args;
}
#endif // TEST_VARIABLES_H