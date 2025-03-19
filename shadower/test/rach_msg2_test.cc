#include "shadower/hdr/utils.h"
#include "srsran/mac/mac_rar_pdu_nr.h"
#include "srsran/phy/phch/pbch_msg_nr.h"
#include "srsran/phy/ue/ue_dl_nr.h"
#include "test_variables.h"
#include <fstream>

uint16_t           rnti      = ra_rnti;
srsran_rnti_type_t rnti_type = srsran_rnti_type_ra;
#if TEST_TYPE == 1
std::string sample_file = "shadower/test/data/srsran-n78-20MHz/rach_msg2.fc32";
uint32_t    slot_number = 3330;
uint8_t     half        = 0;
#elif TEST_TYPE == 2
std::string sample_file = "shadower/test/data/rach_msg2.fc32";
uint32_t    slot_number = 11645;
uint8_t     half        = 1;
#elif TEST_TYPE == 3
std::string sample_file = "shadower/test/data/srsran-n78-40MHz/rach_msg2.fc32";
uint32_t    slot_number = 13550;
uint8_t     half        = 0;
#endif // TEST_TYPE
int main()
{
  /* initialize logger */
  srslog::basic_logger& logger = srslog_init();
  logger.set_level(srslog::basic_levels::debug);

  /* initialize phy cfg */
  srsran::phy_cfg_nr_t phy_cfg = {};
  init_phy_cfg(phy_cfg, config);

  /* init phy state */
  srsue::nr::state phy_state = {};
  init_phy_state(phy_state, config.nof_prb);

  /* load mib configuration and update phy_cfg */
  if (!configure_phy_cfg_from_mib(phy_cfg, mib_config_raw, ncellid)) {
    printf("Failed to configure phy cfg from mib\n");
    return -1;
  }

  /* load sib1 configuration and apply to phy_cfg */
  if (!configure_phy_cfg_from_sib1(phy_cfg, sib1_config_raw, sib1_size)) {
    logger.error("Failed to configure phy cfg from sib1");
    return -1;
  }

  /* UE DL init with configuration from phy_cfg */
  srsran_ue_dl_nr_t ue_dl  = {};
  cf_t*             buffer = srsran_vec_cf_malloc(sf_len);
  if (!init_ue_dl(ue_dl, buffer, phy_cfg)) {
    logger.error("Failed to init UE DL");
    return -1;
  }

  /* load test samples */
  std::vector<cf_t> samples(sf_len);
  if (!load_samples(sample_file, samples.data(), sf_len)) {
    logger.error("Failed to load data from %s", sample_file.c_str());
    return -1;
  }

  /* copy samples to ue_dl processing buffer */
  srsran_vec_cf_copy(buffer, samples.data() + half * slot_len, slot_len);
  /* Initialize slot cfg */
  srsran_slot_cfg_t slot_cfg = {.idx = slot_number};
  /* run ue_dl estimate fft */
  srsran_ue_dl_nr_estimate_fft(&ue_dl, &slot_cfg);

  /* Write OFDM symbols to file for debug purpose */
  char filename[64];

  sprintf(filename, "rach_msg2_raw");
  write_record_to_file(buffer, slot_len, filename);

  sprintf(filename, "ofdm_rach_msg2_fft%u", nof_sc);
  write_record_to_file(ue_dl.sf_symbols[0], nof_re, filename);

  ue_dl_dci_search(ue_dl, phy_cfg, slot_cfg, rnti, rnti_type, phy_state, logger);

  /* get grant from dci search */
  uint32_t                   pid          = 0;
  srsran_sch_cfg_nr_t        pdsch_cfg    = {};
  srsran_harq_ack_resource_t ack_resource = {};
  if (!phy_state.get_dl_pending_grant(slot_cfg.idx, pdsch_cfg, ack_resource, pid)) {
    logger.error("Failed to get grant from dci search");
    return -1;
  }

  /* Initialize the buffer for output*/
  srsran::unique_byte_buffer_t data = srsran::make_byte_buffer();
  if (data == nullptr) {
    logger.error("Error creating byte buffer");
    return -1;
  }
  data->N_bytes = pdsch_cfg.grant.tb[0].tbs / 8U;

  /* Initialize pdsch result*/
  srsran_pdsch_res_nr_t pdsch_res      = {};
  pdsch_res.tb[0].payload              = data->msg;
  srsran_softbuffer_rx_t softbuffer_rx = {};
  if (srsran_softbuffer_rx_init_guru(&softbuffer_rx, SRSRAN_SCH_NR_MAX_NOF_CB_LDPC, SRSRAN_LDPC_MAX_LEN_ENCODED_CB) !=
      0) {
    logger.error("Couldn't allocate and/or initialize softbuffer");
    return -1;
  }

  /* Decode PDSCH */
  if (!ue_dl_pdsch_decode(ue_dl, pdsch_cfg, slot_cfg, pdsch_res, softbuffer_rx, logger)) {
    return -1;
  }
  /* if the message is not decoded correctly, then return */
  if (!pdsch_res.tb[0].crc) {
    logger.debug("Error PDSCH got wrong CRC");
    return -1;
  }

  /* Decode the message as mac_rar_pdu */
  srsran::mac_rar_pdu_nr rar_pdu;
  if (!rar_pdu.unpack(data->msg, data->N_bytes)) {
    logger.error("Error decoding RACH msg2");
    return -1;
  }
  /* Get the first subpdu */
  uint32_t num_subpdus = rar_pdu.get_num_subpdus();
  if (num_subpdus == 0) {
    logger.error("No subpdus in RAR");
    return -1;
  }
  const srsran::mac_rar_subpdu_nr subpdu = rar_pdu.get_subpdu(0);
  /* Extract the rnti */
  uint16_t tc_rnti = subpdu.get_temp_crnti();
  if (tc_rnti == SRSRAN_INVALID_RNTI) {
    logger.error("Invalid RNTI in RAR");
    return -1;
  }
  logger.info("TC-RNTI: %u RA-RNTI: %u", tc_rnti, ra_rnti);

  /* Extract the time advance info */
  uint32_t time_advance = subpdu.get_ta();
  logger.info("Time advance: %u", time_advance);

  uint32_t n_timing_advance = subpdu.get_ta() * 16 * 64 / (1 << config.scs_common) + phy_cfg.t_offset;
  double   ta_time          = static_cast<double>(n_timing_advance) * Tc;
  uint32_t ta_samples       = ta_time * srate;
  logger.info("Uplink sample offset: %u", ta_samples);

  /* Extract the UL grant */
  std::array<uint8_t, srsran::mac_rar_subpdu_nr::UL_GRANT_NBITS> ul_grant = subpdu.get_ul_grant();
  std::ofstream rach_msg2_ul_grant(rach_msg2_ul_grant_file, std::ios::binary);
  rach_msg2_ul_grant.write(reinterpret_cast<char*>(ul_grant.data()), ul_grant.size());
  return 0;
}