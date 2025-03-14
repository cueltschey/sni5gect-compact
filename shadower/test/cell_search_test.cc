#include "shadower/hdr/ssb_cuda.cuh"
#include "shadower/hdr/utils.h"
#include "srsran/phy/phch/pbch_msg_nr.h"
#include "srsran/phy/sync/ssb.h"
#include "srsran/srslog/srslog.h"
#include "test_variables.h"
#include <fstream>

#if TEST_TYPE == 1
std::string sample_file = "shadower/test/data/srsran/ssb.fc32";
#elif TEST_TYPE == 2
std::string sample_file = "shadower/test/data/ssb.fc32";
#elif TEST_TYPE == 3
std::string sample_file = "shadower/test/data/srsran-n78-40MHz/sib.fc32";
#endif // TEST_TYPE
int main()
{
  /* initialize logger */
  srslog::basic_logger& logger = srslog_init();
  logger.set_level(srslog::basic_levels::debug);

  /* initialize ssb */
  srsran_ssb_t ssb = {};
  if (!init_ssb(ssb, srate, dl_freq, ssb_freq, scs, pattern, duplex)) {
    logger.error("Failed to initialize SSB");
    return -1;
  }

  /* load samples from file */
  std::vector<cf_t> samples(sf_len);
  if (!load_samples(sample_file, samples.data(), sf_len)) {
    logger.error("Failed to load data from %s", sample_file.c_str());
    return -1;
  }

  /* search for SSB */
  srsran_ssb_search_res_t res = {};
  if (srsran_ssb_search(&ssb, samples.data(), sf_len, &res) < SRSRAN_SUCCESS) {
    logger.error("Error running srsran_ssb_search");
    return -1;
  }
  if (res.measurements.snr_dB < -10.0f || !res.pbch_msg.crc) {
    logger.error("Failed to decode PBCH message");
    return -1;
  }

  /* decode MIB */
  srsran_mib_nr_t mib = {};
  if (srsran_pbch_msg_nr_mib_unpack(&res.pbch_msg, &mib) < SRSRAN_SUCCESS) {
    logger.error("Error running srsran_pbch_msg_nr_mib_unpack");
    return -1;
  }

  /* get SSB index */
  uint32_t sf_idx   = srsran_ssb_candidate_sf_idx(&ssb, res.pbch_msg.ssb_idx, res.pbch_msg.hrf);
  uint32_t slot_idx = mib.sfn * 10 * slots_per_sf + sf_idx;
  logger.info("SF index: %u Slot index: %u", sf_idx, slot_idx);

  /* write MIB to file */
  std::array<char, 512> mib_info_str = {};
  srsran_pbch_msg_nr_mib_info(&mib, mib_info_str.data(), mib_info_str.size());
  std::ofstream mib_raw{mib_config_raw, std::ios::binary};
  mib_raw.write(reinterpret_cast<const char*>(&mib), sizeof(mib));

  logger.info("Found cell: %s", mib_info_str.data());
  logger.info("Cell id: %u", res.N_id);
  logger.info("Offset: %u", res.t_offset);
  logger.info("CFO: %f", res.measurements.cfo_hz);

  auto                          start1              = std::chrono::high_resolution_clock::now();
  srsran_csi_trs_measurements_t csi_trs_measurement = {};
  srsran_pbch_msg_nr_t          pbch_msg1           = {};
  if (srsran_ssb_find(&ssb, samples.data(), res.N_id, &csi_trs_measurement, &pbch_msg1) != SRSRAN_SUCCESS) {
    logger.error("Error running srsran_ssb_find");
    return -1;
  }
  if (!pbch_msg1.crc) {
    logger.error("PBCH CRC not match (srsran_ssb_find)");
  } else {
    logger.info("PBCH CRC matched (srsran_ssb_find)");
  }
  auto end1     = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  logger.info("srsran_ssb_find: %ld us", duration.count());

  auto                          start        = std::chrono::high_resolution_clock::now();
  srsran_csi_trs_measurements_t measurements = {};
  srsran_pbch_msg_nr_t          pbch_msg     = {};
  uint32_t                      half_frame   = 0;
  if (srsran_ssb_track(&ssb, samples.data(), res.N_id, pbch_msg.ssb_idx, half_frame, &measurements, &pbch_msg) !=
      SRSRAN_SUCCESS) {
    logger.error("Error running srsran_ssb_track");
    return -1;
  }
  auto end       = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  logger.info("srsran_ssb_track: %ld us", duration1.count());

  SSBCuda ssb_cuda(srate, dl_freq, ssb_freq, scs, pattern, duplex);
  if (!ssb_cuda.init(SRSRAN_NID_2_NR(res.N_id))) {
    logger.error("Failed to initialize SSB CUDA");
    return -1;
  }

  auto start2 = std::chrono::high_resolution_clock::now();
  ssb_cuda.ssb_run_sync_find(samples.data(), res.N_id, &measurements, &pbch_msg);
  auto end2      = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
  logger.info("ssb_run_sync_find: %ld us", duration2.count());

  srsran_mib_nr_t mib_cuda = {};
  if (pbch_msg.crc) {
    logger.info("PBCH CRC matched");
  } else {
    logger.error("PBCH CRC not match");
    return -1;
  }
  if (srsran_pbch_msg_nr_mib_unpack(&pbch_msg, &mib_cuda) < SRSRAN_SUCCESS) {
    logger.error("Error running srsran_pbch_msg_nr_mib_unpack");
    return -1;
  }
  srsran_pbch_msg_nr_mib_info(&mib_cuda, mib_info_str.data(), (uint32_t)mib_info_str.size());
  logger.info(YELLOW "Found cell: %s" RESET, mib_info_str.data());
  ssb_cuda.cleanup();
  return 0;
}