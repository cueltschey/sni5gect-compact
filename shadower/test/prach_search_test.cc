#include "shadower/hdr/safe_queue.h"
#include "shadower/hdr/syncer.h"
#include "shadower/hdr/utils.h"
#include "srsran/mac/mac_rar_pdu_nr.h"
#include "srsran/phy/phch/pbch_msg_nr.h"
#include "srsran/phy/phch/prach.h"
#include "srsran/phy/sync/ssb.h"
#include "test_variables.h"
#include <fstream>

std::string     sample_file_path = "/root/records/example.fc32";
SafeQueue<Task> task_queue       = {};

bool on_cell_found(srsran_mib_nr_t& mib, uint32_t ncellid_)
{
  std::array<char, 512> mib_info_str = {};
  srsran_pbch_msg_nr_mib_info(&mib, mib_info_str.data(), (uint32_t)mib_info_str.size());
  printf("Found cell: %s %u\n", mib_info_str.data(), ncellid_);
  return true;
}

void push_new_task(std::shared_ptr<Task>& task)
{
  task_queue.push(task);
}

void handle_syncer_exit(Syncer* syncer, std::thread& prach_search_thread)
{
  std::this_thread::sleep_for(std::chrono::seconds(5));
  syncer->thread_cancel();
  pthread_cancel(prach_search_thread.native_handle());
}

void run_prach_search(srsran::phy_cfg_nr_t& phy_cfg, srslog::basic_logger& logger);

int main(int argc, char* argv[])
{
  /* initialize logger */
  srslog::basic_logger& logger = srslog_init();
  logger.set_level(srslog::basic_levels::debug);

  /* initialize phy cfg */
  srsran::phy_cfg_nr_t phy_cfg = {};
  init_phy_cfg(phy_cfg, config);

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

  std::thread prach_search_thread(run_prach_search, std::ref(phy_cfg), std::ref(logger));

  /* Initialize syncer */
  syncer_args_t syncer_args = {
      .srate       = config.sample_rate,
      .scs         = config.scs_ssb,
      .dl_freq     = config.dl_freq,
      .ssb_freq    = config.ssb_freq,
      .pattern     = config.ssb_pattern,
      .duplex_mode = config.duplex_mode,
  };
  Source* source;
  if (argc > 1) {
    source = new FileSource(argv[1], config.sample_rate);
  } else {
    source = new FileSource(sample_file_path.c_str(), config.sample_rate);
  }

  /* Run syncer to align the subframes */
  Syncer* syncer = new Syncer(syncer_args, source, config);
  syncer->init();
  syncer->on_cell_found    = std::bind(on_cell_found, std::placeholders::_1, std::placeholders::_2);
  syncer->publish_subframe = std::bind(push_new_task, std::placeholders::_1);
  syncer->error_handler    = std::bind(handle_syncer_exit, syncer, std::ref(prach_search_thread));
  syncer->start(0);
  syncer->wait_thread_finish();
  prach_search_thread.join();
  source->close();
}

void run_prach_search(srsran::phy_cfg_nr_t& phy_cfg, srslog::basic_logger& logger)
{
  /* Initialize prach */
  srsran_prach_t prach = {};
  if (srsran_prach_init(&prach, srsran_symbol_sz(config.nof_prb))) {
    logger.error("Failed to initialized PRACH");
    return;
  }

  /* Update prach with configuration from phy_cfg */
  if (srsran_prach_set_cfg(&prach, &phy_cfg.prach, config.nof_prb)) {
    logger.error("Failed to set PRACH config");
    return;
  }
  srsran_prach_set_detect_factor(&prach, 60);

  while (true) {
    std::shared_ptr<Task> task = task_queue.retrieve();
    if (task == nullptr) {
      continue;
    }
    if (task->buffer->empty()) {
      continue;
    }
    uint32_t prach_indices[165] = {};
    float    prach_offsets[165] = {};
    float    prach_p2avg[165]   = {};
    uint32_t prach_nof_det      = 0;
    if (srsran_prach_detect_offset(&prach,
                                   0,
                                   task->buffer->data(),
                                   sf_len - prach.N_cp,
                                   prach_indices,
                                   prach_offsets,
                                   prach_p2avg,
                                   &prach_nof_det)) {
      logger.error("Failed to detect PRACH");
      return;
    }

    if (prach_nof_det) {
      logger.info("Detected %u PRACH signals in %u %u", prach_nof_det, task->task_idx, task->slot_idx);
      for (uint32_t i = 0; i < prach_nof_det; i++) {
        logger.info("PRACH:  preamble=%d, offset=%.1f us, peak2avg=%.1f",
                    prach_indices[i],
                    prach_offsets[i] * 1e6,
                    prach_p2avg[i]);
      }
    }
  }
}