#include <utility>

#include "shadower/hdr/gnb_ul_worker.h"
#include "shadower/hdr/utils.h"
GNBULWorker::GNBULWorker(srslog::basic_logger&             logger_,
                         ShadowerConfig&                   config_,
                         srsue::nr::state&                 phy_state_,
                         WDWorker*                         wd_worker_,
                         Exploit*                          exploit_,
                         std::shared_ptr<srsran::mac_pcap> pcap_writer_) :
  logger(logger_),
  config(config_),
  phy_state(phy_state_),
  pcap_writer(std::move(pcap_writer_)),
  wd_worker(wd_worker_),
  exploit(exploit_)
{
}

GNBULWorker::~GNBULWorker()
{
  if (buffer) {
    free(buffer);
    buffer = nullptr;
  }
  srsran_gnb_ul_free(&gnb_ul);
}

bool GNBULWorker::init(srsran::phy_cfg_nr_t& phy_cfg_)
{
  std::lock_guard<std::mutex> lock(mutex);
  phy_cfg        = phy_cfg_;
  srate          = config.sample_rate;
  sf_len         = srate * SF_DURATION;
  slot_per_sf    = 1 << config.scs_common;
  slot_per_frame = slot_per_sf * NUM_SUBFRAME;
  slot_len       = sf_len / slot_per_sf;
  nof_sc         = config.nof_prb * SRSRAN_NRE;
  nof_re         = nof_sc * SRSRAN_NSYMB_PER_SLOT_NR;
  numerology     = (uint32_t)config.scs_common;
  /* Init buffer */
  buffer = srsran_vec_cf_malloc(sf_len);
  if (!buffer) {
    logger.error("Error allocating buffer");
    return false;
  }
  /* Init gnb_ul instance */
  if (!init_gnb_ul(gnb_ul, buffer, phy_cfg)) {
    logger.error("Error initializing gnb_ul");
    return false;
  }
  /* Initialize softbuffer rx */
  if (srsran_softbuffer_rx_init_guru(&softbuffer_rx, SRSRAN_SCH_NR_MAX_NOF_CB_LDPC, SRSRAN_LDPC_MAX_LEN_ENCODED_CB) !=
      0) {
    logger.error("Couldn't allocate and/or initialize softbuffer");
    return false;
  }
  return true;
}

/* Update the GNB UL configurations */
bool GNBULWorker::update_cfg(srsran::phy_cfg_nr_t& phy_cfg_)
{
  phy_cfg = phy_cfg_;
  if (!update_gnb_ul(gnb_ul, phy_cfg)) {
    logger.error("Failed to update gnb_ul with new phy_cfg");
    return false;
  }
  return true;
}

/* Update the rnti */
void GNBULWorker::set_rnti(uint16_t rnti_, srsran_rnti_type_t rnti_type_)
{
  std::lock_guard<std::mutex> lock(mutex);
  rnti      = rnti_;
  rnti_type = rnti_type_;
}

/* Set the context for the gnb_ul worker */
void GNBULWorker::set_task(std::shared_ptr<Task> task_)
{
  std::lock_guard<std::mutex> lock(mutex);
  task = std::move(task_);
}

/* Worker implementation, decode message send from UE to base station */
void GNBULWorker::work_imp()
{
  std::lock_guard<std::mutex> lock(mutex);
  if (rnti == SRSRAN_INVALID_RNTI) {
    logger.error("RNTI not set");
    return;
  }
  srsran_slot_cfg_t slot_cfg = {.idx = task->slot_idx};
  for (uint32_t slot_in_sf = 0; slot_in_sf < slot_per_sf; slot_in_sf++) {
    /* only copy half of the subframe to the buffer */
    slot_cfg.idx = task->slot_idx + slot_in_sf;
    if (!phy_state.get_ul_pending_grant(slot_cfg.idx, pusch_cfg, pid)) {
      continue;
    }
    /* Update the last received message time */
    update_rx_timestamp();
    /* Copy the samples to the process buffer */
    if (slot_in_sf == 0 && config.ul_sample_offset > 0) {
      /* If it is the first slot, then part of the samples is in the last slot */
      srsran_vec_cf_copy(buffer, task->last_slot->data() + sf_len - config.ul_sample_offset, config.ul_sample_offset);
      srsran_vec_cf_copy(buffer + config.ul_sample_offset, task->buffer->data(), slot_len - config.ul_sample_offset);
    } else {
      /* only copy half of the subframe to the buffer */
      srsran_vec_cf_copy(buffer, task->buffer->data() + slot_in_sf * slot_len - config.ul_sample_offset, slot_len);
    }
    /* estimate FFT will run on first slot */
    srsran_gnb_ul_fft(&gnb_ul);
    /* PUSCH search and decoding */
    handle_pusch(slot_cfg);
  }
}

/* Handle PUSCH decoding */
void GNBULWorker::handle_pusch(srsran_slot_cfg_t& slot_cfg)
{
  /* Apply the CFO to correct the OFDM symbols */
  srsran_vec_apply_cfo(gnb_ul.sf_symbols[0], config.uplink_cfo, gnb_ul.sf_symbols[0], nof_re);
  /* Initialize the buffer for output */
  srsran::unique_byte_buffer_t data = srsran::make_byte_buffer();
  if (data == nullptr) {
    logger.error("Error creating byte buffer");
    return;
  }
  /* Initialize pusch result */
  srsran_pusch_res_nr_t pusch_res = {};
  data->N_bytes                   = pusch_cfg.grant.tb[0].tbs / 8U;
  pusch_res.tb[0].payload         = data->msg;
  /* Decode PUSCH */
  if (!gnb_ul_pusch_decode(gnb_ul, pusch_cfg, slot_cfg, pusch_res, softbuffer_rx, logger, task->task_idx)) {
    return;
  }
  /* If the message is not decoded correctly, then return */
  if (!pusch_res.tb[0].crc) {
    logger.debug("Error PUSCH got wrong CRC");
    return;
  }
  /* Write to pcap */
  pcap_writer->write_ul_crnti_nr(data->msg, data->N_bytes, task->task_idx, 0, slot_cfg.idx);
  /* Pass the decoded to wdissector */
  wd_worker->process(data->msg,
                     data->N_bytes,
                     rnti,
                     slot_cfg.idx / slot_per_frame,
                     slot_cfg.idx % slot_per_frame,
                     slot_cfg.idx,
                     UL,
                     exploit);
}