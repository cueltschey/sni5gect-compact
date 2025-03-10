#include "shadower/hdr/scheduler.h"

Scheduler::Scheduler(ShadowerConfig& config_, Source* source_, Syncer* syncer_, create_exploit_t create_exploit_) :
  config(config_), source(source_), syncer(syncer_), create_exploit(create_exploit_), srsran::thread("Scheduler")
{
  /* Initialize phy config */
  init_phy_cfg(phy_cfg, config);
  /* Initialize broadcast worker */
  broadcast_worker = new BroadCastWorker(config);
  /* Attach the handler to create new UE tracker when new RACH msg2 is found */
  broadcast_worker->on_ue_found = std::bind(&Scheduler::handle_new_ue_found,
                                            this,
                                            std::placeholders::_1,
                                            std::placeholders::_2,
                                            std::placeholders::_3,
                                            std::placeholders::_4);
  /* Attach the handler to apply the configuration from SIB1 */
  broadcast_worker->on_sib1_found = std::bind(&Scheduler::handle_sib1, this, std::placeholders::_1);

  /* bind the cell found handler to broadcast worker, when cell is found, apply the configuration to broadcast worker */
  syncer->on_cell_found = std::bind(&Scheduler::handle_mib, this, std::placeholders::_1, std::placeholders::_2);
  /* bind the new task handler of the syncer */
  syncer->publish_subframe = std::bind(&Scheduler::push_new_task, this, std::placeholders::_1);
  /* bind the syncer error handler */
  syncer->error_handler = std::bind(&Scheduler::syncer_exit_handler, this);

  /* Initialize wdissector */
  wd_worker = new WDWorker(config.duplex_mode, config.worker_log_level);
  /* initialize thread pool */
  thread_pool = new ThreadPool(config.pool_size);
  /* Initialize a list of UE trackers before start */
  pre_initialize_ue();
}

/* Initialize a list of UE trackers before start */
void Scheduler::pre_initialize_ue()
{
  for (uint32_t i = 0; i < config.num_ues; i++) {
    /* Create new UE tracker */
    std::shared_ptr<UETracker> ue = std::make_shared<UETracker>(source, syncer, wd_worker, config, create_exploit);
    /* Call the init function */
    if (!ue->init()) {
      logger.error("Failed to initialize UE tracker");
      continue;
    }
    ue_trackers.push_back(ue);
  }
}

/* handler to handle syncer error event */
void Scheduler::syncer_exit_handler()
{
  std::this_thread::sleep_for(std::chrono::seconds(5));
  logger.error(RED "Syncer error event" RESET);
  running.store(false);
  thread_cancel();
}

/* handler to handle new task from syncer */
void Scheduler::push_new_task(std::shared_ptr<Task>& task)
{
  task_queue.push(task);
}

/* handler to activate new UE tracker when new RACH msg2 is found */
void Scheduler::handle_new_ue_found(uint16_t                   rnti,
                                    std::array<uint8_t, 27UL>& grant,
                                    uint32_t                   current_slot,
                                    uint32_t                   time_advance)
{
  std::shared_ptr<UETracker> selected_ue = nullptr;
  /* select a UE tracker that is not activated */
  for (uint32_t i = 0; i < config.num_ues; i++) {
    auto ue = ue_trackers[i];
    if (!ue->is_active()) {
      selected_ue = ue;
      break;
    }
  }
  if (!selected_ue) {
    logger.error(RED "No available UE tracker" RESET);
    return;
  }
  selected_ue->activate(rnti, srsran_rnti_type_c, time_advance);
  selected_ue->set_ue_rar_grant(grant, current_slot);
}

/* handler to apply MIB configuration to multiple workers */
void Scheduler::handle_mib(srsran_mib_nr_t& mib, uint32_t ncellid)
{
  broadcast_worker->apply_config_from_mib(mib, ncellid);
  thread_pool->enqueue([this, &mib, ncellid]() {
    for (const std::shared_ptr<UETracker>& ue : ue_trackers) {
      ue->apply_config_from_mib(mib, ncellid);
    }
    logger.info(CYAN "MIB applied to all workers" RESET);
  });
}

/* handler to apply sib1 configuration to multiple workers */
void Scheduler::handle_sib1(asn1::rrc_nr::sib1_s& sib1)
{
  broadcast_worker->apply_config_from_sib1(sib1);
  thread_pool->enqueue([this, &sib1]() {
    for (const std::shared_ptr<UETracker>& ue : ue_trackers) {
      ue->apply_config_from_sib1(sib1);
    }
    logger.info(CYAN "SIB1 applied to all workers" RESET);
  });
}

void Scheduler::run_thread()
{
  while (running) {
    std::shared_ptr<Task> task = task_queue.retrieve();
    if (!task) {
      continue;
    }
    /* Run the task on all active ue trackers */
    for (const std::shared_ptr<UETracker>& ue : ue_trackers) {
      if (!ue->is_active()) {
        continue;
      }
      ue->work_on_task(task);
    }
    /* Run the task on the broadcast worker */
    thread_pool->enqueue([this, task]() { broadcast_worker->work(task); });
  }
}