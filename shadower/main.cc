#include "shadower/hdr/exploit.h"
#include "shadower/hdr/scheduler.h"
#include "shadower/hdr/syncer.h"
#include "srsran/srslog/srslog.h"
// Configuration for the overshadower
ShadowerConfig config = {};
// Source pointer
Source* source = nullptr;

int main(int argc, char* argv[])
{
  // load configurations from file
  if (parse_args(config, argc, argv)) {
    printf("Error parsing args\n");
    return -1;
  }
  srslog::basic_logger& logger = srslog_init(&config);
  logger.set_level(config.log_level);

  // Initialize the source based on the configuration
  if (config.use_sdr) {
    source = new SDRSource(config.device_args,
                           config.sample_rate,
                           config.dl_freq,
                           config.ul_freq,
                           config.rx_gain,
                           config.tx_gain,
                           config.device_name);
    logger.info(YELLOW "Initialized source using SDR: %s" RESET, config.device_args.c_str());
  } else {
    source = new FileSource(config.record_file.c_str(), config.sample_rate);
    logger.info(YELLOW "Initialized source using file: %s" RESET, config.record_file.c_str());
  }

  // load exploit module
  create_exploit_t exploit_creator = load_exploit(config.exploit_module);
  logger.info(YELLOW "Loaded exploit module: %s" RESET, config.exploit_module.c_str());
  // Initialize and create the syncer instance
  syncer_args_t syncer_args = {
      .srate       = config.sample_rate,
      .scs         = config.scs_ssb,
      .dl_freq     = config.dl_freq,
      .ssb_freq    = config.ssb_freq,
      .pattern     = config.ssb_pattern,
      .duplex_mode = config.duplex_mode,
  };
  Syncer* syncer = new Syncer(syncer_args, source, config);
  if (!syncer->init()) {
    logger.error(YELLOW "Error initializing syncer" RESET);
    return -1;
  }
  logger.info(YELLOW "Initialized syncer" RESET);

  // Initialize the scheduler
  Scheduler* scheduler = new Scheduler(config, source, syncer, exploit_creator);
  logger.info(YELLOW "Initialized scheduler" RESET);
  // Start the syncer thread with the highest priority
  syncer->start(99);
  // Start the scheduler thread with lower priority
  scheduler->start(80);

  syncer->wait_thread_finish();
  source->close();
  return 0;
}