#include "shadower/comp/workers/wd_worker.h"
#include "shadower/modules/dummy_exploit.h"
#include "shadower/utils/arg_parser.h"
#include "shadower/utils/safe_queue.h"
#include "shadower/utils/utils.h"

uint8_t                          buffer[2048];
uint32_t                         size;
direction_t                      direction = DL;
SafeQueue<std::vector<uint8_t> > dl_msg_queue;
SafeQueue<std::vector<uint8_t> > ul_msg_queue;

void parse_args(int argc, char* argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "pd")) != -1) {
    switch (opt) {
      case 'p': {
        std::string packet = argv[optind];
        if (!hex_to_bytes(packet, buffer, &size)) {
          fprintf(stderr, "Failed to convert hex string to bytes.\n");
          exit(EXIT_FAILURE);
        }
        break;
      }
      case 'd': {
        std::string dir = argv[optind];
        if (dir == "dl") {
          direction = DL;
        } else if (dir == "ul") {
          direction = UL;
        } else {
          fprintf(stderr, "Unknown direction: %s\n", dir.c_str());
          exit(EXIT_FAILURE);
        }
        break;
      }
      default:
        fprintf(stderr, "Unknown option or missing argument.\n");
    }
  }
}

int main(int argc, char* argv[])
{
  parse_args(argc, argv);
  /* initialize logger */
  ShadowerConfig config;
  config.log_level             = srslog::basic_levels::debug;
  srslog::basic_logger& logger = srslog_init(&config);
  /* initialize wd worker */
  srsran_duplex_mode_t duplex_mode = srsran_duplex_mode_t::SRSRAN_DUPLEX_MODE_TDD;
  WDWorker             wd_worker(duplex_mode, config.log_level);
  /* initialize exploit */
  DummyExploit exploit(dl_msg_queue, ul_msg_queue);
  wd_worker.process(buffer, size, 42000, 1, 1, 1, direction, &exploit);
}