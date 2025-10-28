#include "shadower/comp/workers/influx_worker.h"
#include "shadower/utils/arg_parser.h"

DatabaseConfig config;

void parse_args(int argc, char* argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "hpbot")) != -1) {
    switch (opt) {
      case 'h': {
        config.host = std::string(argv[optind]);
        printf("Using host: %s\n", config.host);
        break;
      }
      case 'b': {
        config.bucket = std::string(argv[optind]);
        printf("Using bucket: %s\n", config.bucket);
        break;
      }
      case 'o': {
        config.org = std::string(argv[optind]);
        printf("Using org: %s\n", config.bucket);
        break;
      }
      case 't': {
        config.token = std::string(argv[optind]);
        printf("Using bucket: %s\n", config.token);
        break;
      }
      case 'p': {
        config.port = atoi(argv[optind]);
        printf("Using port: %d\n", config.port);
        break;
      }
      default:
        fprintf(stderr, "Unknown option or missing argument.\n");
        exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char* argv[]){
	parse_args(argc, argv);
}
