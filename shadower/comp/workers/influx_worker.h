#ifndef INFLUX_WORKER
#define INFLUX_WORKER
#include "shadower/comp/workers/influxdb.hpp"
#include "shadower/comp/sync/syncer.h"
#include "shadower/utils/arg_parser.h"
#include "shadower/utils/utils.h"
#include "srsran/common/thread_pool.h"
#include "srsran/common/threads.h"
#include "srsran/phy/gnb/gnb_dl.h"
#include "srsran/srslog/srslog.h"
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>
#include <type_traits>
#include <variant>

class InfluxWorker
{
public:
  explicit InfluxWorker(srslog::basic_logger& logger_, const DatabaseConfig config_);
  ~InfluxWorker() = default;

  // Function to push valid results to the queue
  template <typename T>
  bool push_msg(const T& data){
    std::lock_guard<std::mutex> lock(mutex);
	  msg_queue.push(data);
	  cv.notify_one();
	  return true;
  }

	bool work();


private:
  srslog::basic_logger& logger;
  std::mutex            mutex;
  influxdb_cpp::server_info influx_server_info;
	std::string data_id;

  // Queue of any message type
  std::queue<std::variant<srsran_mib_nr_t, asn1::rrc_nr::sib1_s>> msg_queue;
  std::condition_variable cv;

  bool send_mib(const srsran_mib_nr_t& mib);
  bool send_sib1(const asn1::rrc_nr::sib1_s& sib1);
};

#endif // INFLUX_WORKER
