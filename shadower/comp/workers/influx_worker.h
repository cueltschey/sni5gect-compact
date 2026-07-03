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
#include <string>
#include <vector>
#include <queue>
#include <condition_variable>
#include <type_traits>
#include <variant>

typedef struct influx_band_report_s {
	uint16_t band;
	uint32_t nof_prb;
	uint32_t offset_to_carrier;
	srsran_subcarrier_spacing_t scs_common;
	srsran_subcarrier_spacing_t scs_ssb;
	uint32_t dl_arfcn;
	uint32_t ul_arfcn;
	uint32_t ssb_arfcn;
	double dl_freq;
	double ul_freq;
	double ssb_freq;
	srsran_ssb_pattern_t ssb_pattern;
	double sample_rate;
	double uplink_cfo;
	double downlink_cfo;
} influx_band_report_t;

typedef struct rrc_reconfig_export_s {
  uint16_t rnti;
  uint8_t  rrc_transaction_id;
  bool     sp_cell_cfg_present;
  bool     recfg_with_sync_present;
  bool     phys_cell_group_cfg_present;
  bool     rlc_bearer_present;
  bool     mac_cell_group_cfg_present;
  bool     radio_bearer_cfg_present;
  bool     ded_nas_msg_present;
  bool     meas_cfg_present;
  bool     srb1_present;
  bool     srb2_present;
  bool     drb_present;
  uint32_t pdsch_harq_ack_codebook;
  std::string cell_group_cfg_hex;
} rrc_reconfig_export_t;

typedef struct cell_config_s {
  uint16_t band;
  uint32_t nof_prb;
  uint32_t dl_arfcn;
  uint32_t ul_arfcn;
  double   dl_freq;
  double   ul_freq;
  double   sample_rate;
  double   rx_gain;
  double   tx_gain;
  double   rx_frequency;
  double   tx_frequency;
  std::string scs_common;
  std::string scs_ssb;
  std::string ssb_pattern;
  double   uplink_cfo;
  double   downlink_cfo;
} cell_config_t;

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
  std::queue<std::variant<srsran_mib_nr_t, asn1::rrc_nr::sib1_s, influx_band_report_t, ChannelConfig, cell_config_t, rrc_reconfig_export_t>> msg_queue;
  std::condition_variable cv;

	bool send_band_report(const influx_band_report_t& report);
	bool send_channel_config(const ChannelConfig& ch);
  bool send_mib(const srsran_mib_nr_t& mib);
  bool send_sib1(const asn1::rrc_nr::sib1_s& sib1);
  bool send_cell_config(const cell_config_t& cfg);
  bool send_rrc_reconfig(const rrc_reconfig_export_t& cfg);
};

#endif // INFLUX_WORKER
