#ifndef SRSRAN_UE_TRACKER_H
#define SRSRAN_UE_TRACKER_H
#include "shadower/hdr/arg_parser.h"
#include "shadower/hdr/exploit.h"
#include "shadower/hdr/gnb_dl_worker.h"
#include "shadower/hdr/gnb_ul_worker.h"
#include "shadower/hdr/safe_queue.h"
#include "shadower/hdr/source.h"
#include "shadower/hdr/syncer.h"
#include "shadower/hdr/task.h"
#include "shadower/hdr/ue_dl_worker.h"
#include "shadower/hdr/wd_worker.h"
#include "srsran/adt/circular_map.h"
#include "srsran/asn1/rrc_nr.h"
#include "srsran/common/mac_pcap.h"
#include "srsran/common/thread_pool.h"
#include "srsran/common/threads.h"
#include "srsran/mac/mac_rar_pdu_nr.h"
#include "srsran/srslog/srslog.h"
#include "srsue/hdr/phy/nr/state.h"
#include <atomic>
#include <map>
#include <mutex>
class UETracker : public srsran::thread
{
public:
  UETracker(Source*           source_,
            Syncer*           syncer_,
            WDWorker*         wd_worker_,
            ShadowerConfig&   config_,
            create_exploit_t& create_exploit_handler);
  ~UETracker() override;

  bool init();

  /* Initialize the logger and pcap writer, then enable the UE tracker */
  void activate(uint16_t rnti_, srsran_rnti_type_t rnti_type_);

  /* Deactivate UETracker and put back to the worker pool */
  void deactivate();

  /* Check if the UETracker is active */
  bool is_active() { return active; }

  /* Set the rar grant for UE found in rach msg2 */
  void set_ue_rar_grant(std::array<uint8_t, srsran::mac_rar_subpdu_nr::UL_GRANT_NBITS>& grant, uint32_t slot_idx);

  /* Apply the configuration from SIB1 */
  bool apply_config_from_sib1(asn1::rrc_nr::sib1_s& sib1);

  /* Apply the configuration from MIB */
  bool apply_config_from_mib(srsran_mib_nr_t& mib, uint32_t ncellid);

  /* Apply the configuration from cell group */
  bool apply_config_from_rrc_setup(asn1::rrc_nr::cell_group_cfg_s& cell_group);

  /* Distribute the phy_cfg config */
  bool update_cfg();

  /* UETracker thread implementation */
  void run_thread() override;

  /* Apply the task to the ue_dl and gnb_ul */
  void work_on_task(const std::shared_ptr<Task>& task);

  /* Update the last received message timestamp */
  void update_last_rx_timestamp();

private:
  srslog::basic_logger& logger = srslog::fetch_basic_logger("UETracker");
  /* UE name */
  std::string name;
  /* UE temporary identity */
  uint16_t           rnti{};
  srsran_rnti_type_t rnti_type = srsran_rnti_type_c;
  /* Shadower config */
  ShadowerConfig& config;
  /* Syncer to get the synchronization information */
  Syncer* syncer = nullptr;
  /* Source of the IQ samples */
  Source* source = nullptr;
  /* Wdissector worker */
  WDWorker* wd_worker = nullptr;
  /* UE specific exploit instance */
  Exploit*          exploit = nullptr;
  create_exploit_t& exploit_creator;
  /* Pcap writer, to record the message to pcap file */
  std::shared_ptr<srsran::mac_pcap> pcap_writer;
  /* Indicate if RRC set up from DL is found or not */
  std::atomic<bool> pending_rrc_setup{false};
  /* Indicate if current UETracker is active or not */
  std::atomic<bool> active{false};
  /* messages intended to send to UE */
  SafeQueue<std::vector<uint8_t> > dl_msg_queue;
  /* messages intended to send to base station */
  SafeQueue<std::vector<uint8_t> > ul_msg_queue;
  /* Last received grant time */
  std::chrono::time_point<std::chrono::steady_clock> last_message_time;
  /* Mutexes */
  std::mutex time_mtx;
  std::mutex cfg_mtx;
  /* Parameters required for update phy_cfg from cell_cfg */
  srsran::static_circular_map<uint32_t, srsran_pucch_nr_resource_t, 128> pucch_res_list = {};
  std::map<uint32_t, srsran_csi_rs_zp_resource_t>                        csi_rs_zp_res  = {};
  std::map<uint32_t, srsran_csi_rs_nzp_resource_t>                       csi_rs_nzp_res = {};

  srsran::phy_cfg_nr_t phy_cfg   = {}; // physical configuration
  srsue::nr::state     phy_state = {}; // UE side grant tracker

  /* cell group config */
  asn1::rrc_nr::cell_group_cfg_s  cell_group_cfg = {};
  asn1::rrc_nr::cell_group_cfg_s& get_cell_group_cfg() { return cell_group_cfg; }

  /* UE DL */
  std::vector<UEDLWorker*> ue_dl_workers;
  srsran::thread_pool      ue_dl_pool; // thread pool for ue_dl

  /* UE UL */

  /* GNB DL */
  std::vector<GNBDLWorker*> gnb_dl_workers;
  srsran::thread_pool       gnb_dl_pool; // thread pool for gnb_dl
  std::thread               gnb_dl_thread;

  /* GNB UL */
  std::vector<GNBULWorker*> gnb_ul_workers;
  srsran::thread_pool       gnb_ul_pool; // thread pool for gnb_ul
};

#endif // SRSRAN_UE_TRACKER_H