#include "dummy_exploit.h"
#include "shadower/hdr/msg_helper.h"
#include "shadower/hdr/safe_queue.h"
#include "shadower/hdr/utils.h"
#include "shadower/hdr/wd_worker.h"
#include "srsran/asn1/rrc_nr.h"
#include "test_variables.h"
int main()
{
  srslog::basic_logger& logger = srslog_init();
  logger.set_level(srslog::basic_levels::debug);
  /* Run wdissector for packet summary */
  WDWorker*                        wd_worker = new WDWorker(SRSRAN_DUPLEX_MODE_TDD, srslog::basic_levels::debug);
  SafeQueue<std::vector<uint8_t> > dl_msg_queue;
  SafeQueue<std::vector<uint8_t> > ul_msg_queue;
  DummyExploit*                    exploit = new DummyExploit(dl_msg_queue, ul_msg_queue);

  std::string nas_msg = "7e0396c832e9007e005d020002f0f0e1360102";

  /* Pack to rrc_nr test */
  uint8_t rrc_nr_origin[] = {0x28, 0x82, 0x6f, 0xc0, 0x72, 0xd9, 0x06, 0x5d, 0x20, 0x0f, 0xc0,
                             0x0b, 0xa0, 0x40, 0x00, 0x5e, 0x1e, 0x1c, 0x26, 0xc0, 0x20, 0x40};

  srsran::unique_byte_buffer_t rrc_nr_buffer = srsran::make_byte_buffer();
  asn1::rrc_nr::dl_dcch_msg_s  dl_dcch_msg   = pack_nas_to_dl_dcch(nas_msg);
  if (!pack_dl_dcch_to_rrc_nr(rrc_nr_buffer, dl_dcch_msg)) {
    logger.error("Failed to pack nas to rrc_nr\n");
    return -1;
  }
  compare_two_buffers(rrc_nr_origin, sizeof(rrc_nr_origin), rrc_nr_buffer->msg, rrc_nr_buffer->N_bytes);

  /* Pack to rlc_nr test */
  uint8_t rlc_nr_origin[] = {0xc0, 0x01, 0x00, 0x01, 0x28, 0x82, 0x6f, 0xc0, 0x72, 0xd9, 0x06, 0x5d, 0x20, 0x0f, 0xc0,
                             0x0b, 0xa0, 0x40, 0x00, 0x5e, 0x1e, 0x1c, 0x26, 0xc0, 0x20, 0x40, 0x00, 0x00, 0x00, 0x00};
  uint8_t rrc_nr_mac[4]   = {0};
  srsran::unique_byte_buffer_t rlc_nr_buffer = srsran::make_byte_buffer();
  pack_rrc_nr_to_rlc_nr(rrc_nr_buffer->msg, rrc_nr_buffer->N_bytes, 1, 1, rrc_nr_mac, rlc_nr_buffer);
  compare_two_buffers(rlc_nr_origin, sizeof(rlc_nr_origin), rlc_nr_buffer->msg, rlc_nr_buffer->N_bytes);

  uint8_t               mac_nr_origin[] = {0x01, 0x1e, 0xc0, 0x01, 0x00, 0x01, 0x28, 0x82, 0x6f, 0xc0, 0x72, 0xd9, 0x06,
                                           0x5d, 0x20, 0x0f, 0xc0, 0x0b, 0xa0, 0x40, 0x00, 0x5e, 0x1e, 0x1c, 0x26, 0xc0,
                                           0x20, 0x40, 0x00, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00};
  srsran::byte_buffer_t mac_nr_buffer;
  pack_rlc_nr_to_mac_nr(rlc_nr_buffer->msg, rlc_nr_buffer->N_bytes, 0, mac_nr_buffer, sizeof(mac_nr_origin));
  compare_two_buffers(mac_nr_origin, sizeof(mac_nr_origin), mac_nr_buffer.msg, mac_nr_buffer.N_bytes);

  /* Dissector of rrc_nr encoded */
  wd_worker->process(mac_nr_buffer.msg, mac_nr_buffer.N_bytes, 0, 0, 0, 0, direction_t::DL, exploit);
}
