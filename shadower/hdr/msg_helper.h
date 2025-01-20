#ifndef MSG_HELPER_H
#define MSG_HELPER_H
#include "srsran/asn1/rrc_nr.h"
#include "srsran/common/byte_buffer.h"
#include "srsran/mac/mac_sch_pdu_nr.h"
#include <string>

/* Put the nas message in to dl_dcch_msg */
asn1::rrc_nr::dl_dcch_msg_s pack_nas_to_dl_dcch(const std::string& nas_msg);

/* Put the dl_dcch msg into rrc nr and encode it */
bool pack_dl_dcch_to_rrc_nr(srsran::unique_byte_buffer_t& buffer, const asn1::rrc_nr::dl_dcch_msg_s& dl_dcch_msg);

void pack_rrc_nr_to_rlc_nr(uint8_t*                      rrc_nr_msg,
                           uint32_t                      rrc_nr_len,
                           uint16_t                      am_sn,
                           uint16_t                      pdcp_sn,
                           uint8_t*                      rrc_mac,
                           srsran::unique_byte_buffer_t& output);

void pack_rlc_nr_to_mac_nr(uint8_t*               rlc_nr_msg,
                           uint32_t               rlc_nr_len,
                           uint16_t               ack_sn,
                           srsran::byte_buffer_t& output,
                           uint32_t               pdu_len = 256);

#endif // MSG_HELPER_H