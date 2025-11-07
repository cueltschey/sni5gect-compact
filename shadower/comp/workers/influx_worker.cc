#include "shadower/comp/workers/influx_worker.h"

InfluxWorker::InfluxWorker(srslog::basic_logger& logger_, const DatabaseConfig config_) :
  logger(logger_), influx_server_info(config_.host, config_.port, config_.org, config_.token, config_.bucket), data_id(config_.data_id){
  if(influx_server_info.resp_ != 0){
    logger.error(RED "Failed to connect to InfluxDB" RESET);
  }
}

bool InfluxWorker::work(){
	std::unique_lock<std::mutex> lock(mutex);
	cv.wait(lock, [this](){ return !msg_queue.empty(); });

	auto msg_variant = msg_queue.front();
	msg_queue.pop();

	if(std::holds_alternative<srsran_mib_nr_t>(msg_variant)){
		return send_mib(std::get<srsran_mib_nr_t>(msg_variant));
	} else if(std::holds_alternative<asn1::rrc_nr::sib1_s>(msg_variant)){
		return send_sib1(std::get<asn1::rrc_nr::sib1_s>(msg_variant));
	} else if(std::holds_alternative<influx_band_report_t>(msg_variant)){
		return send_band_report(std::get<influx_band_report_t>(msg_variant));
	} else if(std::holds_alternative<ChannelConfig>(msg_variant)){
		return send_channel_config(std::get<ChannelConfig>(msg_variant));
	}

	logger.warning(YELLOW "Attempted to push unsupported data type to InfluxDB" RESET);
	return false;
}

bool InfluxWorker::send_channel_config(const ChannelConfig& ch){
	logger.info(GREEN "Sending channel config as %s" RESET, data_id.c_str());

	std::string response_text;
	influxdb_cpp::builder()
        .meas("channel_config")
        .tag("sni5gect_data_id", data_id)

        .field("rx_frequency", (double)ch.rx_frequency)
        .field("tx_frequency", (double)ch.tx_frequency)
        .field("rx_offset", (double)ch.rx_offset)
        .field("tx_offset", (double)ch.tx_offset)
        .field("rx_gain", (double)ch.rx_gain)
        .field("tx_gain", (double)ch.tx_gain)
        .field("enabled", (bool)ch.enabled)

        .post_http(influx_server_info, &response_text);


	if (response_text.length() > 0) {
		logger.error(RED "Recieved error from influxdb: %s" RESET, response_text.c_str());
		return false;
	}
	return true;
}

bool InfluxWorker::send_band_report(const influx_band_report_t& report){
	logger.info(GREEN "Sending band report as %s" RESET, data_id.c_str());

	std::string response_text;
	influxdb_cpp::builder()
        .meas("band_report")
        .tag("sni5gect_data_id", data_id)

        .field("band", (long long)report.band)
        .field("nof_prb", (long long)report.nof_prb)
        .field("offset_to_carrier", (long long)report.offset_to_carrier)
        .field("scs_common", std::string(srsran_subcarrier_spacing_to_str(report.scs_common)))
        .field("scs_ssb", std::string(srsran_subcarrier_spacing_to_str(report.scs_ssb)))
        .field("dl_arfcn", (long long)report.dl_arfcn)
        .field("ul_arfcn", (long long)report.ul_arfcn)
        .field("ssb_arfcn", (long long)report.ssb_arfcn)
        .field("ul_freq", (double)report.ul_freq)
        .field("dl_freq", (double)report.dl_freq)
        .field("ssb_freq", (double)report.ssb_freq)
        .field("ssb_pattern", std::string(srsran_ssb_pattern_to_str(report.ssb_pattern)))
        .field("sample_rate", (double)report.sample_rate)
        .field("uplink_cfo", (double)report.uplink_cfo)
        .field("downlink_cfo", (double)report.downlink_cfo)

        .post_http(influx_server_info, &response_text);


	if (response_text.length() > 0) {
		logger.error(RED "Recieved error from influxdb: %s" RESET, response_text.c_str());
		return false;
	}
	return true;
}

bool InfluxWorker::send_mib(const srsran_mib_nr_t& mib){
	logger.info(GREEN "Sending MIB as %s" RESET, data_id.c_str());

	std::string response_text;
	influxdb_cpp::builder()
        .meas("mib")
        .tag("sni5gect_data_id", data_id)

        .field("sfn", (long long)mib.sfn)
        .field("ssb_idx", (long long)mib.ssb_idx)
        .field("hrf", (bool)mib.hrf)
        .field("scs_common", (int)mib.scs_common)
        .field("ssb_offset", (long long)mib.ssb_offset)
        .field("dmrs_typeA_pos", (int)mib.dmrs_typeA_pos)
        .field("coreset0_idx", (long long)mib.coreset0_idx)
        .field("ss0_idx", (long long)mib.ss0_idx)
        .field("cell_barred", (bool)mib.cell_barred)
        .field("intra_freq_reselection", (bool)mib.intra_freq_reselection)
        .field("spare", (long long)mib.spare)

        .post_http(influx_server_info, &response_text);


	if (response_text.length() > 0) {
		logger.error(RED "Recieved error from influxdb: %s" RESET, response_text.c_str());
		return false;
	}
	return true;
}

bool InfluxWorker::send_sib1(const asn1::rrc_nr::sib1_s& sib1){
	logger.info(GREEN "Sending SIB1 as %s" RESET, data_id.c_str());

	std::string response_text;
	influxdb_cpp::builder()
        .meas("prach_cfg")
        .tag("sni5gect_data_id", data_id)

        .field("config_idx", (int) sib1.serving_cell_cfg_common.ul_cfg_common.init_ul_bwp.rach_cfg_common.setup().rach_cfg_generic.prach_cfg_idx)
        .field("root_seq_idx", (int) sib1.serving_cell_cfg_common.ul_cfg_common.init_ul_bwp.rach_cfg_common.setup().prach_root_seq_idx.l839())
        .field("zero_corr_zone", (int) sib1.serving_cell_cfg_common.ul_cfg_common.init_ul_bwp.rach_cfg_common.setup().rach_cfg_generic.zero_correlation_zone_cfg)
        .field("freq_offset", (int) sib1.serving_cell_cfg_common.ul_cfg_common.init_ul_bwp.rach_cfg_common.setup().rach_cfg_generic.msg1_freq_start)
        .field("num_ra_preambles", (int) sib1.serving_cell_cfg_common.ul_cfg_common.init_ul_bwp.rach_cfg_common.setup().total_nof_ra_preambs)

        .post_http(influx_server_info, &response_text);


	if (response_text.length() > 0) {
		logger.error(RED "Recieved error from influxdb: %s" RESET, response_text.c_str());
		return false;
	}

	return true;
}
