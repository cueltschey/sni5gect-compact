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
	} 
	logger.warning(YELLOW "Unknown InfluxDB type supplied" RESET);
	return false;
}

// TODO: send data fully
bool InfluxWorker::send_mib(const srsran_mib_nr_t& mib){
	logger.info(GREEN "Sending MIB as %s" RESET, data_id.c_str());

	std::string response_text;
	influxdb_cpp::builder()
        .meas("rtue_carrier_metric")
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

	/*
	std::string response_text;
	influxdb_cpp::builder()
        .meas("rtue_carrier_metric")
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
	*/
	return true;
}
