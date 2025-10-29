#include "shadower/comp/workers/influx_worker.h"

InfluxWorker::InfluxWorker(srslog::basic_logger& logger_, const DatabaseConfig config_) :
  logger(logger_), influx_server_info(config_.host, config_.port, config_.org, config_.token, config_.bucket) {}

InfluxWorker::~InfluxWorker() {}

void InfluxWorker::work_imp(){
	while(true){
		std::unique_lock<std::mutex> lock(mutex);
		cv.wait(lock, [this](){ return !msg_queue.empty(); });

		auto msg_variant = msg_queue.front();
		msg_queue.pop();

		if(std::holds_alternative<srsran_mib_nr_t>(msg_variant)){
			send_mib(std::get<srsran_mib_nr_t>(msg_variant));
		} else if(std::holds_alternative<asn1::rrc_nr::sib1_s>(msg_variant)){
			send_sib1(std::get<asn1::rrc_nr::sib1_s>(msg_variant));
		}
	}
}

// TODO: send data fully
bool InfluxWorker::send_mib(const srsran_mib_nr_t& mib){
	logger.info(RED "Sending MIB..." RESET);
	return true;
}

bool InfluxWorker::send_sib1(const asn1::rrc_nr::sib1_s& sib1){
	logger.info(RED "Sending SIB1..." RESET);
	return true;
}
