#include "shadower/comp/workers/influx_worker.h"

InfluxWorker::InfluxWorker(srslog::basic_logger& logger_, Source* source_, ShadowerConfig& config_) :
  logger(logger_), config(config_) {}

InfluxWorker::~InfluxWorker() {}

bool InfluxWorker::init()
{
  std::lock_guard<std::mutex> lock(mutex);
	influx_server_info = influxdb_cpp::server_info(config.url, config.port, config.org, config.token, config.bucket);

  return influx_server_info.resp_;
}

template <typename T>
bool InfluxWorker::push_msg(const T& data){
  std::lock_guard<std::mutex> lock(mutex);
	msg_queue.push(data);
	cv.notify_one();
	return true;
}

void InfluxWorker::work_imp(){
	while(true){
		std::lock_guard<std::mutex> lock(mutex);
		cv.wait(lock, [this](){ return !message_queue.empty(); })

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
	logger.log(srslog::level::info, "Sending MIB...");
}

bool InfluxWorker::send_sib1(const asn1::rrc_nr::sib1_s& sib1){
	logger.log(srslog::level::info, "Sending SIB1...");
}
