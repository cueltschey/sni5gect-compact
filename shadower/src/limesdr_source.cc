#include "shadower/hdr/limesdr_source.h"

/* Initialize the radio object and apply the configurations */
LimeSDRSource::LimeSDRSource(std::string device_args,
                             double      srate_,
                             double      rx_freq,
                             double      tx_freq,
                             double      rx_gain,
                             double      tx_gain) :
  lime(new LimePluginContext()), srate(srate_)
{
  logger.set_level(srslog::basic_levels::info);

  lime->samplesFormat = lime::DataFormat::F32;
  lime->rxChannels.resize(number_of_channels);
  lime->txChannels.resize(number_of_channels);
  LimeParamProvider configProvider(device_args.c_str());
  if (LimePlugin_Init(lime, log_callback, &configProvider) != 0) {
    throw std::runtime_error("Failed to initialize LimeSDR");
  }
  state.rx.freq.resize(number_of_channels);
  state.rx.gain.resize(number_of_channels);
  state.rx.bandwidth.resize(number_of_channels);

  state.tx.freq.resize(number_of_channels);
  state.tx.gain.resize(number_of_channels);
  state.tx.bandwidth.resize(number_of_channels);

  state.rf_ports.resize(number_of_channels);

  set_srate(srate);
  set_rx_gain(rx_gain);
  set_tx_gain(tx_gain);
  set_rx_freq(rx_freq + 21.5e3); // TODO: Apply hardware CFO correction after 1st SIB search if needed (>1000 Hz)
  set_tx_freq(tx_freq + 21.5e3);
  // set_rx_freq(rx_freq);
  // set_tx_freq(tx_freq);
}

void LimeSDRSource::close()
{
  int status = LimePlugin_Destroy(lime);
  if (status != 0) {
    logger.error("Failed to close LimeSDR");
  }
}

void LimeSDRSource::set_srate(double sample_rate)
{
  state.rf_ports.resize(number_of_channels);
  state.rf_ports.assign(number_of_channels, {sample_rate, number_of_channels, number_of_channels});
  for (auto& bw : state.rx.bandwidth) {
    bw = sample_rate;
  }
  for (auto& bw : state.tx.bandwidth) {
    bw = sample_rate;
  }
}

void LimeSDRSource::set_tx_gain(double gain)
{
  for (size_t ch = 0; ch < lime->txChannels.size(); ++ch)
    state.tx.gain.at(ch) = gain;
}

void LimeSDRSource::set_rx_gain(double gain)
{
  for (size_t ch = 0; ch < lime->rxChannels.size(); ++ch)
    state.rx.gain.at(ch) = gain;
}

void LimeSDRSource::set_tx_freq(double freq)
{
  for (size_t ch = 0; ch < lime->txChannels.size(); ++ch)
    state.tx.freq.at(ch) = freq;
}

void LimeSDRSource::set_rx_freq(double freq)
{
  for (size_t ch = 0; ch < lime->rxChannels.size(); ++ch)
    state.rx.freq.at(ch) = freq;
}

int LimeSDRSource::receive(cf_t* buffer, uint32_t nof_samples, srsran_timestamp_t* ts)
{
  lime::complex32f_t* dest[SRSRAN_MAX_PORTS] = {0};
  dest[0]                                    = (lime::complex32f_t*)buffer;
  lime::StreamMeta meta;
  meta.waitForTimestamp  = false;
  static bool rx_started = false;
  if (!rx_started) {
    if (LimePlugin_Setup(lime, &state))
      return SRSRAN_ERROR;

    if (LimePlugin_Start(lime))
      return SRSRAN_ERROR_CANT_START;

    rx_started = true;
  }

  int samples_received = LimePlugin_Read_complex32f(lime, dest, nof_samples, 0, meta);
  if (samples_received < 0) {
    logger.error("Failed to receive samples");
    return -1;
  }
  double total_secs = (double)meta.timestamp / srate;
  ts->full_secs     = static_cast<time_t>(total_secs);
  ts->frac_secs     = double(total_secs - ts->full_secs);
  return samples_received;
}

/* TODO implement send the IQ samples at the scheduled time */
int LimeSDRSource::send(cf_t* samples, uint32_t length, srsran_timestamp_t& tx_time, uint32_t slot)
{
  lime::StreamMeta meta{};
  meta.timestamp          = 0;
  meta.waitForTimestamp   = true;
  meta.flushPartialPacket = true;

  if (true) {
    meta.timestamp = srsran_timestamp_uint64(&tx_time, state.rf_ports[0].sample_rate);
    if (tx_time.full_secs < 0)
      return SRSRAN_ERROR;
    if (isnan(tx_time.frac_secs))
      return SRSRAN_ERROR;
  }

  lime::complex32f_t* src[SRSRAN_MAX_CHANNELS] = {};
  for (size_t ch = 0; ch < lime->txChannels.size(); ++ch)
    src[ch] = (lime::complex32f_t*)samples;

  int samplesSent =
      LimePlugin_Write_complex32f(lime, reinterpret_cast<const lime::complex32f_t* const*>(src), length, 0, meta);

  if (samplesSent < 0)
    return SRSRAN_ERROR;

  return samplesSent;
}