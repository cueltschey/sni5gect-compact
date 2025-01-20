#include <atomic>
#include <complex>
#include <condition_variable>
#include <csignal>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <sched.h>
#include <string>
#include <uhd/usrp/multi_usrp.hpp>
#include <vector>
using namespace std;
atomic<bool> stop_flag(false);
typedef struct {
  double                                        freq;
  double                                        srate;
  uint32_t                                      frames;
  queue<shared_ptr<vector<complex<float> > > >* buffer_queue;
  mutex*                                        mtx;
  condition_variable*                           cv;
  string                                        outputFileName;
} sdr_params_t;

const double gain         = 40;   // dB
const double subframeTime = 1e-3; // 1 ms
const string device_args  = "type=b200";

void* read_from_sdr(void* args)
{
  auto* sdr_params = (sdr_params_t*)args;

  queue<shared_ptr<vector<complex<float> > > >* queue  = sdr_params->buffer_queue;
  double                                        freq   = sdr_params->freq;
  double                                        srate  = sdr_params->srate;
  uint32_t                                      frames = sdr_params->frames;
  mutex*                                        mtx    = sdr_params->mtx;
  condition_variable*                           cv     = sdr_params->cv;

  // create a USRP device
  uhd::usrp::multi_usrp::sptr usrp = uhd::usrp::multi_usrp::make(device_args);
  usrp->set_rx_rate(srate);
  usrp->set_rx_freq(freq);
  usrp->set_rx_gain(gain);
  usrp->set_clock_source("internal");
  const size_t buffer_size = srate * subframeTime;
  // Create a receiver streamer
  uhd::rx_streamer::sptr stream = usrp->get_rx_stream(uhd::stream_args_t("fc32", "sc16"));
  // Create a receive metadata structure
  uhd::rx_metadata_t metadata;
  // Start streaming
  uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
  stream_cmd.num_samps  = 0;
  stream_cmd.stream_now = true;
  stream->issue_stream_cmd(stream_cmd);

  long subFrameCount = 0;

  while (subFrameCount < frames && !stop_flag.load()) {
    try {
      shared_ptr<vector<complex<float> > > buffer = make_shared<vector<complex<float> > >(buffer_size);
      stream->recv(buffer->data(), buffer_size, metadata, 3.0);
      subFrameCount++;
      {
        lock_guard<mutex> lock(*mtx);
        queue->push(buffer);
      }
      cv->notify_one();
    } catch (const exception& e) {
      break;
    }
  }
  {
    lock_guard<mutex> lock(*mtx);
    queue->push(nullptr);
  }
  return nullptr;
}

void write_to_file(void* args)
{
  auto*                                         sdr_params = (sdr_params_t*)args;
  queue<shared_ptr<vector<complex<float> > > >* queue      = sdr_params->buffer_queue;
  mutex*                                        mtx        = sdr_params->mtx;
  condition_variable*                           cv         = sdr_params->cv;

  ofstream outfile(sdr_params->outputFileName, ios::binary);
  if (!outfile.is_open()) {
    cerr << "Failed to open output file" << endl;
    stop_flag.store(true);
    return;
  }
  long numSubframes = 0;
  while (!stop_flag.load()) {
    shared_ptr<vector<complex<float> > > buffer;
    {
      unique_lock<mutex> lock(*mtx);
      cv->wait(lock, [queue] { return !queue->empty(); });
      buffer = queue->front();
      queue->pop();
    }
    outfile.write(reinterpret_cast<char*>(buffer->data()), buffer->size() * sizeof(complex<float>));
    if (numSubframes++ % 50 == 0) {
      printf(".");
      fflush(stdout);
    }
  }
  outfile.close();
}

int main(int argc, char* argv[])
{
  double   centerFrequency = 3424.5e6; // 3424.5 MHz
  double   sampleRate      = 46.08e6;  // 46.08 MHz
  string   outputFile      = "output.fc32";
  uint32_t num_frames      = 1200000;

  if (argc > 1) {
    double centerFrequencyMHz = atof(argv[1]);
    centerFrequency           = centerFrequencyMHz * 1e6;
  }

  if (argc > 2) {
    num_frames = atoi(argv[2]);
  }

  if (argc > 3) {
    outputFile = argv[3];
  }

  if (argc > 4) {
    double sampleRateMHz = atof(argv[4]);
    sampleRate           = sampleRateMHz * 1e6;
  }

  queue<shared_ptr<vector<complex<float> > > > buffer_queue;
  mutex                                        mtx;
  condition_variable                           cv;

  pthread_t    receiver_thread, writer_thread;
  sdr_params_t params = {centerFrequency, sampleRate, num_frames * 10, &buffer_queue, &mtx, &cv, outputFile};
  // create receiver thread
  if (pthread_create(&receiver_thread, nullptr, read_from_sdr, (void*)&params)) {
    cerr << "Error creating receiver thread" << endl;
    return -1;
  }
  // set the receiver thread to the highest priority
  struct sched_param sp {};
  sp.sched_priority = sched_get_priority_max(SCHED_RR);
  if (pthread_setschedparam(receiver_thread, SCHED_RR, &sp) != 0) {
    cerr << "Failed to set thread priority" << endl;
  }
  // set the receiver thread affinity to CPU 2
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (pthread_setaffinity_np(receiver_thread, sizeof(cpuset), &cpuset) != 0) {
    cerr << "Failed to set thread affinity" << endl;
  }
  if (pthread_getaffinity_np(receiver_thread, sizeof(cpuset), &cpuset) != 0) {
    cerr << "Failed to get thread affinity" << endl;
  }
  int nproc = sysconf(_SC_NPROCESSORS_ONLN);
  for (int i = 0; i < nproc; i++) {
    cout << CPU_ISSET(i, &cpuset) << " ";
  }
  cout << endl;
  // create writer thread
  if (pthread_create(&writer_thread, nullptr, reinterpret_cast<void* (*)(void*)>(write_to_file), (void*)&params)) {
    cerr << "Error creating writer thread" << endl;
    return -1;
  }
  // set the writer thread to the lowest priority
  struct sched_param sp2 {};
  sp2.sched_priority = sched_get_priority_min(SCHED_IDLE);
  if (pthread_setschedparam(writer_thread, SCHED_IDLE, &sp2) != 0) {
    cerr << "Failed to set thread priority" << endl;
  }

  pthread_join(receiver_thread, nullptr);
  pthread_join(writer_thread, nullptr);
  return 0;
}
