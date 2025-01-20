#include <complex>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;
long sampleRate = 23040000;
long frameSize  = sampleRate * 10e-3;

int main(int argc, char* argv[])
{
  // parse command line arguments
  if (argc < 5) {
    cout << "Usage: cutter <input file> <output file> <skip> <# frames to copy>" << endl;
    return 1;
  }
  string inputFile  = argv[1];
  string outputFile = argv[2];
  float  skip       = atof(argv[3]);
  if (skip < 0) {
    cout << "[ERROR] Skip should be greater than or equal to 0!" << endl;
    return 1;
  }
  long framesToCopy = atol(argv[4]);
  if (framesToCopy <= 0) {
    cout << "[ERROR] Frames to copy should be greater than 0!" << endl;
    return 1;
  }

  if (argc > 4) {
    float sampleRateMHz = atof(argv[5]);
    sampleRate          = sampleRateMHz * 1e6;
    frameSize           = sampleRate * 10e-3;
  }

  // prepare the input stream object
  ifstream in(inputFile, ios::binary);
  if (!in.is_open()) {
    cout << "[ERROR] Failed to open input file" << endl;
    return 1;
  } else {
    cout << "Input file opened successfully!" << endl;
  }
  // prepare the output stream object
  ofstream out(outputFile, ios::binary);
  if (!out.is_open()) {
    cout << "[ERROR] Failed to open output file" << endl;
    return 1;
  }

  // Print the copy information
  long samplesToSkip = frameSize * skip;
  printf(" * * *  Skiped Frames: %.3f \tSamples: %ld\n", skip, samplesToSkip);
  long samplesToCopy = frameSize * framesToCopy;
  printf(" * * *  Copied Frames: %ld \tSamples: %ld\n", framesToCopy, samplesToCopy);

  // initialize offset to 0
  long                    offset = 0;
  long                    round  = samplesToSkip / 4096;
  vector<complex<float> > skipBuffer(4096);
  for (long i = 0; i < round; i++) {
    in.read(reinterpret_cast<char*>(skipBuffer.data()), skipBuffer.size() * sizeof(complex<float>));
    offset += skipBuffer.size();
    streamsize read = in.gcount();
    if (read < static_cast<std::streamsize>(skipBuffer.size())) {
      cout << "[ERROR] Reached end of file, skip failed, read: " << read << "Skipped: " << offset << endl;
      return 1;
    }
  }
  long remaining = samplesToSkip % 4096;
  if (remaining > 0) {
    in.read(reinterpret_cast<char*>(skipBuffer.data()), remaining * sizeof(complex<float>));
    offset += remaining;
    streamsize read = in.gcount();
    if (read < static_cast<std::streamsize>(skipBuffer.size())) {
      cout << "[ERROR] Reached end of file, skip remaining failed, read " << read << endl;
      return 1;
    }
  }

  vector<complex<float> > buffer(frameSize);
  for (long i = 0; i < framesToCopy; i++) {
    in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(complex<float>));
    streamsize read = in.gcount();
    if (read < static_cast<std::streamsize>(skipBuffer.size())) {
      cout << "[ERROR] Reached end of file, copy failed, read " << read << endl;
      return 1;
    }
    out.write(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(complex<float>));
    offset += buffer.size();
  }
  in.close();
  out.close();
  printf("Expected output file size: %zd MB\n", samplesToCopy * sizeof(complex<float>) / 1024 / 1024);
  return 0;
}