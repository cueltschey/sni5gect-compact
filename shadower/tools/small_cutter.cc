#include <complex>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char* argv[])
{
  if (argc < 5) {
    cout << "Usage: cutter <input file> <output file> <skip> <# frames to copy>" << endl;
    return 1;
  }
  string inputFile  = argv[1];
  string outputFile = argv[2];
  long   skip       = atoi(argv[3]);
  long   toCopy     = atoi(argv[4]);

  vector<complex<float> > skip_buffer(skip);
  vector<complex<float> > copy_buffer(toCopy);

  // prepare the input stream object
  ifstream in(inputFile, ios::binary);
  if (!in.is_open()) {
    cout << "[ERROR] Failed to open input file" << endl;
    return 1;
  } else {
    cout << "Input file opened successfully!" << endl;
  }

  // output stream object
  ofstream out(outputFile, ios::binary);
  if (!out.is_open()) {
    cout << "[ERROR] Failed to open output file" << endl;
    return 1;
  } else {
    cout << "Output file opened successfully!" << endl;
  }

  in.read(reinterpret_cast<char*>(skip_buffer.data()), skip_buffer.size() * sizeof(complex<float>));
  printf("Skipped %ld samples\n", skip);
  in.read(reinterpret_cast<char*>(copy_buffer.data()), copy_buffer.size() * sizeof(complex<float>));
  out.write(reinterpret_cast<char*>(copy_buffer.data()), copy_buffer.size() * sizeof(complex<float>));
  in.close();
  out.close();
  printf("Copied %ld samples\n", toCopy);
  return 0;
}