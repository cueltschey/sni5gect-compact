FROM ubuntu:22.04 AS deps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="America/Chicago"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# OPTIONAL: change mirror for faster downloads
# RUN sed -i 's|archive.ubuntu.com|mirror.enzu.com|g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gnupg && \
    add-apt-repository ppa:apt-fast/stable -y  && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends aria2 apt-fast && \
    sed -i "s|_MAXNUM=.*|_MAXNUM=$(nproc)|g" /etc/apt-fast.conf

RUN apt-fast -y install git vim wget curl iproute2 \
    cmake make ninja-build build-essential init gdb adb libuhd-dev uhd-host libfftw3-dev libmbedtls-dev \
    libboost-program-options-dev libconfig++-dev libsctp-dev \
    libyaml-cpp-dev pkg-config  libgtest-dev gcc g++ sudo libzmq3-dev libliquid-dev python3-pip python3-dev \
    software-properties-common kmod bc gzip zstd flex bison swig graphviz libglib2.0-dev libgcrypt20-dev \
    libsnappy-dev libgnutls28-dev libxml2-dev libnghttp2-dev libkrb5-dev libnl-3-dev libspandsp-dev \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libnl-genl-3-dev libnl-route-3-dev \
    libnl-nf-3-dev libcap-dev libbrotli-dev libsmi2-dev liblua5.2-dev libc-ares-dev libsbc-dev \
    libspeexdsp-dev libfreetype6-dev libxss1 libtbb-dev libnss3-dev libudev-dev libpulse-dev \
    libpcre2-dev libasound2-dev libgl1-mesa-dev libssh-dev libmaxminddb-dev libopus-dev \
    libusb-1.0-0 psmisc sshpass libfmt-dev libsodium-dev xxd libc-bin libmbim-glib-dev \
    libgoogle-glog-dev libzstd-dev libevent-dev libunwind-dev libdouble-conversion-dev libgflags-dev \
    qtbase5-dev libqt5multimedia5 libqt5svg5 qttools5-dev qtmultimedia5-dev g++ software-properties-common \
    kmod libglib2.0-dev libsnappy1v5 libsmi2ldbl liblua5.2-0 libc-ares2 libnl-route-3-200 libnl-genl-3-200 \
    libfreetype6 graphviz libtbb12 libxss1 libnss3 libspandsp2 libsbc1 libbrotli1 libnghttp2-14 \
    psmisc sshpass libpulse0 libpcre2-dev libmaxminddb0 libopus0 libspeex1 bc tcpdump libgflags2.2 \
    libzstd1 libunwind8 libcap2 libspeexdsp1 libxtst6 libatk-bridge2.0-0 libusb-1.0-0 meson \
    libnl-genl-3-200 libgraphviz-dev liblz4-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install llvm 15
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 15 && rm llvm.sh
#RUN arch=$(dpkg --print-architecture) \
#    && wget https://security.debian.org/debian-security/pool/updates/main/o/openssl/libssl1.1_1.1.1w-0+deb11u3_${arch}.deb \
#    && dpkg -i libssl1.1_1.1.1w-0+deb11u3_${arch}.deb \
#    && rm libssl1.1_1.1.1w-0+deb11u3_${arch}.deb

# Clone and build wdissector
FROM deps AS wdissector
RUN git clone https://github.com/asset-group/5ghoul-5g-nr-attacks /root/wdissector
WORKDIR /root/wdissector
RUN ./scripts/apply_patches.sh 3rd-party
RUN cd 3rd-party/ && git clone https://github.com/zeromq/libzmq.git --depth=1 && cd libzmq && \
    ./autogen.sh && ./configure && make -j && make install && ldconfig
RUN cd 3rd-party/ && git clone https://github.com/zeromq/czmq.git --depth=1 && cd czmq && \
        ./autogen.sh && ./configure && make -j && make install && ldconfig
RUN cd 3rd-party/ && git clone https://github.com/json-c/json-c.git --depth=1 && cd json-c && \
    mkdir -p build && cd build && cmake ../ && make -j && make install && ldconfig
RUN curl https://raw.githubusercontent.com/jckarter/tbb/refs/heads/master/include/tbb/tbb_stddef.h -o /usr/include/tbb/tbb_stddef.h
RUN sed -i 's/\blong[[:space:]]\+gettid()/__pid_t gettid()/g' /root/wdissector/src/MiscUtils.hpp && \
    sed -i 's/\bif\s*(\s*SWIG_EXIST\s*)/if(NOT SWIG_EXIST)/g' /root/wdissector/CMakeLists.txt && \
    sed -i 's/exit 1/# exit 1/' /root/wdissector/build.sh 
RUN cd /root/wdissector && ./build.sh all


# Copy and build sniffer
FROM wdissector AS sni5gect

COPY . /root/sni5gect
WORKDIR /root/sni5gect

RUN cmake -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
        -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
        --no-warn-unused-cli -B build -G Ninja && \
        ninja -C build
