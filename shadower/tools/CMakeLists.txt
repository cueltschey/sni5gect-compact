cmake_minimum_required(VERSION 3.10)
project(SDRRecorder LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
find_package(UHD REQUIRED)
set(UHD_BOOST_REQUIRED_COMPONENTS
    program_options
    system
    thread
)
set(BOOST_MIN_VERSION 1.65)
include(UHDBoost)
include_directories(
    ${Boost_INCLUDE_DIRS}
    ${UHD_INCLUDE_DIRS}
)
link_directories(${Boost_LIBRARY_DIRS})
add_executable(recorder recorder.cc)
target_link_libraries(recorder ${UHD_LIBRARIES} ${Boost_LIBRARIES})
target_include_directories(recorder PRIVATE ${UHD_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -pedantic")
