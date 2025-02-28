cmake_minimum_required(VERSION 3.12)
project(yolov8_openvino_example)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(SDK_ROOT "/home/auto/linuxSDK_V2.1.0.41")

# 假设当前目录为项目根目录，使用相对路径查找 OpenCV
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${OpenMP_INCLUDE_DIRS}
    /opt/intel/openvino_2024/runtime/include
    ${SDK_ROOT}/include
)

add_executable(
detect 
#main.cc
inference.cc
xunhuan.cc

#yibu.cc
)

target_link_libraries(detect
    ${OpenCV_LIBS}
    ${OpenMP_LIBS}
    /opt/intel/openvino_2024/runtime/lib/intel64/libopenvino.so
    ${SDK_ROOT}/lib/x64/libMVSDK.so
)
