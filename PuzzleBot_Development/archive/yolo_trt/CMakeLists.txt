cmake_minimum_required(VERSION 3.8)
project(yolo_trt)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# CUDA (provides cudart for cudaMalloc/cudaMemcpy/streams).
find_package(CUDA REQUIRED)

# TensorRT (system install from JetPack; not a rosdep).
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include)
find_library(NVINFER nvinfer HINTS /usr/lib/aarch64-linux-gnu)
find_library(NVINFER_PLUGIN nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER)
  message(FATAL_ERROR "TensorRT not found (NvInfer.h / libnvinfer).")
endif()

add_executable(yolo_trt_node src/yolo_trt_node.cpp)
target_include_directories(yolo_trt_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR})
target_link_libraries(yolo_trt_node
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${NVINFER}
  ${NVINFER_PLUGIN})
ament_target_dependencies(yolo_trt_node rclcpp sensor_msgs)

install(TARGETS yolo_trt_node DESTINATION lib/${PROJECT_NAME})

ament_package()
