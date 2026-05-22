cmake_minimum_required(VERSION 3.8)
project(traffic_light)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(detect
  src/detect.cpp
  src/traffic_light_detection.cpp
)
target_include_directories(detect PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(detect ${OpenCV_LIBS})
ament_target_dependencies(detect rclcpp sensor_msgs std_msgs)

install(TARGETS
  detect
  DESTINATION lib/${PROJECT_NAME}
)

# Preserve the existing `ros2 run traffic_light flask_stream` entry point.
install(PROGRAMS
  scripts/flask_stream
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
