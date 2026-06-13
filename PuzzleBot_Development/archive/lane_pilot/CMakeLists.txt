cmake_minimum_required(VERSION 3.8)
project(lane_pilot)

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
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Perception: image -> metric centerline points (base_link)
add_executable(lane_ipm_node
  src/lane_ipm_node.cpp
  src/ipm.cpp
  src/lane_detector.cpp
)
target_include_directories(lane_ipm_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_ipm_node ${OpenCV_LIBS})
ament_target_dependencies(lane_ipm_node
  rclcpp sensor_msgs std_msgs geometry_msgs)

# Control: odom-frame memory + regulated pure pursuit -> /cmd_vel
add_executable(lane_pilot_node
  src/lane_pilot_node.cpp
)
target_include_directories(lane_pilot_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_pilot_node ${OpenCV_LIBS})
ament_target_dependencies(lane_pilot_node
  rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs)

install(TARGETS
  lane_ipm_node
  lane_pilot_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# Calibration / inspection tools (run with `ros2 run lane_pilot <tool>.py`)
install(PROGRAMS
  tools/calibrate_ground_homography.py
  tools/check_homography.py
  tools/grab_frame.py
  tools/tune_lane.py
  tools/lane_debug_ui.py
  tools/lane_debug_qt.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
