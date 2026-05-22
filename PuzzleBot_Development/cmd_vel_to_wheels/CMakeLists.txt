cmake_minimum_required(VERSION 3.8)
project(cmd_vel_to_wheels)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(cmd_vel_to_wheels src/cmd_vel_to_wheels.cpp)
ament_target_dependencies(cmd_vel_to_wheels
  rclcpp
  geometry_msgs
  std_msgs
  nav_msgs
  tf2_ros
)

install(TARGETS
  cmd_vel_to_wheels
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
