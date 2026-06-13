cmake_minimum_required(VERSION 3.8)
project(line_follower)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(follow_line src/follow_line.cpp)
target_include_directories(follow_line PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(follow_line ${OpenCV_LIBS})
ament_target_dependencies(follow_line rclcpp sensor_msgs geometry_msgs)

add_executable(follow_single_line src/follow_single_line.cpp)
target_include_directories(follow_single_line PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(follow_single_line ${OpenCV_LIBS})
ament_target_dependencies(follow_single_line
  rclcpp sensor_msgs geometry_msgs nav_msgs std_msgs)

install(TARGETS
  follow_line
  follow_single_line
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
