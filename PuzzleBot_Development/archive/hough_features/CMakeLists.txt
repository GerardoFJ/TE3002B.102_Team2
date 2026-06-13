cmake_minimum_required(VERSION 3.8)
project(hough_features)

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

add_executable(hough_feature_node
  src/hough_feature_node.cpp
  src/hough_detector.cpp
)
target_include_directories(hough_feature_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hough_feature_node ${OpenCV_LIBS})
ament_target_dependencies(hough_feature_node rclcpp sensor_msgs std_msgs)

install(TARGETS
  hough_feature_node
  DESTINATION lib/${PROJECT_NAME}
)

# Sidecar JSON so Phase 2's Python ml_steering can read the feature order
# without having to parse a C++ header. Installed to
# share/hough_features/feature_index.json.
install(FILES
  share/feature_index.json
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
