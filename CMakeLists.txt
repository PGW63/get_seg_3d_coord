cmake_minimum_required(VERSION 3.8)
project(get_seg_3d_coord)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io segmentation search kdtree)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(segmented_lidar_cluster_node
  src/segmented_lidar_cluster_node.cpp
)

target_include_directories(segmented_lidar_cluster_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(segmented_lidar_cluster_node PUBLIC cxx_std_17)

target_link_libraries(segmented_lidar_cluster_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
  ${OpenCV_LIBS}
)

target_compile_definitions(segmented_lidar_cluster_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

ament_target_dependencies(segmented_lidar_cluster_node
  cv_bridge
  geometry_msgs
  message_filters
  pcl_conversions
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  visualization_msgs
)

install(
  TARGETS segmented_lidar_cluster_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  PROGRAMS
    scripts/segmented_lidar_cluster_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
