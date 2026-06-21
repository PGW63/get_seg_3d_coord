#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/clock.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>

#include "get_seg_3d_coord/local_search_types.hpp"

namespace get_seg_3d_coord {

struct PointSelectionConfig {
  std::string camera_frame;
  std::string output_frame;
  double tf_timeout_sec{0.2};
  bool prefilter_cloud{false};
  std::string prefilter_frame{"base"};
  double prefilter_x_min{0.0};
  double prefilter_x_max{5.0};
  double prefilter_y_min{-2.0};
  double prefilter_y_max{2.0};
  double prefilter_z_min{-0.05};
  double prefilter_z_max{2.0};
};

struct PointSelectionStats {
  std::size_t raw_points{0};
  std::size_t finite_points{0};
  std::size_t prefiltered_points{0};
  std::size_t camera_front_points{0};
  std::size_t bbox_points{0};
  std::size_t selected_points{0};
};

bool collectMaskProjectedPoints(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr &camera_info,
    const cv::Mat &mask,
    const std::vector<BBoxCandidate> &candidates,
    const rclcpp::Time &projection_stamp,
    const PointSelectionConfig &config,
    tf2_ros::Buffer &tf_buffer,
    const rclcpp::Logger &logger,
    rclcpp::Clock &clock,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &accumulated_cloud,
    std::vector<int> &per_point_bbox,
    PointSelectionStats *stats = nullptr,
    std::string *failure_reason = nullptr);

}  // namespace get_seg_3d_coord
