#include "get_seg_3d_coord/point_selector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/point_tests.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>

namespace get_seg_3d_coord {
namespace {

Eigen::Isometry3f transformToEigen(
    const geometry_msgs::msg::Transform &transform) {
  Eigen::Isometry3f eigen_transform = Eigen::Isometry3f::Identity();
  eigen_transform.translation() =
      Eigen::Vector3f(transform.translation.x, transform.translation.y,
                      transform.translation.z);

  const Eigen::Quaternionf rotation(
      transform.rotation.w, transform.rotation.x, transform.rotation.y,
      transform.rotation.z);
  eigen_transform.linear() = rotation.normalized().toRotationMatrix();
  return eigen_transform;
}

}  // namespace

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
    PointSelectionStats *stats,
    std::string *failure_reason) {
  auto input_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::fromROSMsg(*cloud_msg, *input_cloud);
  if (stats) {
    stats->raw_points += input_cloud->points.size();
  }
  if (input_cloud->empty()) {
    return true;
  }

  const std::string camera_frame =
      config.camera_frame.empty() ? camera_info->header.frame_id
                                  : config.camera_frame;
  if (camera_frame.empty()) {
    if (failure_reason) {
      *failure_reason = "camera frame is empty";
    }
    RCLCPP_WARN_THROTTLE(logger, clock, 2000, "Camera frame is empty.");
    return false;
  }

  geometry_msgs::msg::TransformStamped camera_tf;
  try {
    camera_tf = tf_buffer.lookupTransform(
        camera_frame, cloud_msg->header.frame_id, projection_stamp,
        rclcpp::Duration::from_seconds(config.tf_timeout_sec));
  } catch (const tf2::TransformException &ex) {
    if (failure_reason) {
      *failure_reason = "cannot transform cloud frame '" +
                        cloud_msg->header.frame_id + "' to camera frame '" +
                        camera_frame + "': " + ex.what();
    }
    RCLCPP_WARN_THROTTLE(
        logger, clock, 2000,
        "Cannot transform cloud frame '%s' to camera frame '%s' at mask time: %s",
        cloud_msg->header.frame_id.c_str(), camera_frame.c_str(), ex.what());
    return false;
  }

  const std::string output_frame = config.output_frame;
  if (accumulated_cloud->header.frame_id.empty()) {
    accumulated_cloud->header.frame_id = output_frame;
  }

  Eigen::Isometry3f cloud_to_output = Eigen::Isometry3f::Identity();
  if (!output_frame.empty() && output_frame != cloud_msg->header.frame_id) {
    try {
      const auto out_tf = tf_buffer.lookupTransform(
          output_frame, cloud_msg->header.frame_id, projection_stamp,
          rclcpp::Duration::from_seconds(config.tf_timeout_sec));
      cloud_to_output = transformToEigen(out_tf.transform);
    } catch (const tf2::TransformException &ex) {
      if (failure_reason) {
        *failure_reason = "cannot transform cloud frame '" +
                          cloud_msg->header.frame_id +
                          "' to output frame '" + output_frame + "': " +
                          ex.what();
      }
      RCLCPP_WARN_THROTTLE(
          logger, clock, 2000,
          "Cannot transform cloud frame '%s' to output frame '%s' at mask time: %s",
          cloud_msg->header.frame_id.c_str(), output_frame.c_str(), ex.what());
      return false;
    }
  }

  const Eigen::Isometry3f cloud_to_camera =
      transformToEigen(camera_tf.transform);
  Eigen::Isometry3f cloud_to_prefilter = Eigen::Isometry3f::Identity();
  bool use_prefilter = false;
  if (config.prefilter_cloud && !config.prefilter_frame.empty()) {
    try {
      const auto prefilter_tf = tf_buffer.lookupTransform(
          config.prefilter_frame, cloud_msg->header.frame_id, projection_stamp,
          rclcpp::Duration::from_seconds(config.tf_timeout_sec));
      cloud_to_prefilter = transformToEigen(prefilter_tf.transform);
      use_prefilter = true;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(
          logger, clock, 2000,
          "Cannot transform cloud frame '%s' to prefilter frame '%s'; "
          "continuing without cloud prefilter: %s",
          cloud_msg->header.frame_id.c_str(), config.prefilter_frame.c_str(),
          ex.what());
    }
  }

  const double fx = camera_info->k[0];
  const double fy = camera_info->k[4];
  const double cx = camera_info->k[2];
  const double cy = camera_info->k[5];
  const double x_scale =
      camera_info->width > 0
          ? static_cast<double>(mask.cols) /
                static_cast<double>(camera_info->width)
          : 1.0;
  const double y_scale =
      camera_info->height > 0
          ? static_cast<double>(mask.rows) /
                static_cast<double>(camera_info->height)
          : 1.0;

  PointSelectionStats frame_stats;
  frame_stats.raw_points = input_cloud->points.size();
  for (const auto &point : input_cloud->points) {
    if (!pcl::isFinite(point)) {
      continue;
    }
    ++frame_stats.finite_points;

    const Eigen::Vector3f cloud_point(point.x, point.y, point.z);
    if (use_prefilter) {
      const Eigen::Vector3f prefilter_point =
          cloud_to_prefilter * cloud_point;
      if (prefilter_point.x() < config.prefilter_x_min ||
          prefilter_point.x() > config.prefilter_x_max ||
          prefilter_point.y() < config.prefilter_y_min ||
          prefilter_point.y() > config.prefilter_y_max ||
          prefilter_point.z() < config.prefilter_z_min ||
          prefilter_point.z() > config.prefilter_z_max) {
        continue;
      }
    }
    ++frame_stats.prefiltered_points;

    const Eigen::Vector3f camera_point = cloud_to_camera * cloud_point;
    if (camera_point.z() <= 0.0F) {
      continue;
    }
    ++frame_stats.camera_front_points;

    const double u_orig = fx * camera_point.x() / camera_point.z() + cx;
    const double v_orig = fy * camera_point.y() / camera_point.z() + cy;

    int best_bbox = -1;
    float best_area = std::numeric_limits<float>::infinity();
    for (std::size_t b = 0; b < candidates.size(); ++b) {
      const auto &c = candidates[b];
      if (u_orig < c.u_min || u_orig > c.u_max || v_orig < c.v_min ||
          v_orig > c.v_max) {
        continue;
      }
      const float area = (c.u_max - c.u_min) * (c.v_max - c.v_min);
      if (area < best_area) {
        best_area = area;
        best_bbox = static_cast<int>(b);
      }
    }
    if (best_bbox < 0) {
      continue;
    }
    ++frame_stats.bbox_points;

    const int u_mask = static_cast<int>(std::lround(u_orig * x_scale));
    const int v_mask = static_cast<int>(std::lround(v_orig * y_scale));
    if (u_mask < 0 || u_mask >= mask.cols || v_mask < 0 ||
        v_mask >= mask.rows) {
      continue;
    }
    if (mask.at<unsigned char>(v_mask, u_mask) == 0) {
      continue;
    }
    ++frame_stats.selected_points;

    const Eigen::Vector3f output_point = cloud_to_output * cloud_point;
    pcl::PointXYZ out;
    out.x = output_point.x();
    out.y = output_point.y();
    out.z = output_point.z();
    accumulated_cloud->points.push_back(out);
    per_point_bbox.push_back(best_bbox);
  }

  if (stats) {
    stats->finite_points += frame_stats.finite_points;
    stats->prefiltered_points += frame_stats.prefiltered_points;
    stats->camera_front_points += frame_stats.camera_front_points;
    stats->bbox_points += frame_stats.bbox_points;
    stats->selected_points += frame_stats.selected_points;
  }

  RCLCPP_INFO(logger,
              "local_search cloud frame: raw=%zu finite=%zu prefiltered=%zu "
              "camera_front=%zu bbox=%zu bbox_mask=%zu cumulative=%zu",
              frame_stats.raw_points, frame_stats.finite_points,
              frame_stats.prefiltered_points, frame_stats.camera_front_points,
              frame_stats.bbox_points, frame_stats.selected_points,
              accumulated_cloud->points.size());
  return true;
}

}  // namespace get_seg_3d_coord
