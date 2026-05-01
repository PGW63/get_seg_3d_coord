#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/point_tests.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "inha_interfaces/action/local_search.hpp"

class LocalSearchActionServer : public rclcpp::Node {
public:
  using LocalSearch = inha_interfaces::action::LocalSearch;
  using GoalHandle = rclcpp_action::ServerGoalHandle<LocalSearch>;

  LocalSearchActionServer()
      : Node("local_search"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    mask_topic_ = this->declare_parameter<std::string>(
        "mask_topic", "/sam2_segmentation_node/binary_mask/compressed");
    pointcloud_topic_ =
        this->declare_parameter<std::string>("pointcloud_topic", "/livox/lidar");
    camera_info_topic_ = this->declare_parameter<std::string>(
        "camera_info_topic", "/camera/camera_head/color/camera_info");
    filtered_cloud_topic_ = this->declare_parameter<std::string>(
        "filtered_cloud_topic", "/local_search/points");
    centers_topic_ = this->declare_parameter<std::string>(
        "centers_topic", "/local_search/cluster_centers");
    clustered_cloud_topic_ = this->declare_parameter<std::string>(
        "clustered_cloud_topic", "/local_search/clustered_points");
    markers_topic_ = this->declare_parameter<std::string>(
        "markers_topic", "/local_search/cluster_markers");
    action_name_ =
        this->declare_parameter<std::string>("action_name", "local_search");
    camera_frame_param_ =
        this->declare_parameter<std::string>("camera_frame", "");
    mask_threshold_ = this->declare_parameter<int>("mask_threshold", 0);
    cluster_tolerance_ =
        this->declare_parameter<double>("cluster_tolerance", 0.30);
    min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 8);
    max_cluster_size_ =
        this->declare_parameter<int>("max_cluster_size", 30000);
    input_timeout_sec_ =
        this->declare_parameter<double>("input_timeout_sec", 1.0);
    lidar_frame_timeout_sec_ =
        this->declare_parameter<double>("lidar_frame_timeout_sec", 1.0);
    max_decay_frames_ = this->declare_parameter<int>("max_decay_frames", 100);
    decay_num_ = this->declare_parameter<int>("decay_num", 1);
    tf_timeout_sec_ = this->declare_parameter<double>("tf_timeout_sec", 0.1);
    decay_num_ = std::clamp(decay_num_, 1, max_decay_frames_);

    auto best_effort_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    mask_sub_ = this->create_subscription<MaskMsg>(
        mask_topic_, best_effort_qos,
        std::bind(&LocalSearchActionServer::maskCallback, this,
                  std::placeholders::_1));
    cloud_sub_ = this->create_subscription<PointCloud2Msg>(
        pointcloud_topic_, best_effort_qos,
        std::bind(&LocalSearchActionServer::cloudCallback, this,
                  std::placeholders::_1));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, reliable_qos,
        std::bind(&LocalSearchActionServer::cameraInfoCallback, this,
                  std::placeholders::_1));

    filtered_cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            filtered_cloud_topic_, best_effort_qos);
    centers_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
        centers_topic_, reliable_qos);
    clustered_cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            clustered_cloud_topic_, reliable_qos);
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        markers_topic_, reliable_qos);

    action_server_ = rclcpp_action::create_server<LocalSearch>(
        this, action_name_,
        std::bind(&LocalSearchActionServer::handleGoal, this,
                  std::placeholders::_1, std::placeholders::_2),
        std::bind(&LocalSearchActionServer::handleCancel, this,
                  std::placeholders::_1),
        std::bind(&LocalSearchActionServer::handleAccepted, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
                "LocalSearch action node started. action=%s, mask=%s, "
                "cloud=%s, camera_info=%s",
                action_name_.c_str(), mask_topic_.c_str(),
                pointcloud_topic_.c_str(), camera_info_topic_.c_str());
  }

private:
  using MaskMsg = sensor_msgs::msg::CompressedImage;
  using PointCloud2Msg = sensor_msgs::msg::PointCloud2;

  rclcpp_action::GoalResponse
  handleGoal(const rclcpp_action::GoalUUID &,
             std::shared_ptr<const LocalSearch::Goal> goal) {
    if (goal->num_objects <= 0) {
      RCLCPP_WARN(this->get_logger(),
                  "Rejecting goal: num_objects must be greater than zero.");
      return rclcpp_action::GoalResponse::REJECT;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    if (goal_running_ || collecting_) {
      RCLCPP_WARN(this->get_logger(),
                  "Rejecting goal because another goal is already running.");
      return rclcpp_action::GoalResponse::REJECT;
    }
    goal_running_ = true;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse
  handleCancel(const std::shared_ptr<GoalHandle>) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      collecting_ = false;
      goal_running_ = false;
    }
    data_cv_.notify_all();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(const std::shared_ptr<GoalHandle> goal_handle) {
    std::thread{std::bind(&LocalSearchActionServer::executeAction, this,
                          std::placeholders::_1),
                goal_handle}
        .detach();
  }

  void executeAction(const std::shared_ptr<GoalHandle> goal_handle) {
    const auto goal = goal_handle->get_goal();
    const int requested_objects = goal->num_objects;
    auto result = std::make_shared<LocalSearch::Result>();
    auto feedback = std::make_shared<LocalSearch::Feedback>();

    feedback->state = "WAIT_INPUTS";
    feedback->current_num = 0;
    goal_handle->publish_feedback(feedback);

    cv::Mat mask;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;

    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      const bool ready = data_cv_.wait_for(
          lock, std::chrono::duration<double>(input_timeout_sec_),
          [this]() { return has_mask_ && camera_info_ != nullptr; });

      if (!ready || !has_mask_) {
        abortGoal(goal_handle, result, "no seg image");
        return;
      }
      if (!camera_info_) {
        abortGoal(goal_handle, result, "no camera info");
        return;
      }

      mask = latest_mask_.clone();
      camera_info = camera_info_;
      decay_clouds_.clear();
      target_decay_frames_ = decay_num_;
      collecting_ = true;
    }

    feedback->state = "COLLECT_LIDAR";
    goal_handle->publish_feedback(feedback);

    auto clouds = collectLidarFrames(goal_handle, decay_num_);
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      collecting_ = false;
      target_decay_frames_ = 0;
    }
    data_cv_.notify_all();

    if (goal_handle->is_canceling()) {
      goal_running_ = false;
      result->success = false;
      result->message = "canceled";
      goal_handle->canceled(result);
      return;
    }

    if (clouds.empty()) {
      abortGoal(goal_handle, result, "no lidar");
      return;
    }

    feedback->state = "CLUSTER";
    goal_handle->publish_feedback(feedback);

    auto accumulated_cloud = accumulateMaskedClouds(clouds, camera_info, mask);
    const auto cluster_indices = clusterPoints(accumulated_cloud);
    const auto selected_clusters =
        selectLargestClusters(cluster_indices, requested_objects);
    const auto centers = clusterCenters(accumulated_cloud, selected_clusters);
    const auto output_header = clouds.back()->header;

    publishFilteredCloud(output_header, accumulated_cloud);
    publishCenters(output_header, centers);
    publishClusteredCloud(output_header, accumulated_cloud, selected_clusters);
    publishMarkers(output_header, accumulated_cloud, selected_clusters, centers);

    feedback->current_num = static_cast<std::int32_t>(centers.size());
    goal_handle->publish_feedback(feedback);

    for (const auto &center : centers) {
      geometry_msgs::msg::PointStamped point;
      point.header = output_header;
      point.point.x = center.x();
      point.point.y = center.y();
      point.point.z = center.z();
      result->results.push_back(point);
    }

    goal_running_ = false;
    if (centers.empty()) {
      result->success = false;
      result->message = "no points on mask";
      goal_handle->abort(result);
      return;
    }

    if (static_cast<int>(centers.size()) < requested_objects) {
      result->success = false;
      char buf[128];
      std::snprintf(buf, sizeof(buf), "found %zu clusters, need %d",
                    centers.size(), requested_objects);
      result->message = buf;
      goal_handle->abort(result);
      return;
    }

    result->success = true;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "ok: clusters=%zu frames=%zu",
                  centers.size(), clouds.size());
    result->message = buf;
    goal_handle->succeed(result);
  }

  void abortGoal(const std::shared_ptr<GoalHandle> &goal_handle,
                 const std::shared_ptr<LocalSearch::Result> &result,
                 const std::string &message) {
    goal_running_ = false;
    result->success = false;
    result->message = message;
    goal_handle->abort(result);
  }

  std::vector<PointCloud2Msg::ConstSharedPtr> collectLidarFrames(
      const std::shared_ptr<GoalHandle> &goal_handle, int target_frames) {
    std::vector<PointCloud2Msg::ConstSharedPtr> clouds;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(
                              lidar_frame_timeout_sec_ * target_frames);

    std::unique_lock<std::mutex> lock(data_mutex_);
    while (rclcpp::ok() && !goal_handle->is_canceling() &&
           static_cast<int>(clouds.size()) < target_frames) {
      data_cv_.wait_until(lock, deadline, [this, &clouds, target_frames]() {
        return decay_clouds_.size() > clouds.size() ||
               static_cast<int>(decay_clouds_.size()) >= target_frames ||
               !collecting_;
      });

      while (clouds.size() < decay_clouds_.size() &&
             static_cast<int>(clouds.size()) < target_frames) {
        clouds.push_back(decay_clouds_[clouds.size()]);
      }

      if (std::chrono::steady_clock::now() >= deadline || !collecting_) {
        break;
      }
    }

    return clouds;
  }

  void maskCallback(const MaskMsg::ConstSharedPtr msg) {
    cv::Mat mask = imageToBinaryMask(msg);
    if (mask.empty()) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_mask_ = mask;
      has_mask_ = true;
    }
    data_cv_.notify_all();
  }

  void cloudCallback(const PointCloud2Msg::ConstSharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (collecting_ &&
          static_cast<int>(decay_clouds_.size()) < target_decay_frames_) {
        decay_clouds_.push_back(msg);
      }
    }
    data_cv_.notify_all();
  }

  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      camera_info_ = msg;
    }
    data_cv_.notify_all();
  }

  cv::Mat imageToBinaryMask(const MaskMsg::ConstSharedPtr &mask_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(mask_msg);
    } catch (const cv_bridge::Exception &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Failed to convert mask image: %s", ex.what());
      return {};
    }

    cv::Mat gray;
    const cv::Mat &src = cv_ptr->image;
    if (src.channels() == 1) {
      gray = src;
    } else if (src.channels() == 3) {
      cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
      cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Unsupported mask channel count: %d",
                           src.channels());
      return {};
    }

    cv::Mat gray_u8;
    if (gray.depth() == CV_8U) {
      gray_u8 = gray;
    } else {
      gray.convertTo(gray_u8, CV_8U);
    }

    cv::Mat binary;
    cv::threshold(gray_u8, binary, mask_threshold_, 255, cv::THRESH_BINARY);
    return binary;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr accumulateMaskedClouds(
      const std::vector<PointCloud2Msg::ConstSharedPtr> &clouds,
      const sensor_msgs::msg::CameraInfo::SharedPtr &camera_info,
      const cv::Mat &mask) {
    auto accumulated_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    for (const auto &cloud_msg : clouds) {
      auto selected_cloud = filterCloudWithMask(cloud_msg, camera_info, mask);
      accumulated_cloud->points.insert(accumulated_cloud->points.end(),
                                       selected_cloud->points.begin(),
                                       selected_cloud->points.end());
    }

    accumulated_cloud->width =
        static_cast<std::uint32_t>(accumulated_cloud->points.size());
    accumulated_cloud->height = 1;
    accumulated_cloud->is_dense = false;
    return accumulated_cloud;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filterCloudWithMask(
      const PointCloud2Msg::ConstSharedPtr &cloud_msg,
      const sensor_msgs::msg::CameraInfo::SharedPtr &camera_info,
      const cv::Mat &mask) {
    auto lidar_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*cloud_msg, *lidar_cloud);

    if (lidar_cloud->empty()) {
      return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }

    const std::string camera_frame =
        camera_frame_param_.empty() ? camera_info->header.frame_id
                                    : camera_frame_param_;
    if (camera_frame.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Camera frame is empty.");
      return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }

    geometry_msgs::msg::TransformStamped transform;
    try {
      transform = tf_buffer_.lookupTransform(
          camera_frame, cloud_msg->header.frame_id, cloud_msg->header.stamp,
          rclcpp::Duration::from_seconds(tf_timeout_sec_));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Cannot transform cloud frame '%s' to camera frame '%s': %s",
          cloud_msg->header.frame_id.c_str(), camera_frame.c_str(), ex.what());
      return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }

    const Eigen::Isometry3f lidar_to_camera =
        transformToEigen(transform.transform);
    return selectPointsOnMask(lidar_cloud, lidar_to_camera, camera_info, mask);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr selectPointsOnMask(
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &lidar_cloud,
      const Eigen::Isometry3f &lidar_to_camera,
      const sensor_msgs::msg::CameraInfo::SharedPtr &camera_info,
      const cv::Mat &mask) const {
    auto selected = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    selected->header = lidar_cloud->header;
    selected->is_dense = false;

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

    selected->points.reserve(lidar_cloud->points.size());
    for (const auto &point : lidar_cloud->points) {
      if (!pcl::isFinite(point)) {
        continue;
      }

      const Eigen::Vector3f lidar_point(point.x, point.y, point.z);
      const Eigen::Vector3f camera_point = lidar_to_camera * lidar_point;
      if (camera_point.z() <= 0.0F) {
        continue;
      }

      const int u = static_cast<int>(std::lround(
          (fx * camera_point.x() / camera_point.z() + cx) * x_scale));
      const int v = static_cast<int>(std::lround(
          (fy * camera_point.y() / camera_point.z() + cy) * y_scale));

      if (u < 0 || u >= mask.cols || v < 0 || v >= mask.rows) {
        continue;
      }
      if (mask.at<unsigned char>(v, u) == 0) {
        continue;
      }

      selected->points.push_back(point);
    }

    selected->width = static_cast<std::uint32_t>(selected->points.size());
    selected->height = 1;
    return selected;
  }

  std::vector<pcl::PointIndices>
  clusterPoints(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud) const {
    std::vector<pcl::PointIndices> cluster_indices;
    if (static_cast<int>(cloud->points.size()) < min_cluster_size_) {
      return cluster_indices;
    }

    auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> extractor;
    extractor.setClusterTolerance(cluster_tolerance_);
    extractor.setMinClusterSize(min_cluster_size_);
    extractor.setMaxClusterSize(max_cluster_size_);
    extractor.setSearchMethod(tree);
    extractor.setInputCloud(cloud);
    extractor.extract(cluster_indices);
    return cluster_indices;
  }

  std::vector<pcl::PointIndices> selectLargestClusters(
      const std::vector<pcl::PointIndices> &cluster_indices,
      int num_objects) const {
    std::vector<std::size_t> order(cluster_indices.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }

    std::sort(order.begin(), order.end(),
              [&cluster_indices](std::size_t lhs, std::size_t rhs) {
                const auto lhs_size = cluster_indices[lhs].indices.size();
                const auto rhs_size = cluster_indices[rhs].indices.size();
                if (lhs_size == rhs_size) {
                  return lhs < rhs;
                }
                return lhs_size > rhs_size;
              });

    std::vector<pcl::PointIndices> selected;
    selected.reserve(std::min<std::size_t>(
        order.size(), static_cast<std::size_t>(num_objects)));
    for (std::size_t i = 0; i < order.size() &&
                            static_cast<int>(selected.size()) < num_objects;
         ++i) {
      selected.push_back(cluster_indices[order[i]]);
    }
    return selected;
  }

  std::vector<Eigen::Vector3f> clusterCenters(
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
      const std::vector<pcl::PointIndices> &cluster_indices) const {
    std::vector<Eigen::Vector3f> centers;
    centers.reserve(cluster_indices.size());

    for (const auto &cluster : cluster_indices) {
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud, cluster.indices, centroid);
      centers.emplace_back(centroid.x(), centroid.y(), centroid.z());
    }
    return centers;
  }

  void publishFilteredCloud(
      const std_msgs::msg::Header &header,
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud) const {
    PointCloud2Msg output;
    pcl::toROSMsg(*cloud, output);
    output.header = header;
    filtered_cloud_pub_->publish(output);
  }

  void publishCenters(const std_msgs::msg::Header &header,
                      const std::vector<Eigen::Vector3f> &centers) const {
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header = header;

    for (const auto &center : centers) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = center.x();
      pose.position.y = center.y();
      pose.position.z = center.z();
      pose.orientation.w = 1.0;
      pose_array.poses.push_back(pose);
    }

    centers_pub_->publish(pose_array);
  }

  void publishClusteredCloud(
      const std_msgs::msg::Header &header,
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
      const std::vector<pcl::PointIndices> &cluster_indices) const {
    std::size_t total_points = 0;
    for (const auto &cluster : cluster_indices) {
      total_points += cluster.indices.size();
    }

    sensor_msgs::msg::PointCloud2 output;
    output.header = header;
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2Fields(
        4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
        sensor_msgs::msg::PointField::FLOAT32, "z", 1,
        sensor_msgs::msg::PointField::FLOAT32, "cluster_id", 1,
        sensor_msgs::msg::PointField::INT32);
    modifier.resize(total_points);

    sensor_msgs::PointCloud2Iterator<float> iter_x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(output, "z");
    sensor_msgs::PointCloud2Iterator<std::int32_t> iter_cluster_id(
        output, "cluster_id");

    for (std::size_t cluster_id = 0; cluster_id < cluster_indices.size();
         ++cluster_id) {
      for (const int point_idx : cluster_indices[cluster_id].indices) {
        const auto &point = cloud->points[point_idx];
        *iter_x = point.x;
        *iter_y = point.y;
        *iter_z = point.z;
        *iter_cluster_id = static_cast<std::int32_t>(cluster_id);
        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_cluster_id;
      }
    }

    clustered_cloud_pub_->publish(output);
  }

  void publishMarkers(
      const std_msgs::msg::Header &header,
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
      const std::vector<pcl::PointIndices> &cluster_indices,
      const std::vector<Eigen::Vector3f> &centers) const {
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header = header;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    for (std::size_t i = 0; i < cluster_indices.size(); ++i) {
      Eigen::Vector4f min_point;
      Eigen::Vector4f max_point;
      pcl::getMinMax3D(*cloud, cluster_indices[i].indices, min_point,
                       max_point);

      const auto &center = centers[i];
      const double scale_x =
          std::max(static_cast<double>(max_point.x() - min_point.x()), 0.05);
      const double scale_y =
          std::max(static_cast<double>(max_point.y() - min_point.y()), 0.05);
      const double scale_z =
          std::max(static_cast<double>(max_point.z() - min_point.z()), 0.05);

      visualization_msgs::msg::Marker marker;
      marker.header = header;
      marker.ns = "segmented_lidar_clusters";
      marker.id = static_cast<int>(i);
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = center.x();
      marker.pose.position.y = center.y();
      marker.pose.position.z = center.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = scale_x;
      marker.scale.y = scale_y;
      marker.scale.z = scale_z;
      marker.color.r = 0.1F;
      marker.color.g = 0.8F;
      marker.color.b = 1.0F;
      marker.color.a = 0.35F;
      marker_array.markers.push_back(marker);

      visualization_msgs::msg::Marker text_marker;
      text_marker.header = header;
      text_marker.ns = "segmented_lidar_cluster_labels";
      text_marker.id = static_cast<int>(i);
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose.position.x = center.x();
      text_marker.pose.position.y = center.y();
      text_marker.pose.position.z = center.z() + scale_z * 0.5 + 0.15;
      text_marker.pose.orientation.w = 1.0;
      text_marker.scale.z = 0.18;
      text_marker.color.r = 1.0F;
      text_marker.color.g = 1.0F;
      text_marker.color.b = 1.0F;
      text_marker.color.a = 1.0F;
      text_marker.text =
          std::to_string(i) + ": (" + formatFloat(center.x()) + ", " +
          formatFloat(center.y()) + ", " + formatFloat(center.z()) +
          ") n=" + std::to_string(cluster_indices[i].indices.size());
      marker_array.markers.push_back(text_marker);
    }

    markers_pub_->publish(marker_array);
  }

  Eigen::Isometry3f
  transformToEigen(const geometry_msgs::msg::Transform &transform) const {
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

  std::string formatFloat(float value) const {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f", value);
    return std::string(buffer);
  }

  std::string mask_topic_;
  std::string pointcloud_topic_;
  std::string camera_info_topic_;
  std::string filtered_cloud_topic_;
  std::string centers_topic_;
  std::string clustered_cloud_topic_;
  std::string markers_topic_;
  std::string action_name_;
  std::string camera_frame_param_;
  int mask_threshold_{0};
  double cluster_tolerance_{0.30};
  int min_cluster_size_{8};
  int max_cluster_size_{30000};
  double input_timeout_sec_{1.0};
  double lidar_frame_timeout_sec_{1.0};
  int max_decay_frames_{100};
  int decay_num_{1};
  double tf_timeout_sec_{0.05};

  std::mutex data_mutex_;
  std::condition_variable data_cv_;
  bool has_mask_{false};
  bool collecting_{false};
  bool goal_running_{false};
  int target_decay_frames_{0};
  cv::Mat latest_mask_;
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
  std::vector<PointCloud2Msg::ConstSharedPtr> decay_clouds_;

  rclcpp::Subscription<MaskMsg>::SharedPtr mask_sub_;
  rclcpp::Subscription<PointCloud2Msg>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Publisher<PointCloud2Msg>::SharedPtr filtered_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr centers_pub_;
  rclcpp::Publisher<PointCloud2Msg>::SharedPtr clustered_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      markers_pub_;
  rclcpp_action::Server<LocalSearch>::SharedPtr action_server_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalSearchActionServer>());
  rclcpp::shutdown();
  return 0;
}
