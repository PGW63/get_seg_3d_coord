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
#include <geometry_msgs/msg/point.hpp>
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
#include <vision_msgs/msg/detection2_d_array.hpp>
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
    bbox_topic_ = this->declare_parameter<std::string>(
        "bbox_topic", "/detection/bbox_stream");
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
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
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
    bbox_sub_ = this->create_subscription<DetectionArrayMsg>(
        bbox_topic_, best_effort_qos,
        std::bind(&LocalSearchActionServer::bboxCallback, this,
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
                "cloud=%s, camera_info=%s, bbox=%s",
                action_name_.c_str(), mask_topic_.c_str(),
                pointcloud_topic_.c_str(), camera_info_topic_.c_str(),
                bbox_topic_.c_str());
  }

private:
  using MaskMsg = sensor_msgs::msg::CompressedImage;
  using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
  using DetectionArrayMsg = vision_msgs::msg::Detection2DArray;

  struct BBoxCandidate {
    std::string class_id;
    float score{0.0F};
    float u_min{0.0F};
    float u_max{0.0F};
    float v_min{0.0F};
    float v_max{0.0F};
  };

  struct ClusterResult {
    std::size_t bbox_index{0};
    std::string class_id;
    float score{0.0F};
    Eigen::Vector3f center_lidar{Eigen::Vector3f::Zero()};
    Eigen::Vector3f center_map{Eigen::Vector3f::Zero()};
    pcl::PointIndices indices;
  };

  rclcpp_action::GoalResponse
  handleGoal(const rclcpp_action::GoalUUID &,
             std::shared_ptr<const LocalSearch::Goal> goal) {
    if (goal->class_names.empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "Rejecting goal: class_names is empty.");
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
    auto result = std::make_shared<LocalSearch::Result>();
    auto feedback = std::make_shared<LocalSearch::Feedback>();

    feedback->state = "WAIT_INPUTS";
    feedback->current_num = 0;
    goal_handle->publish_feedback(feedback);

    cv::Mat mask;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
    DetectionArrayMsg::ConstSharedPtr bbox_msg;

    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      const bool ready = data_cv_.wait_for(
          lock, std::chrono::duration<double>(input_timeout_sec_), [this]() {
            return has_mask_ && camera_info_ != nullptr &&
                   latest_bboxes_ != nullptr;
          });

      if (!ready || !has_mask_) {
        abortGoal(goal_handle, result, "no seg image");
        return;
      }
      if (!camera_info_) {
        abortGoal(goal_handle, result, "no camera info");
        return;
      }
      if (!latest_bboxes_) {
        abortGoal(goal_handle, result, "no detections");
        return;
      }

      mask = latest_mask_.clone();
      camera_info = camera_info_;
      bbox_msg = latest_bboxes_;
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

    const auto requested_classes = std::vector<std::string>(
        goal->class_names.begin(), goal->class_names.end());
    const auto candidates =
        buildBBoxCandidates(*bbox_msg, requested_classes, *camera_info, mask);

    auto accumulated_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    std::vector<int> per_point_bbox(0);
    accumulateAndAssign(clouds, camera_info, mask, candidates,
                        accumulated_cloud, per_point_bbox);

    std::vector<ClusterResult> clusters =
        clusterPerBBox(accumulated_cloud, per_point_bbox, candidates);

    const auto output_header = clouds.back()->header;
    const std::string lidar_frame = output_header.frame_id;

    if (!transformCentersToMap(clusters, lidar_frame, output_header.stamp)) {
      abortGoal(goal_handle, result, "tf to map failed");
      return;
    }

    publishFilteredCloud(output_header, accumulated_cloud);
    publishClusteredCloud(output_header, accumulated_cloud, clusters);
    publishCenters(output_header, clusters);
    publishMarkers(output_header, accumulated_cloud, clusters);

    const Eigen::Vector3f reference(static_cast<float>(goal->reference_position.x),
                                    static_cast<float>(goal->reference_position.y),
                                    static_cast<float>(goal->reference_position.z));

    int found_count = 0;
    result->coords.resize(requested_classes.size());
    result->found.resize(requested_classes.size());
    for (std::size_t i = 0; i < requested_classes.size(); ++i) {
      const auto &name = requested_classes[i];
      float best_dist_sq = std::numeric_limits<float>::infinity();
      const ClusterResult *best = nullptr;
      for (const auto &cluster : clusters) {
        if (cluster.class_id != name) {
          continue;
        }
        const float d = (cluster.center_map - reference).squaredNorm();
        if (d < best_dist_sq) {
          best_dist_sq = d;
          best = &cluster;
        }
      }
      if (best) {
        result->coords[i].x = best->center_map.x();
        result->coords[i].y = best->center_map.y();
        result->coords[i].z = best->center_map.z();
        result->found[i] = true;
        ++found_count;
      } else {
        result->coords[i].x = 0.0;
        result->coords[i].y = 0.0;
        result->coords[i].z = 0.0;
        result->found[i] = false;
      }
    }

    feedback->current_num = static_cast<std::int32_t>(found_count);
    goal_handle->publish_feedback(feedback);

    goal_running_ = false;
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "found %d/%zu classes (clusters=%zu, frames=%zu)",
                  found_count, requested_classes.size(), clusters.size(),
                  clouds.size());
    result->message = buf;
    result->success = (found_count == static_cast<int>(requested_classes.size()));
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

  std::vector<BBoxCandidate>
  buildBBoxCandidates(const DetectionArrayMsg &bbox_msg,
                      const std::vector<std::string> &requested_classes,
                      const sensor_msgs::msg::CameraInfo &camera_info,
                      const cv::Mat &mask) const {
    std::vector<BBoxCandidate> candidates;
    candidates.reserve(bbox_msg.detections.size());

    const float img_w = camera_info.width > 0
                            ? static_cast<float>(camera_info.width)
                            : static_cast<float>(mask.cols);
    const float img_h = camera_info.height > 0
                            ? static_cast<float>(camera_info.height)
                            : static_cast<float>(mask.rows);

    for (const auto &det : bbox_msg.detections) {
      if (det.results.empty()) {
        continue;
      }
      const auto &top = det.results.front();
      const std::string class_id = top.hypothesis.class_id;
      if (std::find(requested_classes.begin(), requested_classes.end(),
                    class_id) == requested_classes.end()) {
        continue;
      }

      const float cx = static_cast<float>(det.bbox.center.position.x);
      const float cy = static_cast<float>(det.bbox.center.position.y);
      const float sx = static_cast<float>(det.bbox.size_x);
      const float sy = static_cast<float>(det.bbox.size_y);
      BBoxCandidate cand;
      cand.class_id = class_id;
      cand.score = static_cast<float>(top.hypothesis.score);
      cand.u_min = std::max(0.0F, cx - sx * 0.5F);
      cand.u_max = std::min(img_w - 1.0F, cx + sx * 0.5F);
      cand.v_min = std::max(0.0F, cy - sy * 0.5F);
      cand.v_max = std::min(img_h - 1.0F, cy + sy * 0.5F);
      if (cand.u_max <= cand.u_min || cand.v_max <= cand.v_min) {
        continue;
      }
      candidates.push_back(cand);
    }
    return candidates;
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

  void bboxCallback(const DetectionArrayMsg::ConstSharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_bboxes_ = msg;
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

  void accumulateAndAssign(
      const std::vector<PointCloud2Msg::ConstSharedPtr> &clouds,
      const sensor_msgs::msg::CameraInfo::SharedPtr &camera_info,
      const cv::Mat &mask,
      const std::vector<BBoxCandidate> &candidates,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &accumulated_cloud,
      std::vector<int> &per_point_bbox) const {
    accumulated_cloud->points.clear();
    per_point_bbox.clear();

    for (const auto &cloud_msg : clouds) {
      collectFromOneFrame(cloud_msg, camera_info, mask, candidates,
                          accumulated_cloud, per_point_bbox);
    }

    accumulated_cloud->width =
        static_cast<std::uint32_t>(accumulated_cloud->points.size());
    accumulated_cloud->height = 1;
    accumulated_cloud->is_dense = false;
  }

  void collectFromOneFrame(
      const PointCloud2Msg::ConstSharedPtr &cloud_msg,
      const sensor_msgs::msg::CameraInfo::SharedPtr &camera_info,
      const cv::Mat &mask,
      const std::vector<BBoxCandidate> &candidates,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &accumulated_cloud,
      std::vector<int> &per_point_bbox) const {
    auto lidar_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*cloud_msg, *lidar_cloud);
    if (lidar_cloud->empty()) {
      return;
    }

    const std::string camera_frame =
        camera_frame_param_.empty() ? camera_info->header.frame_id
                                    : camera_frame_param_;
    if (camera_frame.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Camera frame is empty.");
      return;
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
      return;
    }

    const Eigen::Isometry3f lidar_to_camera =
        transformToEigen(transform.transform);

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

    for (const auto &point : lidar_cloud->points) {
      if (!pcl::isFinite(point)) {
        continue;
      }
      const Eigen::Vector3f lidar_point(point.x, point.y, point.z);
      const Eigen::Vector3f camera_point = lidar_to_camera * lidar_point;
      if (camera_point.z() <= 0.0F) {
        continue;
      }

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
        const float area =
            (c.u_max - c.u_min) * (c.v_max - c.v_min);
        if (area < best_area) {
          best_area = area;
          best_bbox = static_cast<int>(b);
        }
      }
      if (best_bbox < 0) {
        continue;
      }

      const int u_mask = static_cast<int>(std::lround(u_orig * x_scale));
      const int v_mask = static_cast<int>(std::lround(v_orig * y_scale));
      if (u_mask < 0 || u_mask >= mask.cols || v_mask < 0 ||
          v_mask >= mask.rows) {
        continue;
      }
      if (mask.at<unsigned char>(v_mask, u_mask) == 0) {
        continue;
      }

      accumulated_cloud->points.push_back(point);
      per_point_bbox.push_back(best_bbox);
    }
  }

  std::vector<ClusterResult> clusterPerBBox(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &accumulated_cloud,
      const std::vector<int> &per_point_bbox,
      const std::vector<BBoxCandidate> &candidates) const {
    std::vector<ClusterResult> results;
    if (candidates.empty() || accumulated_cloud->points.empty()) {
      return results;
    }

    std::vector<std::vector<int>> bins(candidates.size());
    for (std::size_t i = 0; i < per_point_bbox.size(); ++i) {
      bins[per_point_bbox[i]].push_back(static_cast<int>(i));
    }

    for (std::size_t b = 0; b < candidates.size(); ++b) {
      const auto &indices = bins[b];
      if (static_cast<int>(indices.size()) < min_cluster_size_) {
        continue;
      }

      auto sub_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      sub_cloud->points.reserve(indices.size());
      for (int idx : indices) {
        sub_cloud->points.push_back(accumulated_cloud->points[idx]);
      }
      sub_cloud->width = static_cast<std::uint32_t>(sub_cloud->points.size());
      sub_cloud->height = 1;
      sub_cloud->is_dense = false;

      auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
      tree->setInputCloud(sub_cloud);

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> extractor;
      extractor.setClusterTolerance(cluster_tolerance_);
      extractor.setMinClusterSize(min_cluster_size_);
      extractor.setMaxClusterSize(max_cluster_size_);
      extractor.setSearchMethod(tree);
      extractor.setInputCloud(sub_cloud);
      extractor.extract(cluster_indices);

      if (cluster_indices.empty()) {
        continue;
      }

      std::size_t dominant = 0;
      for (std::size_t k = 1; k < cluster_indices.size(); ++k) {
        if (cluster_indices[k].indices.size() >
            cluster_indices[dominant].indices.size()) {
          dominant = k;
        }
      }

      ClusterResult cr;
      cr.bbox_index = b;
      cr.class_id = candidates[b].class_id;
      cr.score = candidates[b].score;
      cr.indices.indices.reserve(cluster_indices[dominant].indices.size());
      for (int sub_idx : cluster_indices[dominant].indices) {
        cr.indices.indices.push_back(indices[sub_idx]);
      }

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*accumulated_cloud, cr.indices.indices, centroid);
      cr.center_lidar = Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z());
      results.push_back(std::move(cr));
    }
    return results;
  }

  bool transformCentersToMap(std::vector<ClusterResult> &clusters,
                             const std::string &lidar_frame,
                             const rclcpp::Time &stamp) const {
    if (clusters.empty()) {
      return true;
    }
    geometry_msgs::msg::TransformStamped tf_msg;
    try {
      tf_msg = tf_buffer_.lookupTransform(
          map_frame_, lidar_frame, stamp,
          rclcpp::Duration::from_seconds(tf_timeout_sec_));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(),
                  "Cannot transform '%s' -> '%s': %s", lidar_frame.c_str(),
                  map_frame_.c_str(), ex.what());
      return false;
    }
    const Eigen::Isometry3f lidar_to_map =
        transformToEigen(tf_msg.transform);
    for (auto &c : clusters) {
      c.center_map = lidar_to_map * c.center_lidar;
    }
    return true;
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
                      const std::vector<ClusterResult> &clusters) const {
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header = header;
    for (const auto &c : clusters) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = c.center_lidar.x();
      pose.position.y = c.center_lidar.y();
      pose.position.z = c.center_lidar.z();
      pose.orientation.w = 1.0;
      pose_array.poses.push_back(pose);
    }
    centers_pub_->publish(pose_array);
  }

  void publishClusteredCloud(
      const std_msgs::msg::Header &header,
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
      const std::vector<ClusterResult> &clusters) const {
    std::size_t total_points = 0;
    for (const auto &c : clusters) {
      total_points += c.indices.indices.size();
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

    for (std::size_t k = 0; k < clusters.size(); ++k) {
      for (const int point_idx : clusters[k].indices.indices) {
        const auto &point = cloud->points[point_idx];
        *iter_x = point.x;
        *iter_y = point.y;
        *iter_z = point.z;
        *iter_cluster_id = static_cast<std::int32_t>(k);
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
      const std::vector<ClusterResult> &clusters) const {
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header = header;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    for (std::size_t i = 0; i < clusters.size(); ++i) {
      const auto &cr = clusters[i];
      Eigen::Vector4f min_point;
      Eigen::Vector4f max_point;
      pcl::getMinMax3D(*cloud, cr.indices.indices, min_point, max_point);

      const double scale_x =
          std::max(static_cast<double>(max_point.x() - min_point.x()), 0.05);
      const double scale_y =
          std::max(static_cast<double>(max_point.y() - min_point.y()), 0.05);
      const double scale_z =
          std::max(static_cast<double>(max_point.z() - min_point.z()), 0.05);

      visualization_msgs::msg::Marker marker;
      marker.header = header;
      marker.ns = "local_search_clusters";
      marker.id = static_cast<int>(i);
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = cr.center_lidar.x();
      marker.pose.position.y = cr.center_lidar.y();
      marker.pose.position.z = cr.center_lidar.z();
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
      text_marker.ns = "local_search_cluster_labels";
      text_marker.id = static_cast<int>(i);
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose.position.x = cr.center_lidar.x();
      text_marker.pose.position.y = cr.center_lidar.y();
      text_marker.pose.position.z = cr.center_lidar.z() + scale_z * 0.5 + 0.15;
      text_marker.pose.orientation.w = 1.0;
      text_marker.scale.z = 0.18;
      text_marker.color.r = 1.0F;
      text_marker.color.g = 1.0F;
      text_marker.color.b = 1.0F;
      text_marker.color.a = 1.0F;
      char score_buf[16];
      std::snprintf(score_buf, sizeof(score_buf), "%.2f", cr.score);
      text_marker.text = cr.class_id + " (" + score_buf + ") n=" +
                         std::to_string(cr.indices.indices.size());
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

  std::string mask_topic_;
  std::string pointcloud_topic_;
  std::string camera_info_topic_;
  std::string bbox_topic_;
  std::string filtered_cloud_topic_;
  std::string centers_topic_;
  std::string clustered_cloud_topic_;
  std::string markers_topic_;
  std::string action_name_;
  std::string camera_frame_param_;
  std::string map_frame_;
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
  DetectionArrayMsg::ConstSharedPtr latest_bboxes_;
  std::vector<PointCloud2Msg::ConstSharedPtr> decay_clouds_;

  rclcpp::Subscription<MaskMsg>::SharedPtr mask_sub_;
  rclcpp::Subscription<PointCloud2Msg>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Subscription<DetectionArrayMsg>::SharedPtr bbox_sub_;
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
