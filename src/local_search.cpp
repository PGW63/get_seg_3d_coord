#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.h>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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

#include "get_seg_3d_coord/cluster_extractor.hpp"
#include "get_seg_3d_coord/local_search_types.hpp"
#include "get_seg_3d_coord/point_selector.hpp"
#include "get_seg_3d_coord/position_ekf.hpp"
#include "get_seg_3d_coord/state_odometry.hpp"
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
    pointcloud_topic_ = this->declare_parameter<std::string>(
        "pointcloud_topic", "/approach/submap_cloud");
    camera_info_topic_ = this->declare_parameter<std::string>(
        "camera_info_topic", "/camera/camera_head/color/camera_info");
    bbox_topic_ = this->declare_parameter<std::string>(
        "bbox_topic", "/detection/bbox_stream");
    odom_topic_ =
        this->declare_parameter<std::string>("odom_topic", "/odom");
    filtered_cloud_topic_ = this->declare_parameter<std::string>(
        "filtered_cloud_topic", "~/points");
    centers_topic_ = this->declare_parameter<std::string>(
        "centers_topic", "~/cluster_centers");
    clustered_cloud_topic_ = this->declare_parameter<std::string>(
        "clustered_cloud_topic", "~/clustered_points");
    markers_topic_ = this->declare_parameter<std::string>(
        "markers_topic", "~/cluster_markers");
    diagnostics_topic_ = this->declare_parameter<std::string>(
        "diagnostics_topic", "~/frame_diagnostics");
    action_name_ =
        this->declare_parameter<std::string>("action_name", "local_search");
    camera_frame_param_ =
        this->declare_parameter<std::string>("camera_frame", "camera_head_color_optical_frame");
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    mask_threshold_ = this->declare_parameter<int>("mask_threshold", 0);
    cluster_tolerance_ =
        this->declare_parameter<double>("cluster_tolerance", 0.3);
    min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 3);
    max_cluster_size_ =
        this->declare_parameter<int>("max_cluster_size", 30000);
    input_sync_queue_size_ =
        this->declare_parameter<int>("input_sync_queue_size", 10);
    input_sync_slop_sec_ =
        this->declare_parameter<double>("input_sync_slop_sec", 0.2);
    tf_timeout_sec_ = this->declare_parameter<double>("tf_timeout_sec", 0.5);
    require_stable_odom_ =
        this->declare_parameter<bool>("require_stable_odom", false);
    odom_stable_window_sec_ =
        this->declare_parameter<double>("odom_stable_window_sec", 0.3);
    odom_stable_timeout_sec_ =
        this->declare_parameter<double>("odom_stable_timeout_sec", 1.0);
    max_relative_angular_velocity_rad_s_ =
        this->declare_parameter<double>("max_relative_angular_velocity_rad_s",
                                        0.20);
    odom_buffer_size_ =
        this->declare_parameter<int>("odom_buffer_size", 200);
    prefilter_cloud_ =
        this->declare_parameter<bool>("prefilter_cloud", false);
    prefilter_frame_ =
        this->declare_parameter<std::string>("prefilter_frame", "base");
    prefilter_x_min_ =
        this->declare_parameter<double>("prefilter_x_min", 0.0);
    prefilter_x_max_ =
        this->declare_parameter<double>("prefilter_x_max", 5.0);
    prefilter_y_min_ =
        this->declare_parameter<double>("prefilter_y_min", -2.0);
    prefilter_y_max_ =
        this->declare_parameter<double>("prefilter_y_max", 2.0);
    prefilter_z_min_ =
        this->declare_parameter<double>("prefilter_z_min", -0.05);
    prefilter_z_max_ =
        this->declare_parameter<double>("prefilter_z_max", 2.0);
    submap_timeout_sec_ =
        this->declare_parameter<double>("submap_timeout_sec", 3.0);
    max_distance_from_reference_m_ = this->declare_parameter<double>(
        "max_distance_from_reference_m", 1.5);
    ekf_enabled_ = this->declare_parameter<bool>("ekf_enabled", true);
    ekf_timeout_sec_ =
        this->declare_parameter<double>("ekf_timeout_sec", 10.0);
    ekf_converged_variance_ =
        this->declare_parameter<double>("ekf_converged_variance", 0.01);
    ekf_initial_variance_ =
        this->declare_parameter<double>("ekf_initial_variance", 1.0);
    ekf_measurement_variance_ =
        this->declare_parameter<double>("ekf_measurement_variance", 0.09);
    ekf_process_variance_per_sec_ = this->declare_parameter<double>(
        "ekf_process_variance_per_sec", 0.0025);
    ekf_log_enabled_ =
        this->declare_parameter<bool>("ekf_log_enabled", true);
    ekf_log_dir_ =
        this->declare_parameter<std::string>("ekf_log_dir", "EKF_results");
    max_distance_from_reference_m_ =
        std::max(0.0, max_distance_from_reference_m_);
    ekf_timeout_sec_ = std::max(0.1, ekf_timeout_sec_);
    ekf_converged_variance_ = std::max(1e-6, ekf_converged_variance_);
    ekf_initial_variance_ = std::max(1e-6, ekf_initial_variance_);
    ekf_measurement_variance_ = std::max(1e-6, ekf_measurement_variance_);
    ekf_process_variance_per_sec_ =
        std::max(0.0, ekf_process_variance_per_sec_);
    submap_timeout_sec_ = std::max(0.1, submap_timeout_sec_);
    input_sync_queue_size_ = std::max(1, input_sync_queue_size_);
    input_sync_slop_sec_ = std::max(0.0, input_sync_slop_sec_);
    odom_stable_window_sec_ = std::max(0.01, odom_stable_window_sec_);
    odom_stable_timeout_sec_ = std::max(0.01, odom_stable_timeout_sec_);
    max_relative_angular_velocity_rad_s_ =
        std::max(0.0, max_relative_angular_velocity_rad_s_);
    odom_buffer_size_ = std::max(2, odom_buffer_size_);
    state_odometry_.setMaxBufferSize(
        static_cast<std::size_t>(odom_buffer_size_));
    if (prefilter_x_min_ > prefilter_x_max_) {
      std::swap(prefilter_x_min_, prefilter_x_max_);
    }
    if (prefilter_y_min_ > prefilter_y_max_) {
      std::swap(prefilter_y_min_, prefilter_y_max_);
    }
    if (prefilter_z_min_ > prefilter_z_max_) {
      std::swap(prefilter_z_min_, prefilter_z_max_);
    }

    auto best_effort_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    mask_sub_.subscribe(this, mask_topic_, best_effort_qos.get_rmw_qos_profile());
    bbox_sub_.subscribe(this, bbox_topic_, reliable_qos.get_rmw_qos_profile());
    input_sync_ =
        std::make_shared<InputSynchronizer>(InputSyncPolicy(input_sync_queue_size_),
                                            mask_sub_, bbox_sub_);
    input_sync_->setMaxIntervalDuration(
        rclcpp::Duration::from_seconds(input_sync_slop_sec_));
    input_sync_->registerCallback(
        std::bind(&LocalSearchActionServer::syncedInputCallback, this,
                  std::placeholders::_1, std::placeholders::_2));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, reliable_qos,
        std::bind(&LocalSearchActionServer::cameraInfoCallback, this,
                  std::placeholders::_1));
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, best_effort_qos,
        std::bind(&LocalSearchActionServer::cloudCallback, this,
                  std::placeholders::_1));
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, reliable_qos,
        std::bind(&LocalSearchActionServer::odomCallback, this,
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
    diagnostics_pub_ =
        this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            diagnostics_topic_, reliable_qos);

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
                "cloud=%s, camera_info=%s, bbox=%s, odom=%s, "
                "submap_timeout=%.2fs, require_stable_odom=%s",
                action_name_.c_str(), mask_topic_.c_str(),
                pointcloud_topic_.c_str(), camera_info_topic_.c_str(),
                bbox_topic_.c_str(), odom_topic_.c_str(), submap_timeout_sec_,
                require_stable_odom_ ? "true" : "false");
  }

private:
  using MaskMsg = sensor_msgs::msg::CompressedImage;
  using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
  using DetectionArrayMsg = vision_msgs::msg::Detection2DArray;
  using BBoxCandidate = get_seg_3d_coord::BBoxCandidate;
  using ClusterResult = get_seg_3d_coord::ClusterResult;
  using PointSelectionConfig = get_seg_3d_coord::PointSelectionConfig;
  using PointSelectionStats = get_seg_3d_coord::PointSelectionStats;
  using PositionEkf = get_seg_3d_coord::PositionEkf;
  using StateOdometry = get_seg_3d_coord::StateOdometry;
  using SyncedInput = get_seg_3d_coord::SyncedInput;
  using InputSyncPolicy =
      message_filters::sync_policies::ApproximateTime<MaskMsg,
                                                       DetectionArrayMsg>;
  using InputSynchronizer = message_filters::Synchronizer<InputSyncPolicy>;

  struct MeasurementBatch {
    builtin_interfaces::msg::Time stamp;
    std::vector<Eigen::Vector3f> positions;
    std::vector<bool> found;
    PointSelectionStats selection_stats;
    std::size_t bbox_candidate_count{0};
    std::size_t cluster_count{0};
    std::size_t point_count{0};
    std::size_t matched_class_count{0};
    std::string status{"unknown"};
    std::string reason;
    std::string diagnostics;
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
    if (goal_running_) {
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

    const auto requested_classes = std::vector<std::string>(
        goal->class_names.begin(), goal->class_names.end());
    const rclcpp::Time goal_start_stamp = this->now();
    const Eigen::Vector3f reference(static_cast<float>(goal->reference_position.x),
                                    static_cast<float>(goal->reference_position.y),
                                    static_cast<float>(goal->reference_position.z));
    const double effective_max_distance =
        goal->max_distance > 0.0F
            ? static_cast<double>(goal->max_distance)
            : max_distance_from_reference_m_;

    std::vector<PositionEkf> filters(requested_classes.size());
    std::vector<bool> ever_found(requested_classes.size(), false);
    std::vector<rclcpp::Time> last_update(requested_classes.size(),
                                          goal_start_stamp);
    std::ofstream ekf_log = openEkfLog(goal_start_stamp, requested_classes);
    std::size_t total_measurements = 0;
    std::size_t last_cluster_count = 0;
    std::size_t last_point_count = 0;
    std::string last_diag;
    bool ekf_converged = false;
    rclcpp::Time min_input_stamp = goal_start_stamp;
    const auto ekf_deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(ekf_enabled_ ? ekf_timeout_sec_
                                                       : submap_timeout_sec_));

    while (!goal_handle->is_canceling()) {
      feedback->state = ekf_enabled_ ? "EKF_TRACK" : "WAIT_INPUTS";
      feedback->current_num =
          static_cast<std::int32_t>(std::count(ever_found.begin(),
                                               ever_found.end(), true));
      goal_handle->publish_feedback(feedback);

      MeasurementBatch batch;
      const bool got_measurement = collectMeasurementBatch(
          goal_handle, requested_classes, min_input_stamp, reference,
          effective_max_distance, ekf_deadline, &batch, &min_input_stamp);
      publishFrameDiagnostic(batch, requested_classes);
      if (!got_measurement) {
        break;
      }

      last_cluster_count = batch.cluster_count;
      last_point_count = batch.point_count;
      last_diag = batch.diagnostics;
      bool any_update = false;
      const rclcpp::Time update_stamp = this->now();
      for (std::size_t i = 0; i < requested_classes.size(); ++i) {
        if (!batch.found[i]) {
          continue;
        }
        if (filters[i].initialized()) {
          const double dt =
              std::max(0.0, (update_stamp - last_update[i]).seconds());
          filters[i].predict(dt, ekf_process_variance_per_sec_);
          const double nis = filters[i].normalizedInnovationSquared(
              batch.positions[i], ekf_measurement_variance_);
          writeEkfLog(ekf_log, update_stamp, requested_classes[i],
                      batch.positions[i], filters[i], nis);
          filters[i].update(batch.positions[i], ekf_measurement_variance_);
        } else {
          filters[i].initialize(batch.positions[i], ekf_initial_variance_);
          writeEkfLog(ekf_log, update_stamp, requested_classes[i],
                      batch.positions[i], filters[i], -1.0);
        }
        last_update[i] = update_stamp;
        ever_found[i] = true;
        any_update = true;
        ++total_measurements;
      }

      if (!ekf_enabled_) {
        break;
      }

      bool all_converged = true;
      for (std::size_t i = 0; i < requested_classes.size(); ++i) {
        if (!filters[i].initialized() ||
            filters[i].maxVariance() > ekf_converged_variance_) {
          all_converged = false;
          break;
        }
      }
      if (all_converged && any_update) {
        ekf_converged = true;
        break;
      }
      if (std::chrono::steady_clock::now() >= ekf_deadline) {
        break;
      }
    }

    if (goal_handle->is_canceling()) {
      goal_running_ = false;
      result->success = false;
      result->message = "canceled";
      goal_handle->canceled(result);
      return;
    }

    int found_count = 0;
    double max_variance = 0.0;
    result->coords.resize(requested_classes.size());
    result->found.resize(requested_classes.size());
    for (std::size_t i = 0; i < requested_classes.size(); ++i) {
      if (filters[i].initialized()) {
        const auto &position = filters[i].position();
        max_variance = std::max(max_variance, filters[i].maxVariance());
        result->coords[i].x = position.x();
        result->coords[i].y = position.y();
        result->coords[i].z = position.z();
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
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "found %d/%zu classes (measurements=%zu, clusters=%zu, "
                  "points=%zu, ekf=%s, converged=%s, "
                  "max_var=%.4f) %s",
                  found_count, requested_classes.size(), total_measurements,
                  last_cluster_count, last_point_count, ekf_enabled_ ? "on" : "off",
                  ekf_converged ? "true" : "false", max_variance,
                  last_diag.c_str());
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

  std::ofstream openEkfLog(
      const rclcpp::Time &run_stamp,
      const std::vector<std::string> &requested_classes) const {
    std::ofstream log;
    if (!ekf_log_enabled_) {
      return log;
    }

    try {
      std::filesystem::create_directories(ekf_log_dir_);
      std::ostringstream path;
      path << ekf_log_dir_ << "/ekf_"
           << run_stamp.seconds() << ".csv";
      log.open(path.str(), std::ios::out);
    } catch (const std::exception &ex) {
      RCLCPP_WARN(this->get_logger(), "Failed to open EKF log: %s", ex.what());
      return {};
    }

    if (!log.is_open()) {
      RCLCPP_WARN(this->get_logger(), "Failed to open EKF log file.");
      return {};
    }

    log << std::setprecision(9);
    log << "# classes=";
    for (std::size_t i = 0; i < requested_classes.size(); ++i) {
      if (i > 0) log << "|";
      log << requested_classes[i];
    }
    log << "\n";
    log << "# ekf_initial_variance=" << ekf_initial_variance_ << "\n";
    log << "# ekf_measurement_variance=" << ekf_measurement_variance_
        << "\n";
    log << "# ekf_process_variance_per_sec="
        << ekf_process_variance_per_sec_ << "\n";
    log << "# ekf_converged_variance=" << ekf_converged_variance_ << "\n";
    log << "stamp,class,meas_x,meas_y,meas_z,est_x,est_y,est_z,"
           "cov_x,cov_y,cov_z,nis\n";
    return log;
  }

  void writeEkfLog(std::ofstream &log,
                   const rclcpp::Time &stamp,
                   const std::string &class_id,
                   const Eigen::Vector3f &measurement,
                   const PositionEkf &filter,
                   double nis) const {
    if (!log.is_open()) {
      return;
    }
    const auto &position = filter.position();
    const auto covariance = filter.covariance().diagonal();
    log << stamp.seconds() << "," << class_id << ","
        << measurement.x() << "," << measurement.y() << ","
        << measurement.z() << "," << position.x() << ","
        << position.y() << "," << position.z() << ","
        << covariance.x() << "," << covariance.y() << ","
        << covariance.z() << "," << nis << "\n";
  }

  bool collectMeasurementBatch(
      const std::shared_ptr<GoalHandle> &goal_handle,
      const std::vector<std::string> &requested_classes,
      const rclcpp::Time &min_input_stamp,
      const Eigen::Vector3f &reference,
      double effective_max_distance,
      const std::chrono::steady_clock::time_point &deadline,
      MeasurementBatch *batch,
      rclcpp::Time *next_min_input_stamp) {
    cv::Mat mask;
    std_msgs::msg::Header mask_header;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
    DetectionArrayMsg::ConstSharedPtr bbox_msg;

    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      const bool has_input = data_cv_.wait_until(
          lock, deadline,
          [this, &goal_handle, &min_input_stamp, &requested_classes]() {
            if (goal_handle->is_canceling()) return true;
            if (!has_synced_input_ || camera_info_ == nullptr) return false;
            const auto &bm = latest_synced_input_.bbox_msg;
            if (!bm) return false;
            const rclcpp::Time bbox_stamp(bm->header.stamp);
            if (bbox_stamp <= min_input_stamp) return false;
            for (const auto &det : bm->detections) {
              if (det.results.empty()) continue;
              const auto &cid = det.results.front().hypothesis.class_id;
              if (std::find(requested_classes.begin(),
                            requested_classes.end(),
                            cid) != requested_classes.end()) {
                return true;
              }
            }
            return false;
          });
      if (!has_input || goal_handle->is_canceling()) {
        batch->stamp = this->now().to_msg();
        if (goal_handle->is_canceling()) {
          batch->status = "canceled";
          batch->reason = "action canceled while waiting for input";
        } else if (!has_synced_input_) {
          batch->status = "input_timeout";
          batch->reason = "no synchronized mask and bbox were received";
        } else if (camera_info_ == nullptr) {
          batch->stamp = latest_synced_input_.mask_header.stamp;
          batch->status = "input_timeout";
          batch->reason = "camera info was not received";
        } else if (!latest_synced_input_.bbox_msg) {
          batch->stamp = latest_synced_input_.mask_header.stamp;
          batch->status = "input_timeout";
          batch->reason = "synchronized bbox message was empty";
        } else {
          batch->stamp = latest_synced_input_.mask_header.stamp;
          const auto &latest_bbox = latest_synced_input_.bbox_msg;
          bool has_requested_class = false;
          for (const auto &det : latest_bbox->detections) {
            if (det.results.empty()) continue;
            const auto &class_id = det.results.front().hypothesis.class_id;
            if (std::find(requested_classes.begin(), requested_classes.end(),
                          class_id) != requested_classes.end()) {
              has_requested_class = true;
              break;
            }
          }
          if (!has_requested_class) {
            batch->status = "no_requested_bbox";
            batch->reason = "no bbox for any requested class was received";
          } else {
            batch->status = "input_timeout";
            batch->reason = "no newer synchronized mask and bbox arrived";
          }
        }
        return false;
      }

      mask = latest_synced_input_.mask.clone();
      mask_header = latest_synced_input_.mask_header;
      camera_info = camera_info_;
      bbox_msg = latest_synced_input_.bbox_msg;
    }

    const rclcpp::Time projection_stamp(mask_header.stamp);
    batch->stamp = mask_header.stamp;
    *next_min_input_stamp = rclcpp::Time(bbox_msg->header.stamp);
    batch->positions.assign(requested_classes.size(), Eigen::Vector3f::Zero());
    batch->found.assign(requested_classes.size(), false);
    std::string odom_failure_reason;
    if (require_stable_odom_ &&
        !waitForStableOdom(goal_handle, projection_stamp,
                           &odom_failure_reason)) {
      batch->diagnostics = "[unstable odom]";
      batch->status = "unstable_odom";
      batch->reason = "odometry was not stable around the mask timestamp";
      if (!odom_failure_reason.empty()) {
        batch->reason += ": " + odom_failure_reason;
      }
      return true;
    }

    const rclcpp::Time collection_start_stamp = this->now();
    PointCloud2Msg::ConstSharedPtr submap_cloud;
    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      auto cloud_deadline =
          std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::duration<double>(submap_timeout_sec_));
      if (deadline < cloud_deadline) {
        cloud_deadline = deadline;
      }
      while (!submap_cloud && !goal_handle->is_canceling()) {
        if (!cloud_queue_.empty()) {
          auto cm = cloud_queue_.front();
          cloud_queue_.pop_front();
          const rclcpp::Time cstamp(cm->header.stamp);
          if (cstamp >= collection_start_stamp) {
            submap_cloud = cm;
          }
          continue;
        }
        if (data_cv_.wait_until(lock, cloud_deadline) ==
            std::cv_status::timeout) {
          break;
        }
      }
    }
    if (goal_handle->is_canceling() || !submap_cloud) {
      batch->status = goal_handle->is_canceling() ? "canceled"
                                                  : "cloud_timeout";
      batch->reason = goal_handle->is_canceling()
                          ? "action canceled while waiting for point cloud"
                          : "no point cloud newer than collection start arrived "
                            "before timeout";
      return false;
    }

    const auto candidates =
        buildBBoxCandidates(*bbox_msg, requested_classes, *camera_info, mask);
    batch->bbox_candidate_count = candidates.size();
    auto accumulated_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    std::vector<int> per_point_bbox;
    const PointSelectionConfig selection_config = makePointSelectionConfig();
    std::string selection_failure_reason;
    const bool selection_ok = get_seg_3d_coord::collectMaskProjectedPoints(
        submap_cloud, camera_info, mask, candidates, projection_stamp,
        selection_config, tf_buffer_, this->get_logger(), *this->get_clock(),
        accumulated_cloud, per_point_bbox, &batch->selection_stats,
        &selection_failure_reason);
    accumulated_cloud->width =
        static_cast<std::uint32_t>(accumulated_cloud->points.size());
    accumulated_cloud->height = 1;
    accumulated_cloud->is_dense = false;

    std::vector<ClusterResult> clusters =
        get_seg_3d_coord::clusterPerBBox(
            accumulated_cloud, per_point_bbox, candidates, cluster_tolerance_,
            min_cluster_size_, max_cluster_size_, this->get_logger());

    std_msgs::msg::Header output_header;
    output_header.stamp = projection_stamp;
    output_header.frame_id = map_frame_;

    for (auto &c : clusters) {
      c.center_map = c.center_lidar;
    }

    publishFilteredCloud(output_header, accumulated_cloud);
    publishClusteredCloud(output_header, accumulated_cloud, clusters);
    publishCenters(output_header, clusters);
    publishMarkers(output_header, accumulated_cloud, clusters);

    batch->cluster_count = clusters.size();
    batch->point_count = accumulated_cloud->points.size();

    const float max_dist_sq =
        static_cast<float>(effective_max_distance * effective_max_distance);
    for (std::size_t i = 0; i < requested_classes.size(); ++i) {
      const auto &name = requested_classes[i];
      float best_dist_sq = std::numeric_limits<float>::infinity();
      float nearest_rejected_dist = std::numeric_limits<float>::infinity();
      int matching_class_clusters = 0;
      const ClusterResult *best = nullptr;
      for (const auto &cluster : clusters) {
        if (cluster.class_id != name) {
          continue;
        }
        ++matching_class_clusters;
        const float d = (cluster.center_map - reference).squaredNorm();
        if (d > max_dist_sq) {
          if (d < nearest_rejected_dist) nearest_rejected_dist = d;
          continue;
        }
        if (d < best_dist_sq) {
          best_dist_sq = d;
          best = &cluster;
        }
      }

      if (best) {
        batch->positions[i] = best->center_map;
        batch->found[i] = true;
        ++batch->matched_class_count;
      } else {
        char dbuf[128];
        if (matching_class_clusters == 0) {
          std::snprintf(dbuf, sizeof(dbuf), "[%s: no cluster]", name.c_str());
        } else {
          std::snprintf(dbuf, sizeof(dbuf),
                        "[%s: %d cluster(s) but nearest %.2fm > %.2fm gate]",
                        name.c_str(), matching_class_clusters,
                        std::sqrt(nearest_rejected_dist),
                        effective_max_distance);
        }
        batch->diagnostics += dbuf;
      }
    }

    if (!selection_ok) {
      batch->status = "projection_failed";
      batch->reason = selection_failure_reason.empty()
                          ? "point projection failed"
                          : selection_failure_reason;
    } else if (candidates.empty()) {
      batch->status = "no_bbox_candidate";
      batch->reason = "no valid bbox matched the requested classes";
    } else if (batch->selection_stats.selected_points == 0) {
      batch->status = "no_matching_lidar";
      if (batch->selection_stats.raw_points == 0) {
        batch->reason = "input point cloud was empty";
      } else if (batch->selection_stats.finite_points == 0) {
        batch->reason = "input point cloud had no finite points";
      } else if (batch->selection_stats.prefiltered_points == 0) {
        batch->reason = "no LiDAR point passed the spatial prefilter";
      } else if (batch->selection_stats.camera_front_points == 0) {
        batch->reason = "no prefiltered LiDAR point was in front of the camera";
      } else if (batch->selection_stats.bbox_points == 0) {
        batch->reason = "no projected LiDAR point landed inside a target bbox";
      } else {
        batch->reason = "bbox LiDAR points did not overlap the segmentation mask";
      }
    } else if (clusters.empty()) {
      batch->status = "no_cluster";
      batch->reason = "matched LiDAR points did not form a valid cluster";
    } else if (batch->matched_class_count == requested_classes.size()) {
      batch->status = "ok";
      batch->reason = "all requested classes matched";
    } else if (batch->matched_class_count > 0) {
      batch->status = "partial_match";
      batch->reason = batch->diagnostics;
    } else {
      batch->status = "no_target_cluster";
      batch->reason = batch->diagnostics.empty()
                          ? "no cluster matched the requested classes"
                          : batch->diagnostics;
    }

    return true;
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

  void syncedInputCallback(const MaskMsg::ConstSharedPtr mask_msg,
                           const DetectionArrayMsg::ConstSharedPtr bbox_msg) {
    cv::Mat mask = imageToBinaryMask(mask_msg);
    if (mask.empty()) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_synced_input_.mask = mask;
      latest_synced_input_.mask_header = mask_msg->header;
      latest_synced_input_.bbox_msg = bbox_msg;
      has_synced_input_ = true;
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

  void cloudCallback(const PointCloud2Msg::ConstSharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      cloud_queue_.push_back(msg);
      while (cloud_queue_.size() >
             static_cast<std::size_t>(cloud_queue_max_size_)) {
        cloud_queue_.pop_front();
      }
    }
    data_cv_.notify_all();
  }

  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      state_odometry_.push(msg);
    }
    data_cv_.notify_all();
  }

  bool waitForStableOdom(const std::shared_ptr<GoalHandle> &goal_handle,
                         const rclcpp::Time &stamp,
                         std::string *failure_reason) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(odom_stable_timeout_sec_));
    while (!goal_handle->is_canceling()) {
      std::string reason;
      if (state_odometry_.isStableAround(
              stamp, odom_stable_window_sec_,
              max_relative_angular_velocity_rad_s_, &reason)) {
        return true;
      }
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Waiting stable odom near mask stamp: %s",
                            reason.c_str());
      if (data_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        if (failure_reason) {
          *failure_reason = reason;
        }
        RCLCPP_WARN(this->get_logger(),
                    "Odom was not stable around mask stamp within %.2fs: %s",
                    odom_stable_timeout_sec_, reason.c_str());
        return false;
      }
    }
    return false;
  }

  PointSelectionConfig makePointSelectionConfig() const {
    PointSelectionConfig config;
    config.camera_frame = camera_frame_param_;
    config.output_frame = map_frame_;
    config.tf_timeout_sec = tf_timeout_sec_;
    config.prefilter_cloud = prefilter_cloud_;
    config.prefilter_frame = prefilter_frame_;
    config.prefilter_x_min = prefilter_x_min_;
    config.prefilter_x_max = prefilter_x_max_;
    config.prefilter_y_min = prefilter_y_min_;
    config.prefilter_y_max = prefilter_y_max_;
    config.prefilter_z_min = prefilter_z_min_;
    config.prefilter_z_max = prefilter_z_max_;
    return config;
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

  void publishFrameDiagnostic(
      const MeasurementBatch &batch,
      const std::vector<std::string> &requested_classes) const {
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = batch.stamp;
    array.header.frame_id = map_frame_;

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "local_search/frame";
    status.hardware_id = "local_search";
    status.message = batch.status;
    if (!batch.reason.empty()) {
      status.message += ": " + batch.reason;
    }
    if (batch.status == "ok") {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    } else if (batch.status == "projection_failed") {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    } else {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    }

    const auto add_value = [&status](const std::string &key,
                                     const std::string &value) {
      diagnostic_msgs::msg::KeyValue item;
      item.key = key;
      item.value = value;
      status.values.push_back(std::move(item));
    };
    add_value("status", batch.status);
    add_value("reason", batch.reason);
    std::ostringstream class_names;
    for (std::size_t i = 0; i < requested_classes.size(); ++i) {
      if (i > 0) class_names << "|";
      class_names << requested_classes[i];
    }
    add_value("requested_classes", std::to_string(requested_classes.size()));
    add_value("requested_class_names", class_names.str());
    add_value("matched_classes", std::to_string(batch.matched_class_count));
    add_value("bbox_candidates", std::to_string(batch.bbox_candidate_count));
    add_value("raw_points",
              std::to_string(batch.selection_stats.raw_points));
    add_value("finite_points",
              std::to_string(batch.selection_stats.finite_points));
    add_value("prefiltered_points",
              std::to_string(batch.selection_stats.prefiltered_points));
    add_value("camera_front_points",
              std::to_string(batch.selection_stats.camera_front_points));
    add_value("bbox_points",
              std::to_string(batch.selection_stats.bbox_points));
    add_value("matched_points",
              std::to_string(batch.selection_stats.selected_points));
    add_value("clusters", std::to_string(batch.cluster_count));
    add_value("class_diagnostics", batch.diagnostics);
    array.status.push_back(std::move(status));
    diagnostics_pub_->publish(array);
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

  std::string mask_topic_;
  std::string pointcloud_topic_;
  std::string camera_info_topic_;
  std::string bbox_topic_;
  std::string odom_topic_;
  std::string filtered_cloud_topic_;
  std::string centers_topic_;
  std::string clustered_cloud_topic_;
  std::string markers_topic_;
  std::string diagnostics_topic_;
  std::string action_name_;
  std::string camera_frame_param_;
  std::string map_frame_;
  int mask_threshold_{0};
  double cluster_tolerance_{0.30};
  int min_cluster_size_{8};
  int max_cluster_size_{30000};
  int input_sync_queue_size_{10};
  double input_sync_slop_sec_{0.2};
  double tf_timeout_sec_{0.05};
  bool require_stable_odom_{false};
  double odom_stable_window_sec_{0.3};
  double odom_stable_timeout_sec_{1.0};
  double max_relative_angular_velocity_rad_s_{0.20};
  int odom_buffer_size_{200};
  bool prefilter_cloud_{false};
  std::string prefilter_frame_{"base"};
  double prefilter_x_min_{0.0};
  double prefilter_x_max_{5.0};
  double prefilter_y_min_{-2.0};
  double prefilter_y_max_{2.0};
  double prefilter_z_min_{-0.05};
  double prefilter_z_max_{2.0};
  double submap_timeout_sec_{3.0};
  int cloud_queue_max_size_{60};
  double max_distance_from_reference_m_{1.0};
  bool ekf_enabled_{true};
  double ekf_timeout_sec_{10.0};
  double ekf_converged_variance_{0.01};
  double ekf_initial_variance_{1.0};
  double ekf_measurement_variance_{0.09};
  double ekf_process_variance_per_sec_{0.0025};
  bool ekf_log_enabled_{true};
  std::string ekf_log_dir_{"EKF_results"};

  std::mutex data_mutex_;
  std::condition_variable data_cv_;
  bool has_synced_input_{false};
  bool goal_running_{false};
  SyncedInput latest_synced_input_;
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
  std::deque<PointCloud2Msg::ConstSharedPtr> cloud_queue_;
  StateOdometry state_odometry_;

  message_filters::Subscriber<MaskMsg> mask_sub_;
  message_filters::Subscriber<DetectionArrayMsg> bbox_sub_;
  std::shared_ptr<InputSynchronizer> input_sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<PointCloud2Msg>::SharedPtr filtered_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr centers_pub_;
  rclcpp::Publisher<PointCloud2Msg>::SharedPtr clustered_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      markers_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_pub_;
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
