#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/common/point_tests.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace {

using DiagnosticArray = diagnostic_msgs::msg::DiagnosticArray;
using CompressedImage = sensor_msgs::msg::CompressedImage;
using PointCloud2 = sensor_msgs::msg::PointCloud2;

std::int64_t stampNanoseconds(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

std::string stampFileStem(const builtin_interfaces::msg::Time &stamp) {
  std::ostringstream out;
  out << stamp.sec << "_" << std::setw(9) << std::setfill('0')
      << stamp.nanosec;
  return out.str();
}

std::string csvEscape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const char ch : value) {
    if (ch == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

Eigen::Isometry3f transformToEigen(
    const geometry_msgs::msg::Transform &transform) {
  Eigen::Isometry3f result = Eigen::Isometry3f::Identity();
  result.translation() =
      Eigen::Vector3f(transform.translation.x, transform.translation.y,
                      transform.translation.z);
  const Eigen::Quaternionf rotation(
      transform.rotation.w, transform.rotation.x, transform.rotation.y,
      transform.rotation.z);
  result.linear() = rotation.normalized().toRotationMatrix();
  return result;
}

}  // namespace

class LocalSearchLogger : public rclcpp::Node {
public:
  LocalSearchLogger()
      : Node("local_search_logger"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    mask_topic_ = this->declare_parameter<std::string>(
        "mask_topic", "/sam2_segmentation_node/binary_mask/compressed");
    matched_cloud_topic_ = this->declare_parameter<std::string>(
        "matched_cloud_topic", "/local_search/points");
    camera_info_topic_ = this->declare_parameter<std::string>(
        "camera_info_topic", "/camera/camera_head/color/camera_info");
    diagnostics_topic_ = this->declare_parameter<std::string>(
        "diagnostics_topic", "/local_search/frame_diagnostics");
    camera_frame_ = this->declare_parameter<std::string>(
        "camera_frame", "camera_head_color_optical_frame");
    output_dir_ = this->declare_parameter<std::string>(
        "output_dir", "/home/nvidia/inha_log/module/local_search");
    artifact_wait_sec_ =
        std::max(0.0, this->declare_parameter<double>("artifact_wait_sec", 0.5));
    tf_timeout_sec_ =
        std::max(0.0, this->declare_parameter<double>("tf_timeout_sec", 0.2));
    cache_size_ = static_cast<int>(std::max<std::int64_t>(
        10, this->declare_parameter<std::int64_t>("cache_size", 100)));
    point_radius_px_ = static_cast<int>(std::max<std::int64_t>(
        1, this->declare_parameter<std::int64_t>("point_radius_px", 2)));
    save_matched_cloud_ =
        this->declare_parameter<bool>("save_matched_cloud", true);

    initializeOutput();

    const auto best_effort_qos =
        rclcpp::QoS(rclcpp::KeepLast(30)).best_effort();
    const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(30)).reliable();
    mask_sub_ = this->create_subscription<CompressedImage>(
        mask_topic_, best_effort_qos,
        std::bind(&LocalSearchLogger::maskCallback, this,
                  std::placeholders::_1));
    cloud_sub_ = this->create_subscription<PointCloud2>(
        matched_cloud_topic_, best_effort_qos,
        std::bind(&LocalSearchLogger::cloudCallback, this,
                  std::placeholders::_1));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, reliable_qos,
        std::bind(&LocalSearchLogger::cameraInfoCallback, this,
                  std::placeholders::_1));
    diagnostics_sub_ = this->create_subscription<DiagnosticArray>(
        diagnostics_topic_, reliable_qos,
        std::bind(&LocalSearchLogger::diagnosticsCallback, this,
                  std::placeholders::_1));
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&LocalSearchLogger::flushReadyFrames, this));

    RCLCPP_INFO(this->get_logger(),
                "Local search logger started. output=%s",
                output_dir_.c_str());
  }

private:
  struct PendingFrame {
    DiagnosticArray::ConstSharedPtr diagnostic;
    std::chrono::steady_clock::time_point received_at;
  };

  struct FrameWork {
    DiagnosticArray::ConstSharedPtr diagnostic;
    CompressedImage::ConstSharedPtr mask;
    PointCloud2::ConstSharedPtr cloud;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info;
  };

  void initializeOutput() {
    try {
      std::filesystem::create_directories(output_dir_);
    } catch (const std::exception &ex) {
      RCLCPP_FATAL(this->get_logger(), "Cannot initialize log output: %s",
                   ex.what());
      throw;
    }
  }

  template <typename ValueT>
  void trimCache(std::map<std::int64_t, ValueT> *cache) {
    while (cache->size() > static_cast<std::size_t>(cache_size_)) {
      cache->erase(cache->begin());
    }
  }

  void maskCallback(const CompressedImage::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_cache_[stampNanoseconds(msg->header.stamp)] = msg;
    trimCache(&mask_cache_);
  }

  void cloudCallback(const PointCloud2::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    cloud_cache_[stampNanoseconds(msg->header.stamp)] = msg;
    trimCache(&cloud_cache_);
  }

  void cameraInfoCallback(
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    camera_info_ = msg;
  }

  void diagnosticsCallback(const DiagnosticArray::ConstSharedPtr msg) {
    if (msg->status.empty() || msg->status.front().name != "local_search/frame") {
      return;
    }
    const auto key = stampNanoseconds(msg->header.stamp);
    std::lock_guard<std::mutex> lock(mutex_);
    pending_[key] = PendingFrame{msg, std::chrono::steady_clock::now()};
    trimCache(&pending_);
  }

  void flushReadyFrames() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<FrameWork> work;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = pending_.begin(); it != pending_.end();) {
        const double age =
            std::chrono::duration<double>(now - it->second.received_at).count();
        if (age < artifact_wait_sec_) {
          ++it;
          continue;
        }
        FrameWork item;
        item.diagnostic = it->second.diagnostic;
        const auto mask_it = mask_cache_.find(it->first);
        if (mask_it != mask_cache_.end()) {
          item.mask = mask_it->second;
        }
        const auto cloud_it = cloud_cache_.find(it->first);
        if (cloud_it != cloud_cache_.end()) {
          item.cloud = cloud_it->second;
        }
        item.camera_info = camera_info_;
        work.push_back(std::move(item));
        it = pending_.erase(it);
      }
    }

    for (const auto &item : work) {
      saveFrame(item);
    }
  }

  static std::map<std::string, std::string> diagnosticValues(
      const DiagnosticArray &diagnostic) {
    std::map<std::string, std::string> values;
    if (diagnostic.status.empty()) {
      return values;
    }
    for (const auto &item : diagnostic.status.front().values) {
      values[item.key] = item.value;
    }
    return values;
  }

  static std::string valueOr(const std::map<std::string, std::string> &values,
                             const std::string &key,
                             const std::string &fallback = "") {
    const auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  }

  static std::string sanitizePathComponent(const std::string &value) {
    std::string sanitized;
    sanitized.reserve(value.size());
    for (const unsigned char ch : value) {
      if (std::isalnum(ch) || ch == '-' || ch == '_') {
        sanitized.push_back(static_cast<char>(ch));
      } else {
        sanitized.push_back('_');
      }
    }
    return sanitized.empty() ? "unknown_action" : sanitized;
  }

  std::size_t projectCloud(const PointCloud2 &cloud,
                           const sensor_msgs::msg::CameraInfo &camera_info,
                           cv::Mat *overlay) {
    const std::string camera_frame =
        camera_frame_.empty() ? camera_info.header.frame_id : camera_frame_;
    if (camera_frame.empty()) {
      throw std::runtime_error("camera frame is empty");
    }

    const rclcpp::Time stamp(cloud.header.stamp);
    const auto transform = tf_buffer_.lookupTransform(
        camera_frame, cloud.header.frame_id, stamp,
        rclcpp::Duration::from_seconds(tf_timeout_sec_));
    const Eigen::Isometry3f cloud_to_camera =
        transformToEigen(transform.transform);

    pcl::PointCloud<pcl::PointXYZ> points;
    pcl::fromROSMsg(cloud, points);
    const double x_scale =
        camera_info.width > 0
            ? static_cast<double>(overlay->cols) / camera_info.width
            : 1.0;
    const double y_scale =
        camera_info.height > 0
            ? static_cast<double>(overlay->rows) / camera_info.height
            : 1.0;
    const double fx = camera_info.k[0];
    const double fy = camera_info.k[4];
    const double cx = camera_info.k[2];
    const double cy = camera_info.k[5];

    std::size_t projected = 0;
    for (const auto &point : points.points) {
      if (!pcl::isFinite(point)) {
        continue;
      }
      const Eigen::Vector3f camera_point =
          cloud_to_camera * Eigen::Vector3f(point.x, point.y, point.z);
      if (camera_point.z() <= 0.0F) {
        continue;
      }
      const int u = static_cast<int>(std::lround(
          (fx * camera_point.x() / camera_point.z() + cx) * x_scale));
      const int v = static_cast<int>(std::lround(
          (fy * camera_point.y() / camera_point.z() + cy) * y_scale));
      if (u < 0 || u >= overlay->cols || v < 0 || v >= overlay->rows) {
        continue;
      }
      cv::circle(*overlay, cv::Point(u, v), point_radius_px_,
                 cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
      ++projected;
    }
    return projected;
  }

  void saveFrame(const FrameWork &work) {
    const auto &stamp = work.diagnostic->header.stamp;
    const auto values = diagnosticValues(*work.diagnostic);
    const std::string action_id = sanitizePathComponent(
        valueOr(values, "action_id", "unknown_action"));
    const auto action_dir = std::filesystem::path(output_dir_) / action_id;
    try {
      std::filesystem::create_directories(action_dir / "images");
      std::filesystem::create_directories(action_dir / "pointclouds");
    } catch (const std::exception &ex) {
      RCLCPP_ERROR(this->get_logger(),
                   "Cannot create output directory for %s: %s",
                   action_id.c_str(), ex.what());
      return;
    }
    const std::string stem = stampFileStem(stamp);
    const std::string mask_relative = "images/" + stem + "_mask.png";
    const std::string projected_relative =
        "images/" + stem + "_projected.png";
    const std::string cloud_relative =
        "pointclouds/" + stem + "_matched.pcd";

    std::string mask_file;
    std::string projected_file;
    std::string cloud_file;
    std::string artifact_error;
    std::size_t projected_points = 0;
    const bool cloud_has_points =
        work.cloud && work.cloud->width > 0 && work.cloud->height > 0 &&
        !work.cloud->data.empty();
    cv::Mat mask;
    cv::Mat overlay;

    try {
      if (!work.mask) {
        throw std::runtime_error("mask with matching stamp was not received");
      }
      mask = cv_bridge::toCvCopy(work.mask)->image;
      if (mask.empty()) {
        throw std::runtime_error("decoded mask is empty");
      }
      if (!cv::imwrite((action_dir / mask_relative).string(), mask)) {
        throw std::runtime_error("failed to write mask image");
      }
      mask_file = mask_relative;

      if (mask.channels() == 1) {
        cv::cvtColor(mask, overlay, cv::COLOR_GRAY2BGR);
      } else if (mask.channels() == 4) {
        cv::cvtColor(mask, overlay, cv::COLOR_BGRA2BGR);
      } else {
        overlay = mask.clone();
      }
    } catch (const std::exception &ex) {
      artifact_error = ex.what();
    }

    if (cloud_has_points && save_matched_cloud_) {
      try {
        pcl::PointCloud<pcl::PointXYZ> points;
        pcl::fromROSMsg(*work.cloud, points);
        const auto path = action_dir / cloud_relative;
        if (pcl::io::savePCDFileBinary(path.string(), points) != 0) {
          throw std::runtime_error("failed to write matched point cloud");
        }
        cloud_file = cloud_relative;
      } catch (const std::exception &ex) {
        if (!artifact_error.empty()) artifact_error += "; ";
        artifact_error += ex.what();
      }
    }

    if (!cloud_has_points && valueOr(values, "matched_points", "0") != "0") {
      if (!artifact_error.empty()) artifact_error += "; ";
      artifact_error += "matched cloud with matching stamp was not received";
    }

    if (!overlay.empty()) {
      try {
        if (cloud_has_points) {
          if (!work.camera_info) {
            throw std::runtime_error("camera info was not received");
          }
          projected_points = projectCloud(*work.cloud, *work.camera_info, &overlay);
        }
        const std::string label =
            valueOr(values, "status", "unknown") + " matched=" +
            valueOr(values, "matched_points", "0") + " clusters=" +
            valueOr(values, "clusters", "0");
        cv::putText(overlay, label, cv::Point(12, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 3,
                    cv::LINE_AA);
        cv::putText(overlay, label, cv::Point(12, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
        if (!cv::imwrite((action_dir / projected_relative).string(), overlay)) {
          throw std::runtime_error("failed to write projected image");
        }
        projected_file = projected_relative;
      } catch (const tf2::TransformException &ex) {
        if (!artifact_error.empty()) artifact_error += "; ";
        artifact_error += std::string("overlay TF failed: ") + ex.what();
      } catch (const std::exception &ex) {
        if (!artifact_error.empty()) artifact_error += "; ";
        artifact_error += ex.what();
      }
    }

    const auto csv_path = action_dir / "frames.csv";
    try {
      const bool write_header =
          !std::filesystem::exists(csv_path) ||
          std::filesystem::file_size(csv_path) == 0;
      std::ofstream csv(csv_path, std::ios::out | std::ios::app);
      if (!csv.is_open()) {
        throw std::runtime_error("cannot open " + csv_path.string());
      }
      if (write_header) {
        csv << "action_id,stamp_ns,status,reason,requested_classes,"
               "requested_class_names,matched_classes,bbox_candidates,"
               "raw_points,finite_points,prefiltered_points,"
               "camera_front_points,bbox_points,matched_points,clusters,"
               "mask_file,projected_file,cloud_file,projected_points,"
               "artifact_error\n";
      }
      csv << csvEscape(action_id) << "," << stampNanoseconds(stamp) << ","
          << csvEscape(valueOr(values, "status", "unknown")) << ","
          << csvEscape(valueOr(values, "reason")) << ","
          << valueOr(values, "requested_classes", "0") << ","
          << csvEscape(valueOr(values, "requested_class_names")) << ","
          << valueOr(values, "matched_classes", "0") << ","
          << valueOr(values, "bbox_candidates", "0") << ","
          << valueOr(values, "raw_points", "0") << ","
          << valueOr(values, "finite_points", "0") << ","
          << valueOr(values, "prefiltered_points", "0") << ","
          << valueOr(values, "camera_front_points", "0") << ","
          << valueOr(values, "bbox_points", "0") << ","
          << valueOr(values, "matched_points", "0") << ","
          << valueOr(values, "clusters", "0") << ","
          << csvEscape(mask_file) << "," << csvEscape(projected_file) << ","
          << csvEscape(cloud_file) << "," << projected_points << ","
          << csvEscape(artifact_error) << "\n";
    } catch (const std::exception &ex) {
      RCLCPP_ERROR(this->get_logger(), "Cannot append %s: %s",
                   csv_path.c_str(), ex.what());
    }

    if (!artifact_error.empty()) {
      RCLCPP_WARN(this->get_logger(), "Frame %s logged with artifact error: %s",
                  stem.c_str(), artifact_error.c_str());
    }
  }

  std::string mask_topic_;
  std::string matched_cloud_topic_;
  std::string camera_info_topic_;
  std::string diagnostics_topic_;
  std::string camera_frame_;
  std::string output_dir_;
  double artifact_wait_sec_{0.5};
  double tf_timeout_sec_{0.2};
  int cache_size_{100};
  int point_radius_px_{2};
  bool save_matched_cloud_{true};

  std::mutex mutex_;
  std::map<std::int64_t, CompressedImage::ConstSharedPtr> mask_cache_;
  std::map<std::int64_t, PointCloud2::ConstSharedPtr> cloud_cache_;
  std::map<std::int64_t, PendingFrame> pending_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;

  rclcpp::Subscription<CompressedImage>::SharedPtr mask_sub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Subscription<DiagnosticArray>::SharedPtr diagnostics_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalSearchLogger>());
  rclcpp::shutdown();
  return 0;
}
