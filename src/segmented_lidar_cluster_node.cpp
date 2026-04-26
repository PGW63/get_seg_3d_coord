#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/imgproc.hpp>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class SegmentedLidarClusterNode : public rclcpp::Node {
public:
  SegmentedLidarClusterNode()
      : Node("segmented_lidar_cluster_node"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    mask_topic_ = this->declare_parameter<std::string>("mask_topic",
                                                       "/sam2_segmentation_node/binary_mask/compressed");
    pointcloud_topic_ = this->declare_parameter<std::string>("pointcloud_topic",
                                                             "/livox/lidar");
    camera_info_topic_ = this->declare_parameter<std::string>(
        "camera_info_topic", "/camera/camera_head/color/camera_info");
    filtered_cloud_topic_ = this->declare_parameter<std::string>(
        "filtered_cloud_topic", "/segmented_lidar/points");
    centers_topic_ = this->declare_parameter<std::string>(
        "centers_topic", "/segmented_lidar/cluster_centers");
    markers_topic_ = this->declare_parameter<std::string>(
        "markers_topic", "/segmented_lidar/cluster_markers");
    camera_frame_param_ =
        this->declare_parameter<std::string>("camera_frame", "");
    mask_threshold_ = this->declare_parameter<int>("mask_threshold", 0);
    cluster_tolerance_ =
        this->declare_parameter<double>("cluster_tolerance", 0.30);
    min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 8);
    max_cluster_size_ =
        this->declare_parameter<int>("max_cluster_size", 30000);
    const int sync_queue_size =
        this->declare_parameter<int>("sync_queue_size", 10);
    const double sync_slop = this->declare_parameter<double>("sync_slop", 0.10);
    tf_timeout_sec_ = this->declare_parameter<double>("tf_timeout_sec", 0.05);

    auto best_effort_qos =
        rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, reliable_qos,
        std::bind(&SegmentedLidarClusterNode::cameraInfoCallback, this,
                  std::placeholders::_1));

    filtered_cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            filtered_cloud_topic_, best_effort_qos);
    centers_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
        centers_topic_, reliable_qos);
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        markers_topic_, reliable_qos);

    mask_sub_.subscribe(this, mask_topic_,
                        best_effort_qos.get_rmw_qos_profile());
    cloud_sub_.subscribe(this, pointcloud_topic_,
                         best_effort_qos.get_rmw_qos_profile());

    sync_ = std::make_shared<Synchronizer>(
        SyncPolicy(sync_queue_size), mask_sub_, cloud_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_slop));
    sync_->registerCallback(std::bind(
        &SegmentedLidarClusterNode::syncedCallback, this, std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(),
                "Segmented LiDAR clustering node started. mask=%s, cloud=%s, "
                "camera_info=%s",
                mask_topic_.c_str(), pointcloud_topic_.c_str(),
                camera_info_topic_.c_str());
  }

private:
  using MaskMsg = sensor_msgs::msg::CompressedImage;
  using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<MaskMsg, PointCloud2Msg>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    camera_info_ = msg;
  }

  void syncedCallback(const MaskMsg::ConstSharedPtr &mask_msg,
                      const PointCloud2Msg::ConstSharedPtr &cloud_msg) {
    if (!camera_info_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Waiting for CameraInfo.");
      return;
    }

    const std::string camera_frame =
        camera_frame_param_.empty() ? camera_info_->header.frame_id
                                    : camera_frame_param_;
    if (camera_frame.empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "Camera frame is empty. Set camera_frame or publish "
                  "CameraInfo frame_id.");
      return;
    }

    cv::Mat mask = imageToBinaryMask(mask_msg);
    if (mask.empty()) {
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

    auto lidar_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*cloud_msg, *lidar_cloud);

    if (lidar_cloud->empty()) {
      publishEmpty(cloud_msg->header);
      return;
    }

    const Eigen::Isometry3f lidar_to_camera =
        transformToEigen(transform.transform);
    auto selected_cloud = selectPointsOnMask(lidar_cloud, lidar_to_camera, mask);
    auto cluster_indices = clusterPoints(selected_cloud);
    auto centers = clusterCenters(selected_cloud, cluster_indices);

    publishFilteredCloud(cloud_msg->header, selected_cloud);
    publishCenters(cloud_msg->header, centers);
    publishMarkers(cloud_msg->header, selected_cloud, cluster_indices, centers);
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

  pcl::PointCloud<pcl::PointXYZ>::Ptr selectPointsOnMask(
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &lidar_cloud,
      const Eigen::Isometry3f &lidar_to_camera, const cv::Mat &mask) const {
    auto selected = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    selected->header = lidar_cloud->header;
    selected->is_dense = false;

    const double fx = camera_info_->k[0];
    const double fy = camera_info_->k[4];
    const double cx = camera_info_->k[2];
    const double cy = camera_info_->k[5];
    const double x_scale = camera_info_->width > 0
                               ? static_cast<double>(mask.cols) /
                                     static_cast<double>(camera_info_->width)
                               : 1.0;
    const double y_scale = camera_info_->height > 0
                               ? static_cast<double>(mask.rows) /
                                     static_cast<double>(camera_info_->height)
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

  void publishEmpty(const std_msgs::msg::Header &header) const {
    auto empty_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    publishFilteredCloud(header, empty_cloud);
    publishCenters(header, {});
    publishMarkers(header, empty_cloud, {}, {});
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
  std::string markers_topic_;
  std::string camera_frame_param_;
  int mask_threshold_{0};
  double cluster_tolerance_{0.30};
  int min_cluster_size_{8};
  int max_cluster_size_{30000};
  double tf_timeout_sec_{0.05};

  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Publisher<PointCloud2Msg>::SharedPtr filtered_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr centers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      markers_pub_;

  message_filters::Subscriber<MaskMsg> mask_sub_;
  message_filters::Subscriber<PointCloud2Msg> cloud_sub_;
  std::shared_ptr<Synchronizer> sync_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SegmentedLidarClusterNode>());
  rclcpp::shutdown();
  return 0;
}
