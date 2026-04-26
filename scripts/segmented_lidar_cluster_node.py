#!/usr/bin/env python3

from collections import defaultdict, deque
import math

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class SegmentedLidarClusterNode(Node):
    def __init__(self):
        super().__init__("segmented_lidar_cluster_node")
        self.bridge = CvBridge()
        self.camera_info = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter("mask_topic", "/sam2_segmentation_node/binary_mask/compressed")
        self.declare_parameter("pointcloud_topic", "/lidar/points")
        self.declare_parameter("camera_info_topic", "/camera/camera_head/color/camera_info")
        self.declare_parameter("filtered_cloud_topic", "/segmented_lidar/points")
        self.declare_parameter("centers_topic", "/segmented_lidar/cluster_centers")
        self.declare_parameter("markers_topic", "/segmented_lidar/cluster_markers")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("mask_threshold", 0)
        self.declare_parameter("cluster_tolerance", 0.30)
        self.declare_parameter("min_cluster_size", 8)
        self.declare_parameter("max_cluster_size", 30000)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.10)
        self.declare_parameter("tf_timeout_sec", 0.05)

        self.mask_topic = self.get_parameter("mask_topic").value
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.camera_frame_param = self.get_parameter("camera_frame").value
        self.mask_threshold = int(self.get_parameter("mask_threshold").value)
        self.cluster_tolerance = float(self.get_parameter("cluster_tolerance").value)
        self.min_cluster_size = int(self.get_parameter("min_cluster_size").value)
        self.max_cluster_size = int(self.get_parameter("max_cluster_size").value)
        self.tf_timeout = Duration(seconds=float(self.get_parameter("tf_timeout_sec").value))

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            reliable_qos,
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter("filtered_cloud_topic").value,
            qos,
        )
        self.centers_pub = self.create_publisher(
            PoseArray,
            self.get_parameter("centers_topic").value,
            reliable_qos,
        )
        self.markers_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter("markers_topic").value,
            reliable_qos,
        )

        self.mask_sub = message_filters.Subscriber(self, CompressedImage, self.mask_topic, qos_profile=qos)
        self.cloud_sub = message_filters.Subscriber(
            self,
            PointCloud2,
            self.pointcloud_topic,
            qos_profile=qos,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.mask_sub, self.cloud_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(
            "Segmented LiDAR clustering node started. "
            f"mask={self.mask_topic}, cloud={self.pointcloud_topic}, camera_info={self.camera_info_topic}"
        )

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def synced_callback(self, mask_msg, cloud_msg):
        if self.camera_info is None:
            self.get_logger().warn("Waiting for CameraInfo.", throttle_duration_sec=2.0)
            return

        camera_frame = self.camera_frame_param or self.camera_info.header.frame_id
        if not camera_frame:
            self.get_logger().warn("Camera frame is empty. Set camera_frame or publish CameraInfo frame_id.")
            return

        mask = self._mask_to_binary(mask_msg)
        if mask is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                camera_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                timeout=self.tf_timeout,
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"Cannot transform cloud frame '{cloud_msg.header.frame_id}' to camera frame "
                f"'{camera_frame}': {exc}",
                throttle_duration_sec=2.0,
            )
            return

        lidar_points = self._read_xyz_points(cloud_msg)
        if lidar_points.size == 0:
            self._publish_empty(cloud_msg)
            return

        camera_points = self._transform_points(lidar_points, transform)
        selected_points = self._select_points_on_mask(lidar_points, camera_points, mask)
        clusters = self._cluster_points(selected_points)
        centers = self._cluster_centers(selected_points, clusters)

        self._publish_filtered_cloud(cloud_msg, selected_points)
        self._publish_centers(cloud_msg.header, centers)
        self._publish_markers(cloud_msg.header, centers, clusters, selected_points)

    def _mask_to_binary(self, mask_msg):
        try:
            mask = self.bridge.compressed_imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert mask image: {exc}", throttle_duration_sec=2.0)
            return None

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask > self.mask_threshold

    def _read_xyz_points(self, cloud_msg):
        points = point_cloud2.read_points(
            cloud_msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        if isinstance(points, np.ndarray):
            if points.dtype.names:
                array = np.column_stack((points["x"], points["y"], points["z"])).astype(np.float32)
            else:
                array = np.asarray(points, dtype=np.float32)
        else:
            array = np.asarray(list(points), dtype=np.float32)

        if array.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return array.reshape((-1, 3))

    def _transform_points(self, points, transform):
        rotation = self._quaternion_to_matrix(transform.transform.rotation)
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float32,
        )
        return points @ rotation.T + translation

    def _select_points_on_mask(self, lidar_points, camera_points, mask):
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        z = camera_points[:, 2]
        valid_depth = z > 0.0
        if not np.any(valid_depth):
            return np.empty((0, 3), dtype=np.float32)

        projected_u = np.zeros(camera_points.shape[0], dtype=np.int32)
        projected_v = np.zeros(camera_points.shape[0], dtype=np.int32)
        valid_indices = np.where(valid_depth)[0]

        projected_u[valid_indices] = np.round(
            fx * camera_points[valid_indices, 0] / z[valid_indices] + cx
        ).astype(np.int32)
        projected_v[valid_indices] = np.round(
            fy * camera_points[valid_indices, 1] / z[valid_indices] + cy
        ).astype(np.int32)

        height, width = mask.shape[:2]
        if self.camera_info.width > 0 and self.camera_info.height > 0:
            projected_u = np.round(projected_u * (width / self.camera_info.width)).astype(np.int32)
            projected_v = np.round(projected_v * (height / self.camera_info.height)).astype(np.int32)

        in_image = (
            valid_depth
            & (projected_u >= 0)
            & (projected_u < width)
            & (projected_v >= 0)
            & (projected_v < height)
        )
        if not np.any(in_image):
            return np.empty((0, 3), dtype=np.float32)

        image_indices = np.where(in_image)[0]
        on_mask = mask[projected_v[image_indices], projected_u[image_indices]]
        selected_indices = image_indices[on_mask]
        return lidar_points[selected_indices]

    def _cluster_points(self, points):
        if points.shape[0] < self.min_cluster_size:
            return []

        cell_size = self.cluster_tolerance
        cells = defaultdict(list)
        for index, point in enumerate(points):
            key = tuple(np.floor(point / cell_size).astype(np.int32).tolist())
            cells[key].append(index)

        visited = np.zeros(points.shape[0], dtype=bool)
        clusters = []

        for start_index in range(points.shape[0]):
            if visited[start_index]:
                continue

            cluster = []
            queue = deque([start_index])
            visited[start_index] = True

            while queue:
                current = queue.popleft()
                cluster.append(current)
                current_key = tuple(np.floor(points[current] / cell_size).astype(np.int32).tolist())

                for neighbor_key in self._neighbor_cell_keys(current_key):
                    for neighbor in cells.get(neighbor_key, []):
                        if visited[neighbor]:
                            continue
                        if np.linalg.norm(points[current] - points[neighbor]) <= self.cluster_tolerance:
                            visited[neighbor] = True
                            queue.append(neighbor)

            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                clusters.append(np.array(cluster, dtype=np.int32))

        return clusters

    def _neighbor_cell_keys(self, key):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield key[0] + dx, key[1] + dy, key[2] + dz

    def _cluster_centers(self, points, clusters):
        centers = []
        for cluster in clusters:
            centers.append(np.mean(points[cluster], axis=0))
        return centers

    def _publish_filtered_cloud(self, source_msg, selected_points):
        cloud = point_cloud2.create_cloud_xyz32(source_msg.header, selected_points.tolist())
        self.cloud_pub.publish(cloud)

    def _publish_centers(self, header, centers):
        pose_array = PoseArray()
        pose_array.header = header

        for center in centers:
            pose = Pose()
            pose.position.x = float(center[0])
            pose.position.y = float(center[1])
            pose.position.z = float(center[2])
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.centers_pub.publish(pose_array)

    def _publish_markers(self, header, centers, clusters, points):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for marker_id, (center, cluster) in enumerate(zip(centers, clusters)):
            cluster_points = points[cluster]
            min_xyz = np.min(cluster_points, axis=0)
            max_xyz = np.max(cluster_points, axis=0)
            extent = np.maximum(max_xyz - min_xyz, 0.05)

            marker = Marker()
            marker.header = header
            marker.ns = "segmented_lidar_clusters"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(extent[0])
            marker.scale.y = float(extent[1])
            marker.scale.z = float(extent[2])
            marker.color.r = 0.1
            marker.color.g = 0.8
            marker.color.b = 1.0
            marker.color.a = 0.35
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)

            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "segmented_lidar_cluster_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(center[0])
            text_marker.pose.position.y = float(center[1])
            text_marker.pose.position.z = float(center[2] + extent[2] * 0.5 + 0.15)
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.18
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = (
                f"{marker_id}: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) "
                f"n={len(cluster)}"
            )
            marker_array.markers.append(text_marker)

        self.markers_pub.publish(marker_array)

    def _publish_empty(self, source_msg):
        self._publish_filtered_cloud(source_msg, np.empty((0, 3), dtype=np.float32))
        self._publish_centers(source_msg.header, [])
        self._publish_markers(source_msg.header, [], [], np.empty((0, 3), dtype=np.float32))

    def _quaternion_to_matrix(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return np.eye(3, dtype=np.float32)

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )


def main(args=None):
    rclpy.init(args=args)
    node = SegmentedLidarClusterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
