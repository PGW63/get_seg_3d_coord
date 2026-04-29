#!/usr/bin/env python3
"""
LocalSearchNode

Nav2가 사전 맵 기반으로 expected_pose 근처에 도착한 뒤, 실제로 그 자리에
타겟 객체가 있는지 검증하고, 검증되면 정밀한 3D pose(map frame)를 반환하는
액션 서버.

상위 BT/state machine은 이 액션의 result.target_pose로 정밀 접근 goal을
다시 Nav2에 줄 수 있다.

검증 전략 (단일 프레임 false positive 방지를 위해 두 게이트를 AND로):
  (1) Temporal consistency  : detect_window_sec 동안 min_consistent_frames 이상
                              연속 bbox 검출
  (2) 3D position sanity    : SAM2 mask × Livox 클러스터 중심을 map frame으로
                              TF한 뒤 expected_pose와 position_tolerance 이내

검증 실패 시 active perception:
  Nav2 /spin 액션으로 yaw_step_deg 만큼 제자리 회전 후 재시도.
  max_yaw_attempts 까지 못 찾으면 NOT_FOUND.

전제 노드:
  - generic_object_detection_node (GroundingDINO TRT) — /detection/stream srv,
    /detection/bbox_stream 토픽
  - sam2_segmentation_node — binary_mask 토픽
  - table_person_association — associate_cluster 액션, cluster_centers 토픽
  - Nav2 — /spin 액션
"""

from collections import deque
import math
import threading
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseArray, PoseStamped
from vision_msgs.msg import Detection2DArray
from nav2_msgs.action import Spin

from tf2_ros import Buffer, TransformException, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped  # noqa: F401  (registers converter)

from get_seg_3d_coord.action import GetSegCoord, LocalSearch
from inha_interfaces.srv import DetectionControl


class LocalSearchNode(Node):

    def __init__(self):
        super().__init__('local_search_node')

        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('bbox_topic', '/detection/bbox_stream')
        self.declare_parameter('cluster_centers_topic',
                               '/segmented_lidar/cluster_centers')
        self.declare_parameter('detection_stream_srv', '/detection/stream')
        self.declare_parameter('get_seg_coord_action', 'get_seg_coord')
        self.declare_parameter('spin_action', 'spin')
        self.declare_parameter('local_search_action', 'local_search')
        self.declare_parameter('bbox_score_threshold', 0.30)
        self.declare_parameter('action_call_timeout_sec', 8.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.bbox_score_threshold = self.get_parameter(
            'bbox_score_threshold').value
        self.action_call_timeout = self.get_parameter(
            'action_call_timeout_sec').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )

        self._bbox_lock = threading.Lock()
        self._bbox_history = deque(maxlen=64)  # (stamp_sec, n_valid_dets)
        self.create_subscription(
            Detection2DArray,
            self.get_parameter('bbox_topic').value,
            self._bbox_callback,
            sensor_qos,
            callback_group=self.cb_group,
        )

        self._centers_lock = threading.Lock()
        self._latest_centers = None  # (stamp_sec, frame_id, list[(x,y,z)])
        self.create_subscription(
            PoseArray,
            self.get_parameter('cluster_centers_topic').value,
            self._centers_callback,
            sensor_qos,
            callback_group=self.cb_group,
        )

        self.detect_stream_cli = self.create_client(
            DetectionControl,
            self.get_parameter('detection_stream_srv').value,
            callback_group=self.cb_group,
        )
        self.get_seg_coord_cli = ActionClient(
            self, GetSegCoord,
            self.get_parameter('get_seg_coord_action').value,
            callback_group=self.cb_group,
        )
        self.spin_cli = ActionClient(
            self, Spin,
            self.get_parameter('spin_action').value,
            callback_group=self.cb_group,
        )

        self._action_server = ActionServer(
            self, LocalSearch,
            self.get_parameter('local_search_action').value,
            execute_callback=self._execute_callback,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=self.cb_group,
        )

        self.get_logger().info('LocalSearchNode ready.')

    # ------------------------------------------------------------------ subs
    def _bbox_callback(self, msg: Detection2DArray):
        n_valid = sum(
            1 for det in msg.detections
            if det.results and max(r.hypothesis.score for r in det.results)
            >= self.bbox_score_threshold
        )
        stamp = self.get_clock().now().nanoseconds * 1e-9
        with self._bbox_lock:
            self._bbox_history.append((stamp, n_valid))

    def _centers_callback(self, msg: PoseArray):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pts = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        with self._centers_lock:
            self._latest_centers = (stamp, msg.header.frame_id, pts)

    # -------------------------------------------------------- helper methods
    def _set_dino_stream(self, target_text: str) -> bool:
        if not self.detect_stream_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('detection/stream service unavailable')
            return False
        req = DetectionControl.Request()
        req.target_object = target_text
        future = self.detect_stream_cli.call_async(req)
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            return False
        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().warn(
                f'stream srv failed: {resp.message if resp else "no resp"}')
            return False
        return True

    def _count_recent_consistent_frames(self, window_sec: float) -> int:
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._bbox_lock:
            return sum(
                1 for (ts, n) in self._bbox_history
                if n > 0 and (now - ts) <= window_sec
            )

    def _wait_for_temporal_consistency(self, min_frames: int,
                                       window_sec: float,
                                       deadline: float) -> bool:
        rate = 0.05
        while time.monotonic() < deadline:
            if self._count_recent_consistent_frames(window_sec) >= min_frames:
                return True
            time.sleep(rate)
        return False

    def _request_cluster_centers(self, decay_frames: int):
        if not self.get_seg_coord_cli.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('get_seg_coord action unavailable')
            return None

        # Mark a watermark so we only accept centers published AFTER this call.
        watermark = self.get_clock().now().nanoseconds * 1e-9

        goal = GetSegCoord.Goal()
        goal.start = True
        goal.decay_num = float(decay_frames)

        send_future = self.get_seg_coord_cli.send_goal_async(goal)
        if not self._spin_until(send_future, self.action_call_timeout):
            return None
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn('get_seg_coord goal rejected')
            return None

        result_future = gh.get_result_async()
        if not self._spin_until(result_future, self.action_call_timeout):
            self.get_logger().warn('get_seg_coord timed out')
            return None
        result_wrapper = result_future.result()
        if result_wrapper is None:
            return None
        result = result_wrapper.result
        if not result.success:
            self.get_logger().info(
                f'get_seg_coord no result: {result.message}')
            return None

        # Prefer freshly-published PoseArray (has frame_id); fall back to flat.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._centers_lock:
                latest = self._latest_centers
            if latest is not None and latest[0] >= watermark and latest[2]:
                return latest  # (stamp, frame_id, pts)
            time.sleep(0.02)

        flat = result.center
        if not flat or len(flat) % 3 != 0:
            return None
        pts = [(flat[i], flat[i + 1], flat[i + 2])
               for i in range(0, len(flat), 3)]
        self.get_logger().warn(
            'centers PoseArray not received in time; using action result '
            'with no frame_id (assuming map frame is wrong — check TF).')
        return (watermark, '', pts)

    @staticmethod
    def _spin_until(future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        return future.done()

    def _transform_to_map(self, frame_id: str, x: float, y: float, z: float,
                          stamp_sec: float):
        if frame_id == self.map_frame:
            return (x, y, z)
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp.sec = int(stamp_sec)
        ps.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.w = 1.0
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3))
            out = do_transform_pose_stamped(ps, tf)
            return (out.pose.position.x, out.pose.position.y,
                    out.pose.position.z)
        except TransformException as e:
            self.get_logger().warn(f'TF {frame_id}->{self.map_frame}: {e}')
            return None

    def _spin_robot(self, yaw_deg: float) -> bool:
        if not self.spin_cli.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('spin action unavailable')
            return False
        goal = Spin.Goal()
        goal.target_yaw = math.radians(yaw_deg)
        send_future = self.spin_cli.send_goal_async(goal)
        if not self._spin_until(send_future, 5.0):
            return False
        gh = send_future.result()
        if gh is None or not gh.accepted:
            return False
        result_future = gh.get_result_async()
        if not self._spin_until(result_future, 30.0):
            return False
        return True

    # -------------------------------------------------------- main algorithm
    def _execute_callback(self, goal_handle):
        goal = goal_handle.request
        target = (goal.target_text or '').strip()
        if not target:
            return self._fail(goal_handle, 'target_text is empty')

        expected = goal.expected_pose
        if expected.header.frame_id and expected.header.frame_id != self.map_frame:
            return self._fail(
                goal_handle,
                f'expected_pose must be in {self.map_frame}, got '
                f'{expected.header.frame_id!r}')

        tol = goal.position_tolerance if goal.position_tolerance > 0 else 1.2
        yaw_step = goal.yaw_step_deg if goal.yaw_step_deg > 0 else 30.0
        max_attempts = goal.max_yaw_attempts if goal.max_yaw_attempts > 0 else 12
        decay = goal.decay_frames if goal.decay_frames > 0 else 10
        settle = goal.settle_time_sec if goal.settle_time_sec > 0 else 0.6
        min_frames = goal.min_consistent_frames if goal.min_consistent_frames > 0 else 4
        window = goal.detect_window_sec if goal.detect_window_sec > 0 else 1.5

        ex_xy = (expected.pose.position.x, expected.pose.position.y)

        self._publish_feedback(goal_handle, 'SETTLE', 0, float('inf'))
        time.sleep(settle)

        if not self._set_dino_stream(target):
            return self._fail(goal_handle, 'failed to enable DINO stream')

        best_overall = float('inf')
        best_pose_map = None

        try:
            for attempt in range(max_attempts):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return LocalSearch.Result(
                        success=False, message='canceled')

                # (1) temporal consistency
                self._publish_feedback(goal_handle, 'DETECT', attempt,
                                       best_overall)
                ok = self._wait_for_temporal_consistency(
                    min_frames, window,
                    deadline=time.monotonic() + max(window * 2.0, 3.0))
                if not ok:
                    self.get_logger().info(
                        f'[attempt {attempt}] no consistent detection')
                    self._spin_or_break(yaw_step, attempt, max_attempts,
                                        goal_handle)
                    continue

                # (2) 3D cluster + position sanity
                self._publish_feedback(goal_handle, 'CLUSTER', attempt,
                                       best_overall)
                centers = self._request_cluster_centers(decay)
                if centers is None:
                    self.get_logger().info(
                        f'[attempt {attempt}] no cluster centers')
                    self._spin_or_break(yaw_step, attempt, max_attempts,
                                        goal_handle)
                    continue

                stamp_sec, frame_id, pts = centers
                best_dist = float('inf')
                best_xyz_map = None
                for (x, y, z) in pts:
                    xyz_map = self._transform_to_map(frame_id, x, y, z,
                                                     stamp_sec)
                    if xyz_map is None:
                        continue
                    d = math.hypot(xyz_map[0] - ex_xy[0],
                                   xyz_map[1] - ex_xy[1])
                    if d < best_dist:
                        best_dist = d
                        best_xyz_map = xyz_map

                if best_xyz_map is not None and best_dist < best_overall:
                    best_overall = best_dist
                    best_pose_map = best_xyz_map

                self.get_logger().info(
                    f'[attempt {attempt}] {len(pts)} centers, '
                    f'best_dist={best_dist:.2f}m (tol={tol:.2f})')

                if best_xyz_map is not None and best_dist <= tol:
                    return self._succeed(goal_handle, best_xyz_map, best_dist)

                self._spin_or_break(yaw_step, attempt, max_attempts,
                                    goal_handle)

            # Exhausted attempts. Optionally accept best-so-far if within
            # a relaxed tolerance? We choose strict NOT_FOUND.
            if best_pose_map is not None:
                msg = (f'no candidate within {tol:.2f}m '
                       f'(best={best_overall:.2f}m)')
            else:
                msg = 'no valid 3D candidate'
            return self._fail(goal_handle, msg, best_overall)

        finally:
            self._set_dino_stream('')  # disable stream regardless of outcome

    def _spin_or_break(self, yaw_step, attempt, max_attempts, goal_handle):
        if attempt + 1 >= max_attempts:
            return
        self._publish_feedback(goal_handle, 'ROTATE', attempt, float('inf'))
        self._spin_robot(yaw_step)
        # purge stale bbox history so next attempt judges fresh frames only
        with self._bbox_lock:
            self._bbox_history.clear()

    # -------------------------------------------------------------- result
    def _succeed(self, goal_handle, xyz_map, dist):
        result = LocalSearch.Result()
        result.success = True
        result.message = f'verified at {dist:.2f}m from expected'
        result.target_pose.header.frame_id = self.map_frame
        result.target_pose.header.stamp = self.get_clock().now().to_msg()
        result.target_pose.pose.position.x = float(xyz_map[0])
        result.target_pose.pose.position.y = float(xyz_map[1])
        result.target_pose.pose.position.z = float(xyz_map[2])
        result.target_pose.pose.orientation.w = 1.0
        result.position_error = float(dist)
        goal_handle.succeed()
        return result

    def _fail(self, goal_handle, message, dist=float('inf')):
        result = LocalSearch.Result()
        result.success = False
        result.message = message
        result.position_error = float(dist) if math.isfinite(dist) else -1.0
        goal_handle.abort()
        self.get_logger().info(f'LocalSearch FAIL: {message}')
        return result

    def _publish_feedback(self, goal_handle, state, attempt, best_dist):
        fb = LocalSearch.Feedback()
        fb.state = state
        fb.attempt = int(attempt)
        fb.best_distance = float(
            best_dist if math.isfinite(best_dist) else -1.0)
        try:
            goal_handle.publish_feedback(fb)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = LocalSearchNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
