# get_seg_3d_coord

ROS 2 package for estimating 3D target coordinates from segmentation masks,
2D detections, recent LiDAR submaps, and odometry.

The main experimental path is `local_search`, which can log raw measurements
and EKF state so the EKF covariance parameters can be tuned from real runs.

## Local Search Summary

`local_search` expects:

- binary segmentation mask
- detection bbox stream
- recent local submap point cloud
- odometry
- reference position from the action goal

Pipeline:

1. Synchronize mask and bbox.
2. Reject measurements when odometry is not stable.
3. Project one recent submap cloud into the camera at the mask timestamp.
4. Select points inside bbox and mask.
5. Cluster selected points.
6. Pick the cluster nearest to the reference position.
7. Feed the selected 3D measurement into a static-position EKF.
8. Return when covariance is below threshold or timeout expires.

Set `pointcloud_topic` to the recent submap topic before real experiments.

## Local Search Frame Logs

`local_search_logger` runs as a separate process so image encoding and disk I/O
do not block the `local_search` action node. It is started by
`segmented_lidar_cluster.launch.py` and writes to:

```text
/home/nvidia/inha_log/module/local_search/
  frames.csv
  images/<stamp>_mask.png
  images/<stamp>_projected.png
  pointclouds/<stamp>_matched.pcd
```

The projected image overlays the LiDAR points that passed the prefilter, bbox,
and segmentation-mask tests. `frames.csv` records the raw, finite, prefiltered,
camera-front, bbox, and mask-matched point counts; bbox and cluster counts;
matched class count; status; failure reason; artifact paths; and logger-side
image/TF/write errors.

Frame statuses include `ok`, `partial_match`, `input_timeout`, `cloud_timeout`,
`no_requested_bbox`, `unstable_odom`, `projection_failed`, `no_bbox_candidate`,
`no_matching_lidar`, `no_cluster`, and `no_target_cluster`.

The output path and logger topics can be changed in `config/params.yaml` under
`local_search_logger.ros__parameters`.

## EKF Experiment Workflow

The EKF parameters are hard to choose analytically because the measurement
noise depends on segmentation quality, bbox quality, calibration error, submap
density, and robot motion. The intended workflow is:

1. Run `local_search` with EKF logging enabled.
2. Collect multiple CSV logs in `EKF_results/`.
3. Sweep candidate EKF parameters offline.
4. Score each parameter set using NIS consistency and pseudo-GT error.
5. Put the best parameters back into `config/params.yaml`.

### 1. Collect Logs

Enable logging:

```yaml
ekf_enabled: true
ekf_log_enabled: true
ekf_log_dir: "EKF_results"
```

Each action run writes one CSV file. The log includes:

- measured cluster center: `meas_x`, `meas_y`, `meas_z`
- EKF estimate before/at update: `est_x`, `est_y`, `est_z`
- covariance diagonal: `cov_x`, `cov_y`, `cov_z`
- normalized innovation squared: `nis`
- parameters used in the run header:
  - `ekf_initial_variance`
  - `ekf_measurement_variance`
  - `ekf_process_variance_per_sec`
  - `ekf_converged_variance`

`EKF_results/` is ignored by git.

### 2. Tune Parameters

Tune these three values together:

```yaml
ekf_initial_variance: 1.0          # P0
ekf_measurement_variance: 0.09     # R
ekf_process_variance_per_sec: 0.0025 # Q growth
```

Interpretation:

- Larger `ekf_measurement_variance` trusts each cluster measurement less.
- Smaller `ekf_measurement_variance` follows measurements more aggressively.
- Larger `ekf_process_variance_per_sec` lets covariance grow faster over time.
- Smaller `ekf_process_variance_per_sec` assumes the target is more static.
- Larger `ekf_initial_variance` makes the first few measurements dominate less.

### 3. Evaluate Without GT

There is no true ground truth in the current setup, so use consistency metrics.

Use NIS:

```text
NIS = innovation^T * S^-1 * innovation
```

For a 3D position measurement, reasonable NIS values should usually be near the
3D chi-square distribution. Very large repeated NIS means the chosen covariance
is too optimistic or the measurements contain outliers.

Use pseudo-GT MAE:

1. For each run, compute a robust final position from the measurements or EKF
   estimates, such as median of the last stable samples.
2. Treat that robust final position as pseudo-GT.
3. Compute MAE between each EKF estimate and the pseudo-GT.

Good parameters should:

- keep NIS from exploding repeatedly
- reduce pseudo-GT MAE quickly
- converge covariance before `ekf_timeout_sec`
- avoid following single-frame measurement jumps too aggressively

### 4. Apply Best Parameters

After offline tuning, copy the selected values into:

```yaml
local_search:
  ros__parameters:
    ekf_initial_variance: ...
    ekf_measurement_variance: ...
    ekf_process_variance_per_sec: ...
    ekf_converged_variance: ...
```

Then rerun the same scenarios and compare new logs against previous logs.

## Important Runtime Parameters

```yaml
pointcloud_topic: "/recent_submap/points"
submap_timeout_sec: 3.0
require_stable_odom: true
odom_stable_window_sec: 0.3
odom_stable_timeout_sec: 1.0
max_relative_angular_velocity_rad_s: 0.20
ekf_enabled: true
ekf_timeout_sec: 10.0
ekf_converged_variance: 0.01
```

`submap_timeout_sec` replaces previous raw LiDAR frame accumulation. The node
uses one recent submap cloud per measurement update.

## Build

This package depends on `inha_interfaces`. Make sure that package is available
and sourced before building.

```bash
colcon build --packages-select get_seg_3d_coord
```
