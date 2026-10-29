# EKF Tuning Workflow


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

