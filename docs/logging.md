# Local Search Frame Logs


`local_search_logger` runs as a separate process so image encoding and disk I/O
do not block the `local_search` action node. It is started by
`segmented_lidar_cluster.launch.py` and writes to:

```text
/home/nvidia/inha_log/module/local_search/
  action_20260622_153045_123_KST/
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

