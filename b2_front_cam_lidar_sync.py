import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SensorSync-CAM-REF")

CAM_SYNC_PATH = "data_processed/sync/camera_synced.parquet"
LIDAR_PATH = "data_source/lidar/lidar_top_360fov.chunk_0000/0a948f59-0a06-41a2-8e20-ac3a39ff4d61.lidar_top_360fov.parquet"

OUTPUT_PARQUET = "data_processed/sync/sensor_synced_cam_ref.parquet"
OUTPUT_CSV = "data_processed/sync/sensor_synced_cam_ref_clean.csv"

# =========================
# LOAD
# =========================
cam_df = pd.read_parquet(CAM_SYNC_PATH)
lidar_df = pd.read_parquet(LIDAR_PATH)

logger.info(f"Camera: {len(cam_df)} | LiDAR: {len(lidar_df)}")

# =========================
# TIMESTAMP NORMALIZATION
# =========================
cam_df["timestamp_sec"] = cam_df["timestamp"]

# LiDAR microseconds → seconds
lidar_df["timestamp_sec"] = lidar_df["reference_timestamp"] / 1e6

# =========================
# SORT
# =========================
cam_df = cam_df.sort_values("timestamp_sec").reset_index(drop=True)
lidar_df = lidar_df.sort_values("timestamp_sec").reset_index(drop=True)

cam_ts = cam_df["timestamp_sec"].values
lidar_ts = lidar_df["timestamp_sec"].values

# =========================
# HELPER
# =========================
def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))

# =========================
# SYNC LOOP (Camera → LiDAR)
# =========================
rows = []
clean_rows = []

for i, cam_row in cam_df.iterrows():

    cam_time = cam_row["timestamp_sec"]

    # find nearest LiDAR frame
    lidar_idx = nearest_idx(lidar_ts, cam_time)
    lidar_time = lidar_ts[lidar_idx]

    diff = abs(lidar_time - cam_time)

    # =========================
    # FULL OUTPUT
    # =========================
    rows.append({
        "camera_frame": i,
        "camera_ts": cam_time,
        "lidar_frame": lidar_idx,
        "lidar_ts": lidar_time,
        "diff": diff
    })

    # =========================
    # CLEAN CSV OUTPUT
    # =========================
    clean_rows.append({
        "front wide(Frame + ts)": f"F_{i} ({cam_time:.4f})",
        "LIDAR(Frame + ts)": f"LD_{lidar_idx} ({lidar_time:.4f})"
    })

    # Debug first few
    if i < 10:
        print(f"F {i} (ts={cam_time:.4f}) → LD {lidar_idx} (diff={diff:.4f})")

# =========================
# SAVE
# =========================
final_df = pd.DataFrame(rows)
clean_df = pd.DataFrame(clean_rows)

os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

final_df.to_parquet(OUTPUT_PARQUET, index=False)
clean_df.to_csv(OUTPUT_CSV, index=False)

logger.info(f"Saved Parquet: {OUTPUT_PARQUET}")
logger.info(f"Saved Clean CSV: {OUTPUT_CSV}")
logger.info(f"Total Camera Frames: {len(final_df)}")