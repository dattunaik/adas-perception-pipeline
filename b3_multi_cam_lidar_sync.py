import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinalFusion")

# =========================
# PATHS
# =========================
CAM_PATH = "data_processed/sync/camera_synced.parquet"
LIDAR_PATH = "data_processed/sync/sensor_synced_cam_ref.parquet"

OUT_DIR = "data_processed/final_fusion"
OUT_PARQUET = os.path.join(OUT_DIR, "final_fused.parquet")
OUT_CSV = os.path.join(OUT_DIR, "final_fused_table.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
logger.info("Loading input files...")

cam_df = pd.read_parquet(CAM_PATH)
lidar_df = pd.read_parquet(LIDAR_PATH)

logger.info(f"Camera rows: {len(cam_df)}")
logger.info(f"LiDAR rows: {len(lidar_df)}")

# =========================
# RENAME FRONT FRAME
# =========================
cam_df = cam_df.rename(columns={
    "camera_front_wide_120fov.chunk_0000_frame": "camera_frame"
})

# =========================
# MERGE
# =========================
df = pd.merge(
    cam_df,
    lidar_df,
    on="camera_frame",
    how="left"
)

logger.info(f"Final fused rows: {len(df)}")

# =========================
# BUILD CLEAN TABLE
# =========================
table_rows = []

for _, r in df.iterrows():

    # FRONT
    front_frame = int(r["camera_frame"])
    front_ts = r["timestamp"]

    # =========================
    # LEFT
    # =========================
    lf = r.get("camera_cross_left_120fov.chunk_0000_frame")
    ld = r.get("camera_cross_left_120fov.chunk_0000_diff")

    if pd.notna(lf) and pd.notna(ld):
        left_frame = int(lf)
        left_ts = front_ts + ld
        fl_diff = ld
    else:
        left_frame = None
        left_ts = None
        fl_diff = None

    # =========================
    # RIGHT
    # =========================
    rf = r.get("camera_cross_right_120fov.chunk_0000_frame")
    rd = r.get("camera_cross_right_120fov.chunk_0000_diff")

    if pd.notna(rf) and pd.notna(rd):
        right_frame = int(rf)
        right_ts = front_ts + rd
        fr_diff = rd
    else:
        right_frame = None
        right_ts = None
        fr_diff = None

    # =========================
    # LIDAR
    # =========================
    lidar_frame = r.get("lidar_frame")
    lidar_ts = r.get("lidar_ts")

    if pd.notna(lidar_frame):
        lidar_frame = int(lidar_frame)
        fld_diff = lidar_ts - front_ts
    else:
        lidar_frame = None
        lidar_ts = None
        fld_diff = None

    # =========================
    # FINAL ROW
    # =========================
    table_rows.append({
        "front_frame": front_frame,
        "front_timestamp": front_ts,

        "left_frame": left_frame,
        "left_timestamp": left_ts,
        "f_l_timestamp_diff": fl_diff,

        "right_frame": right_frame,
        "right_timestamp": right_ts,
        "f_r_timestamp_diff": fr_diff,

        "lidar_frame": lidar_frame,
        "lidar_timestamp": lidar_ts,
        "f_ld_timestamp_diff": fld_diff
    })

# =========================
# SAVE OUTPUT
# =========================
df_table = pd.DataFrame(table_rows)

df.to_parquet(OUT_PARQUET, index=False)
df_table.to_csv(OUT_CSV, index=False)

logger.info(f"Parquet saved → {OUT_PARQUET}")
logger.info(f"CSV saved → {OUT_CSV}")

logger.info("===== FINAL FUSION COMPLETED =====")