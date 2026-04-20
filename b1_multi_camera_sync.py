import os
import pandas as pd
import numpy as np
import logging
import traceback

def main():

    # =========================
    # LOGGER
    # =========================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logger = logging.getLogger("CameraSync")

    logger.info("===== CAMERA SYNC STARTED =====")

    # =========================
    # PATHS (MATCH EXTRACTION SCRIPT)
    # =========================
    INPUT_BASE = os.environ.get(
        "INPUT_BASE",
        "data_processed/extracted_frames"
    )

    OUTPUT_BASE = os.environ.get(
        "OUTPUT_BASE",
        "data_processed/camera_sync"
    )

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # =========================
    # TIME CONFIG
    # =========================
    GOOD_THRESHOLD = 0.033
    MAX_THRESHOLD = 0.1

    all_data = {}

    # =========================
    # LOAD DATA
    # =========================
    if not os.path.exists(INPUT_BASE):
        raise ValueError(f"Input path not found: {INPUT_BASE}")

    for cam in os.listdir(INPUT_BASE):

        cam_path = os.path.join(INPUT_BASE, cam)

        if not os.path.isdir(cam_path):
            continue

        for clip in os.listdir(cam_path):

            meta_path = os.path.join(cam_path, clip, "metadata.parquet")

            if not os.path.exists(meta_path):
                continue

            try:
                df = pd.read_parquet(meta_path)

                if "timestamp" not in df.columns:
                    continue

                df["timestamp_sec"] = df["timestamp"] / 1e6
                df = df.sort_values("timestamp_sec").reset_index(drop=True)

                if clip not in all_data:
                    all_data[clip] = {}

                all_data[clip][cam] = df

            except Exception as e:
                logger.error(f"Failed reading {meta_path}")
                traceback.print_exc()

    # =========================
    # PROCESS CLIPS
    # =========================
    for clip_id, cam_data in all_data.items():

        try:
            if len(cam_data) < 2:
                logger.warning(f"{clip_id}: insufficient cameras")
                continue

            # =========================
            # SELECT BASE CAMERA
            # =========================
            base_cam = None
            for cam in cam_data:
                if "front_wide" in cam:
                    base_cam = cam
                    break

            if base_cam is None:
                logger.warning(f"{clip_id}: front_wide not found")
                continue

            logger.info(f"{clip_id}: Base camera → {base_cam}")

            base_df = cam_data[base_cam]

            def nearest_index(arr, val):
                return int(np.argmin(np.abs(arr - val)))

            rows = []

            # =========================
            # SYNC
            # =========================
            for _, row in base_df.iterrows():

                base_ts = row["timestamp_sec"]
                base_frame = int(row["frame_id"])

                sync_row = {
                    "clip_id": clip_id,
                    "timestamp": base_ts,
                    f"{base_cam}_frame": base_frame,
                    f"{base_cam}_path": row.get("image_path")
                }

                valid = 1

                for cam, df in cam_data.items():

                    if cam == base_cam:
                        continue

                    ts = df["timestamp_sec"].values
                    idx = nearest_index(ts, base_ts)

                    match_ts = ts[idx]
                    diff = abs(match_ts - base_ts)
                    diff_ms = diff * 1000

                    frame_id = int(df.iloc[idx]["frame_id"])

                    # =========================
                    # FALLBACK LOGIC
                    # =========================
                    if diff > MAX_THRESHOLD:

                        fallback_row = df.iloc[0]

                        frame_id = int(fallback_row["frame_id"])
                        match_ts = fallback_row["timestamp_sec"]

                        diff = abs(match_ts - base_ts)

                        logger.warning(
                            f"{clip_id} | FALLBACK {cam} diff={diff_ms:.2f}ms"
                        )

                    elif diff <= GOOD_THRESHOLD:
                        valid += 1

                    sync_row[f"{cam}_frame"] = frame_id
                    sync_row[f"{cam}_path"] = df.iloc[idx].get("image_path")
                    sync_row[f"{cam}_diff"] = float(diff)

                if valid >= 2:
                    rows.append(sync_row)

            if not rows:
                logger.warning(f"{clip_id}: no valid frames")
                continue

            synced_df = pd.DataFrame(rows)

            # =========================
            # SAVE OUTPUT
            # =========================
            out_dir = os.path.join(OUTPUT_BASE, clip_id)
            os.makedirs(out_dir, exist_ok=True)

            parquet_path = os.path.join(out_dir, "camera_sync.parquet")
            csv_path = os.path.join(out_dir, "camera_sync.csv")

            synced_df.to_parquet(parquet_path, index=False)
            synced_df.to_csv(csv_path, index=False)

            logger.info(f"{clip_id}: saved")

        except Exception as e:
            logger.error(f"{clip_id}: processing failed")
            traceback.print_exc()

    logger.info("===== CAMERA SYNC COMPLETED =====")


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    main()