import os
import traceback
import logging
import cv2
import pandas as pd

def main():

    # =========================
    # LOGGER (inside main ONLY)
    # =========================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logger = logging.getLogger("FrameExtraction")

    logger.info("Frame extraction job started")

    # =========================
    # PATHS (LOCAL + AWS SAFE)
    # =========================
    INPUT_BASE = os.environ.get(
        "INPUT_BASE",
        "data_source/camera"
    )

    OUTPUT_BASE = os.environ.get(
        "OUTPUT_BASE",
        "data_processed/extracted_frames"
    )

    METADATA_OUTPUT = os.path.join(
        os.environ.get("OUTPUT_BASE", "data_processed"),
        "metadata"
    )

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(METADATA_OUTPUT, exist_ok=True)

    total_saved = 0
    global_metadata = []

    # =========================
    # DISCOVER CAMERA FOLDERS
    # =========================
    if not os.path.exists(INPUT_BASE):
        raise ValueError(f"Camera input path not found: {INPUT_BASE}")

    camera_dirs = [
        d for d in os.listdir(INPUT_BASE)
        if os.path.isdir(os.path.join(INPUT_BASE, d))
    ]

    logger.info(f"Detected camera folders: {camera_dirs}")

    # =========================
    # PROCESS EACH CAMERA
    # =========================
    for cam_folder in camera_dirs:

        cam_input = os.path.join(INPUT_BASE, cam_folder)
        cam_output = os.path.join(OUTPUT_BASE, cam_folder)

        logger.info(f"Processing camera folder: {cam_folder}")

        try:
            files = os.listdir(cam_input)
        except Exception as e:
            logger.error(f"Failed to read directory: {cam_input}")
            continue

        for file in files:

            if not file.endswith(".mp4"):
                continue

            video_path = os.path.join(cam_input, file)
            clip_id = file.split(".")[0]

            ts_file = file.replace(".mp4", ".timestamps.parquet")
            ts_path = os.path.join(cam_input, ts_file)

            if not os.path.exists(ts_path):
                logger.error(f"Timestamp file missing for clip: {clip_id}")
                continue

            logger.info(f"Processing clip: {clip_id}")

            try:
                # =========================
                # LOAD TIMESTAMPS
                # =========================
                ts_df = pd.read_parquet(ts_path)

                if "timestamp" not in ts_df.columns:
                    logger.error(f"Timestamp column missing: {clip_id}")
                    continue

                # =========================
                # OUTPUT STRUCTURE
                # =========================
                clip_output = os.path.join(cam_output, clip_id)
                frames_dir = os.path.join(clip_output, "frames")

                os.makedirs(frames_dir, exist_ok=True)

                metadata = []

                # =========================
                # VIDEO READ
                # =========================
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    logger.error(f"Failed to open video: {video_path}")
                    continue

                frame_id = 0
                saved_count = 0

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    if frame is None:
                        frame_id += 1
                        continue

                    # =========================
                    # TIMESTAMP ALIGNMENT
                    # =========================
                    if frame_id >= len(ts_df):
                        logger.warning(
                            f"Frame index exceeded timestamps: {clip_id}"
                        )
                        break

                    timestamp = ts_df.iloc[frame_id]["timestamp"]

                    frame_name = f"{clip_id}_frame_{frame_id:06d}.jpg"
                    frame_path = os.path.join(frames_dir, frame_name)

                    if cv2.imwrite(frame_path, frame):

                        record = {
                            "frame_id": int(frame_id),
                            "timestamp": float(timestamp),
                            "image_path": frame_path,
                            "camera": cam_folder,
                            "clip_id": clip_id
                        }

                        metadata.append(record)
                        global_metadata.append(record)

                        saved_count += 1
                        total_saved += 1

                    frame_id += 1

                cap.release()

                # =========================
                # SAVE PER-CLIP METADATA
                # =========================
                meta_df = pd.DataFrame(metadata)

                meta_path = os.path.join(clip_output, "metadata.parquet")
                meta_df.to_parquet(meta_path, index=False)

                logger.info(
                    f"{clip_id} processed | Frames saved: {saved_count}"
                )

            except Exception as e:
                logger.error(f"Error processing clip: {clip_id}")
                logger.error(str(e))
                traceback.print_exc()

    # =========================
    # GLOBAL METADATA (CRITICAL)
    # =========================
    global_meta_path = os.path.join(
        METADATA_OUTPUT,
        "frame_metadata.parquet"
    )

    pd.DataFrame(global_metadata).to_parquet(
        global_meta_path,
        index=False
    )

    logger.info(f"Global metadata saved: {global_meta_path}")
    logger.info(f"Total frames saved: {total_saved}")

# =========================
# ENTRYPOINT (SAFE)
# =========================
if __name__ == "__main__":
    main()