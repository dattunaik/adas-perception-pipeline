# ADAS LiDAR–Camera Fusion Pipeline (v1)

## 1. Overview

This repository implements a complete perception pipeline for ADAS use cases.
The pipeline processes multi-camera video and LiDAR data to generate:

* Synchronized multi-sensor datasets
* Fused panoramic images
* Object detections (YOLOv8)
* Depth-aware annotations using LiDAR
* Tracking and ADAS visualization outputs

The system is designed for **offline processing with production-grade structure**, modular scripts, and reproducible outputs.

---

## 2. Dataset Structure

### 2.1 Raw Data (`data_source/`)

```
data_source/
    calibration/
        camera_intrinsics.chunk_0000.parquet
        sensor_extrinsics.chunk_0000.parquet
        vehicle_dimensions.chunk_0000.parquet

    camera/
        camera_cross_left_120fov.chunk_0000/
        camera_cross_right_120fov.chunk_0000/
        camera_front_wide_120fov.chunk_0000/

    lidar/
        lidar_top_360fov.chunk_0000/

    labels/
        egomotion.chunk_0000/

    metadata/
        data_collection.parquet
        sensor_presence.parquet
```

---

### 2.2 Processed Data (`data_processed/`)

```
data_processed/
    extracted_frames/
    sync/
    sensor_sync/
    final_fusion/
    fused_detections/
    adas_annotated/
```

---

## 3. Pipeline Summary

| Stage | Name               | Output                 | Frames |
| ----- | ------------------ | ---------------------- | ------ |
| 1     | Frame Extraction   | Extracted frames       | 1815   |
| 2     | Camera Sync        | camera_synced.parquet  | 605    |
| 3     | Sensor Sync        | sensor_synced.parquet  | 605    |
| 4     | Final Fusion Table | final_fused.parquet    | 605    |
| 5     | Image Fusion       | fused frames           | 605    |
| 6     | Detection          | detection JSON/Parquet | 605    |
| 7     | ADAS Annotation    | annotated frames       | 605    |

---

## 4. Script-Level Documentation

---

## Script 01: Frame Extraction

### Input

| Type       | Path                         |
| ---------- | ---------------------------- |
| Video      | `data_source/camera/*/*.mp4` |
| Timestamps | `*.timestamps.parquet`       |

### Process

* Read video using OpenCV
* Extract frames sequentially
* Align frames with timestamps
* Generate metadata (frame_id, timestamp)

### Output

| Type     | Path                                               |
| -------- | -------------------------------------------------- |
| Frames   | `data_processed/extracted_frames/.../frames/*.jpg` |
| Metadata | `metadata.parquet`                                 |

### Limitations

* Assumes constant FPS
* Frame drops not handled explicitly
* No motion blur correction

---

## Script 02: Multi-Camera Synchronization

### Input

| Source           | Data                           |
| ---------------- | ------------------------------ |
| Extracted frames | metadata.parquet (all cameras) |

### Process

* Convert timestamps to seconds
* Use front camera as reference
* Find nearest frames for left/right cameras
* Apply thresholds:

  * Good: ≤33 ms
  * Weak: ≤100 ms
  * Fallback: reuse frame

### Output

| File                    |
| ----------------------- |
| `camera_synced.parquet` |

### Limitations

* No interpolation between frames
* Drift not corrected over long sequences

---

## Script 03: Camera–LiDAR Synchronization

### Input

| Source      | Data                    |
| ----------- | ----------------------- |
| Camera sync | `camera_synced.parquet` |
| LiDAR       | `.parquet`              |

### Process

* Normalize timestamps
* Match nearest LiDAR frame for each camera frame
* Apply fallback for missing matches

### Output

| File                            |
| ------------------------------- |
| `sensor_synced_cam_ref.parquet` |

### Limitations

* Nearest-neighbor matching only
* No temporal interpolation

---

## Script 04: Final Fusion Metadata

### Input

| Source      | File                    |
| ----------- | ----------------------- |
| Camera sync | `camera_synced.parquet` |
| Sensor sync | `sensor_synced.parquet` |

### Process

* Merge datasets on frame_id
* Compute timestamp differences
* Build unified mapping

### Output

| File                  |
| --------------------- |
| `final_fused.parquet` |

### Limitations

* Assumes valid sync from previous stages

---

## Script 05: Image Fusion (Panorama)

### Input

| Source      | Data                    |
| ----------- | ----------------------- |
| Frames      | Extracted images        |
| Calibration | Intrinsics + extrinsics |

### Process

* Load camera calibration
* Convert quaternion to rotation matrix
* Align camera views geometrically
* Compute overlap regions
* Apply:

  * Color correction (LAB)
  * Seam blending
  * Vertical alignment
  * Temporal smoothing

### Output

| Type         | Path                 |
| ------------ | -------------------- |
| Fused images | `final_fusion/*.jpg` |

### Limitations

* Sensitive to calibration errors
* Ghosting in dynamic scenes
* No depth-aware blending

---

## Script 06: YOLO Detection

### Input

| Source       | Data                 |
| ------------ | -------------------- |
| Fused images | `final_fusion/*.jpg` |

### Process

* Image enhancement (CLAHE)
* YOLOv8 inference
* Class filtering (ADAS relevant)
* Save detections

### Output

| Type    | Path       |
| ------- | ---------- |
| JSON    | per-frame  |
| Parquet | aggregated |

### Limitations

* No tracking at this stage
* Performance depends on model quality

---

## Script 11: LiDAR Decoding & Cleaning

### Input

| Source | Data       |
| ------ | ---------- |
| LiDAR  | `.parquet` |

### Process

* Decode Draco blobs
* Remove ego vehicle points
* Remove ground points
* Filter by distance and height
* Apply KNN-based noise filtering
* Adaptive voxel downsampling

### Output

| Type                      |
| ------------------------- |
| Cleaned LiDAR point cloud |

### Limitations

* Fixed thresholds
* May remove valid low-height objects

---

## Script 12: LiDAR Projection & Fusion

### Input

| Source      | Data                  |
| ----------- | --------------------- |
| LiDAR       | Cleaned points        |
| Calibration | Intrinsics/extrinsics |
| Images      | Camera frames         |

### Process

* Transform LiDAR → camera frame
* Project using fisheye model
* Select best camera using angle
* Map to panorama space
* Apply Z-buffer for occlusion
* Colorize by depth

### Output

| Type                        |
| --------------------------- |
| LiDAR projected fused image |

### Limitations

* Sparse LiDAR → incomplete coverage
* Projection errors near edges

---

## Script 13: ADAS Annotation (Final Output)

### Input

| Source       | Data                |
| ------------ | ------------------- |
| Fused images | Panorama            |
| Detections   | YOLO JSON           |
| LiDAR        | Depth map           |
| Sync         | final_fused.parquet |

---

### Process

#### Depth Estimation

* Build depth map from LiDAR
* Use minimum depth per pixel
* Extract object depth using percentile (5%)

#### Tracking

* IoU-based tracking
* Assign track IDs
* Estimate velocity

#### Classification

* Assign priority (VRU vs vehicle)
* Map to ADAS categories

#### Visualization

* Corner-based bounding boxes
* Depth bars
* Threat zones
* HUD overlay

#### Threat Logic

| Distance | Level    |
| -------- | -------- |
| < 8m     | CRITICAL |
| 8–15m    | DANGER   |
| 15–25m   | WARNING  |
| 25–40m   | CAUTION  |
| > 40m    | SAFE     |

---

### Output

| Type             | Path                   |
| ---------------- | ---------------------- |
| Annotated frames | `adas_annotated/*.jpg` |

---

### Limitations

* IoU tracker is basic (no re-ID)
* Velocity estimation is noisy
* Depth missing in sparse regions
* No lane detection

---

## 5. Key Features

* Multi-camera panoramic fusion
* LiDAR depth integration
* Object detection + tracking
* Threat-level classification
* ADAS-style visualization
* Occlusion-aware rendering

---

## 6. Configuration Parameters

| Parameter          | Description           |
| ------------------ | --------------------- |
| VOXEL_SIZE_NEAR    | LiDAR resolution near |
| VOXEL_SIZE_FAR     | LiDAR resolution far  |
| FRONT_FOV_HALF_DEG | Camera FOV            |
| THREAT_ZONES       | Distance thresholds   |

---

## 7. Execution

```
python frame_extraction.py
python camera_sync.py
python sensor_sync.py
python fusion.py
python detection.py
python adas_annotation.py
```

---

## 8. Known Limitations

* Sparse LiDAR affects depth quality
* No advanced tracking (DeepSORT not used)
* No temporal smoothing in detections
* Limited ground truth dataset
* No lane or free-space detection

---

## 9. Future Improvements (v2)

* Advanced tracking (DeepSORT / ByteTrack)
* Lane detection integration
* Real-time pipeline
* Sensor fusion with radar
* Improved depth completion
* Robust QA dashboard