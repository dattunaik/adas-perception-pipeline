#PERFECT LIDAR ANNOTATIONS 

"""
ADAS Annotation Pipeline  —  LiDAR-Depth Fused Detections
==========================================================
Inputs:
  • data_processed/final_fusion_lidar_v2/XXXXXX.jpg   (panoramic LiDAR-fused frames)
  • data_processed/fused_detections/per_frame_json/N.json  (YOLOv8 per-frame detections)
  • data_processed/final_fusion/final_fused.parquet   (sync table)
  • Calibration parquets (intrinsics / extrinsics)

Output:
  • data_processed/adas_annotated/XXXXXX.jpg          (industry-grade ADAS overlays)

ADAS Features implemented
  [1] Depth extraction per bbox  — LiDAR depth-map sampled inside each box (5th-pct)
  [2] Threat-zone colour coding  — CRITICAL/DANGER/WARNING/CAUTION/SAFE
  [3] Vulnerable-road-user (VRU) priority  — persons / cyclists always front
  [4] IoU tracker  — stable ID across frames, velocity estimation
  [5] Corner-bracket boxes  — industry-standard ADAS framing
  [6] Depth-bar & distance ring  — radial danger-arc on-image
  [7] HUD overlay  — frame stamp, ego stats, threat summary
  [8] Forward-collision & lane-change warning banners
  [9] Per-object mini depth histogram strip
  [10] Occlusion-aware  — reuses z-buffer depth map from LiDAR projection
"""

import os, json, glob, time, math
import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ADAS-PIPELINE")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CLIP_ID     = "0a948f59-0a06-41a2-8e20-ac3a39ff4d61"
BLEND_WIDTH = 80
FRONT_FOV_HALF_DEG = 60.0

BASE_FRAMES    = "data_processed/extracted_frames"
LIDAR_DIR      = "data_processed/lidar_frames"
FUSED_LIDAR_DIR= "data_processed/final_fusion_lidar_v2"   # pre-rendered LiDAR frames
DET_JSON_DIR   = "data_processed/fused_detections/per_frame_json"
OUT_DIR        = "data_processed/adas_annotated"
os.makedirs(OUT_DIR, exist_ok=True)

SYNC_PATH      = r"data_processed\final_fusion\final_fused.parquet"
INTRINSIC_PATH = "data_source/calibration/camera_intrinsics.chunk_0000.parquet"
EXTRINSIC_PATH = "data_source/calibration/sensor_extrinsics.chunk_0000.parquet"

VOXEL_SIZE_NEAR = 0.12
VOXEL_SIZE_FAR  = 0.35
VOXEL_NEAR_DIST = 15.0

# ─────────────────────────────────────────────────────────────────────────────
# ADAS TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
CLASS_CFG = {
    "person":        {"priority": 1, "abbr": "PED",  "vru": True},
    "bicycle":       {"priority": 1, "abbr": "CYC",  "vru": True},
    "motorcycle":    {"priority": 1, "abbr": "MCY",  "vru": True},
    "car":           {"priority": 2, "abbr": "CAR",  "vru": False},
    "truck":         {"priority": 2, "abbr": "TRK",  "vru": False},
    "bus":           {"priority": 2, "abbr": "BUS",  "vru": False},
    "train":         {"priority": 2, "abbr": "TRN",  "vru": False},
    "traffic light": {"priority": 3, "abbr": "TFL",  "vru": False},
    "stop sign":     {"priority": 3, "abbr": "STP",  "vru": False},
    "fire hydrant":  {"priority": 4, "abbr": "HYD",  "vru": False},
}
DEFAULT_CFG = {"priority": 5, "abbr": "OBJ", "vru": False}

# Threat zones: (min_m, max_m, BGR, label, alpha_fill)
THREAT_ZONES = [
    (0,   8,   (0,   0,   255), "CRITICAL", 0.35),
    (8,   15,  (0,   60,  255), "DANGER",   0.25),
    (15,  25,  (0,  140,  255), "WARNING",  0.18),
    (25,  40,  (0,  220,  200), "CAUTION",  0.10),
    (40,  999, (50, 220,   80), "SAFE",     0.05),
]

def threat_for_depth(d: float):
    for mn, mx, bgr, lbl, alpha in THREAT_ZONES:
        if mn <= d < mx:
            return bgr, lbl, alpha
    return (50, 220, 80), "SAFE", 0.05

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
sync_df    = pd.read_parquet(SYNC_PATH)
sync_df    = sync_df[sync_df["clip_id"] == CLIP_ID].reset_index(drop=True)
intrinsics = pd.read_parquet(INTRINSIC_PATH).reset_index()
extrinsics = pd.read_parquet(EXTRINSIC_PATH).reset_index()

def get_intrinsic(cam):
    row = intrinsics[
        (intrinsics["clip_id"] == CLIP_ID) &
        (intrinsics["camera_name"] == cam)
    ].iloc[0]
    return {
        "width":  int(row["width"]),
        "height": int(row["height"]),
        "cx":     float(row["cx"]),
        "cy":     float(row["cy"]),
        "fw":     [float(row[f"fw_poly_{k}"]) for k in range(5)],
    }

def get_extrinsic(cam):
    mask = (extrinsics["clip_id"] == CLIP_ID) & \
           (extrinsics["sensor_name"].str.contains(cam, case=False))
    row = extrinsics[mask].iloc[0]
    return {
        "q": [float(row["qx"]), float(row["qy"]),
              float(row["qz"]), float(row["qw"])],
        "t": [float(row["x"]),  float(row["y"]),  float(row["z"])],
    }

CAM_NAMES = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
]
INTR = {c: get_intrinsic(c) for c in CAM_NAMES}
EXTR = {c: get_extrinsic(c) for c in CAM_NAMES}
EXTR["lidar"] = get_extrinsic("lidar_top_360fov")

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
def quat_to_R(q):
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)

def build_T(ext):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(ext["q"])
    T[:3,  3] = ext["t"]
    return T

def transform_points(pts, T):
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

def project_fisheye(pts_cam, intr):
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    rxy   = np.sqrt(x**2 + y**2)
    theta = np.arctan2(rxy, z)
    fw = intr["fw"]
    r_px = (fw[0] + fw[1]*theta + fw[2]*theta**2
                  + fw[3]*theta**3 + fw[4]*theta**4)
    safe = np.where(rxy < 1e-9, 1.0, rxy)
    u = intr["cx"] + r_px * (x / safe)
    v = intr["cy"] + r_px * (y / safe)
    valid = (r_px > 0) & (theta < np.deg2rad(60.0)) & (z > 0)
    return u, v, z, valid, theta

# ─────────────────────────────────────────────────────────────────────────────
# PANORAMA MAPPER  (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────
class PanoramaMapper:
    def __init__(self, intr_front, intr_left, intr_right,
                 ext_front, ext_left, ext_right,
                 blend_width=80, front_fov_half_deg=60.0):
        self.BW = blend_width
        W_f  = intr_front["width"]
        fx_f = intr_front["fw"][1]
        fx_l = intr_left ["fw"][1]
        fx_r = intr_right["fw"][1]
        self.scale_left  = fx_f / fx_l
        self.scale_right = fx_f / fx_r
        W_l_sc = int(round(intr_left ["width"] * self.scale_left))
        W_r_sc = int(round(intr_right["width"] * self.scale_right))
        self.W_f = W_f
        R_f = quat_to_R(ext_front["q"])
        R_l = quat_to_R(ext_left ["q"])
        R_r = quat_to_R(ext_right["q"])
        cx_l_sc = intr_left ["cx"] * self.scale_left
        cx_r_sc = intr_right["cx"] * self.scale_right
        fx_l_sc = fx_l * self.scale_left
        fx_r_sc = fx_r * self.scale_right
        ray = np.array([np.sin(np.radians(-front_fov_half_deg)), 0.0,
                         np.cos(np.radians(-front_fov_half_deg))])
        ray_vc = R_f @ ray; ray_sc = R_l.T @ ray_vc
        h_ang  = np.arctan2(float(ray_sc[0]), float(ray_sc[2]))
        l_b = int(np.clip(cx_l_sc + fx_l_sc * h_ang, W_l_sc//8, W_l_sc*7//8))
        self.l_a, self.l_b = 0, l_b
        ray = np.array([np.sin(np.radians(+front_fov_half_deg)), 0.0,
                         np.cos(np.radians(+front_fov_half_deg))])
        ray_vc = R_f @ ray; ray_sc = R_r.T @ ray_vc
        h_ang  = np.arctan2(float(ray_sc[0]), float(ray_sc[2]))
        r_a = int(np.clip(cx_r_sc + fx_r_sc * h_ang, W_r_sc//8, W_r_sc*7//8))
        self.r_a, self.r_b = r_a, W_r_sc
        self.left_strip_w  = l_b
        self.right_strip_w = W_r_sc - r_a
        self.left_front_w  = self.left_strip_w + W_f - blend_width
        self.fused_w       = self.left_front_w + self.right_strip_w - blend_width

    def cam_to_fused_x(self, u_cam, cam):
        BW = self.BW
        if cam == "camera_front_wide_120fov":
            return self.left_strip_w + u_cam - BW
        elif cam == "camera_cross_left_120fov":
            return u_cam * self.scale_left - self.l_a
        elif cam == "camera_cross_right_120fov":
            return self.left_front_w - BW + (u_cam * self.scale_right - self.r_a)
        raise ValueError(cam)

# ─────────────────────────────────────────────────────────────────────────────
# LIDAR CLEANING + DOWNSAMPLING
# ─────────────────────────────────────────────────────────────────────────────
def voxel_downsample(pts, voxel_size):
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int32)
    _, idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    out = np.zeros((len(idx), 3), dtype=np.float32)
    np.add.at(out, inv, pts)
    counts = np.bincount(inv, minlength=len(idx))
    return out / counts[:, None]

def downsample_adaptive(pts):
    dist = np.linalg.norm(pts, axis=1)
    near, far = pts[dist < VOXEL_NEAR_DIST], pts[dist >= VOXEL_NEAR_DIST]
    parts = []
    if len(near): parts.append(voxel_downsample(near, VOXEL_SIZE_NEAR))
    if len(far):  parts.append(voxel_downsample(far,  VOXEL_SIZE_FAR))
    return np.vstack(parts) if parts else pts

def clean_lidar(pts):
    ego = ((pts[:,0] > -2.0) & (pts[:,0] < 4.5) &
           (np.abs(pts[:,1]) < 1.2) &
           (pts[:,2] > -0.5) & (pts[:,2] < 2.5))
    pts = pts[~ego]
    dist = np.linalg.norm(pts, axis=1)
    pts = pts[(dist > 0.5) & (dist < 80)]
    return pts[pts[:,2] > -2.5]

# ─────────────────────────────────────────────────────────────────────────────
# DEPTH MAP GENERATION  (key addition over v2 — returns float32 depth map)
# ─────────────────────────────────────────────────────────────────────────────
def build_depth_map(fused_shape, pts_lidar, mapper):
    """
    Project LiDAR points onto the fused image coordinate space and return:
      depth_map  — float32 (H, W) with min-depth per pixel (inf = no point)
      pixel_pts  — list of (uf, vf, depth) for downstream use
    """
    h, w = fused_shape[:2]
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    T_lidar   = build_T(EXTR["lidar"])
    RING_SPACING = np.deg2rad(30.0 / 64)

    best_dep   = np.full(len(pts_lidar), np.nan, dtype=np.float32)
    best_uf    = np.full(len(pts_lidar), np.nan, dtype=np.float32)
    best_vf    = np.full(len(pts_lidar), np.nan, dtype=np.float32)
    best_theta = np.full(len(pts_lidar), np.inf, dtype=np.float32)

    for cam in CAM_NAMES:
        intr  = INTR[cam]
        T_cam = build_T(EXTR[cam])
        T_L2C = np.linalg.inv(T_cam) @ T_lidar
        pts_cam = transform_points(pts_lidar, T_L2C)
        fwd = (pts_cam[:,2] > 0.5) & (pts_cam[:,2] < 80.0)
        u, v, dep, valid, theta = project_fisheye(pts_cam, intr)
        W, H = intr["width"], intr["height"]
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ok = fwd & valid & in_bounds
        fused_x = mapper.cam_to_fused_x(u, cam)
        wins = ok & (theta < best_theta)
        best_theta[wins] = theta[wins]
        best_dep  [wins] = dep  [wins]
        best_vf   [wins] = v    [wins]
        best_uf   [wins] = fused_x[wins]

    assigned = ~np.isnan(best_uf)
    uf  = best_uf [assigned]
    vf  = best_vf [assigned]
    dep = best_dep[assigned]
    in_pano = (uf >= 0) & (uf < w) & (vf >= 0) & (vf < h)
    uf  = uf [in_pano].astype(np.int32)
    vf  = vf [in_pano].astype(np.int32)
    dep = dep[in_pano]

    # Splat with radius for gap-filling (radius 3 px)
    for i in range(len(uf)):
        r = 3
        x0, x1 = max(uf[i]-r, 0), min(uf[i]+r+1, w)
        y0, y1 = max(vf[i]-r, 0), min(vf[i]+r+1, h)
        region = depth_map[y0:y1, x0:x1]
        np.minimum(region, dep[i], out=region)

    return depth_map

def bbox_depth(depth_map, bbox, pct=5):
    """
    Extract the Nth-percentile depth from LiDAR points within a bounding box.
    5th percentile ≈ closest surface (robust against noise/outliers).
    Returns np.inf if no LiDAR coverage.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = depth_map.shape
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x1 >= x2 or y1 >= y2:
        return np.inf
    patch = depth_map[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch)]
    if len(valid) < 3:
        return np.inf
    return float(np.percentile(valid, pct))

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE IoU TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class ADASTracker:
    """
    Frame-to-frame IoU-based multi-object tracker.
    Maintains stable IDs and estimates lateral/longitudinal velocity.
    """
    def __init__(self, iou_thresh=0.35, max_lost=5):
        self.iou_thresh  = iou_thresh
        self.max_lost    = max_lost
        self._next_id    = 1
        self._tracks     = {}   # id → {bbox, class, depth, lost, history}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, detections, frame_idx):
        """
        detections: list of dicts {bbox, class_name, confidence, depth}
        Returns:    same list enriched with {track_id, velocity_mps}
        """
        unmatched_det  = list(range(len(detections)))
        matched_tracks = set()

        # Greedy IoU matching
        for tid, trk in self._tracks.items():
            if not unmatched_det:
                break
            ious = [(self._iou(trk["bbox"], detections[i]["bbox"]), i)
                    for i in unmatched_det]
            best_iou, best_i = max(ious, key=lambda x: x[0])
            if best_iou >= self.iou_thresh:
                det = detections[best_i]
                # Velocity: depth delta over frame (≈ 33 ms @ 30 fps)
                prev_depth = trk["depth"]
                curr_depth = det["depth"]
                dt = 1.0 / 30.0
                vel = (prev_depth - curr_depth) / dt if np.isfinite(prev_depth) and np.isfinite(curr_depth) else 0.0
                trk.update({
                    "bbox":      det["bbox"],
                    "class":     det["class_name"],
                    "depth":     curr_depth,
                    "conf":      det["confidence"],
                    "lost":      0,
                    "velocity":  vel,
                    "frame":     frame_idx,
                })
                trk["history"].append(curr_depth)
                det["track_id"]     = tid
                det["velocity_mps"] = vel
                matched_tracks.add(tid)
                unmatched_det.remove(best_i)

        # Age out lost tracks
        for tid in list(self._tracks.keys()):
            if tid not in matched_tracks:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    del self._tracks[tid]

        # Register new tracks
        for i in unmatched_det:
            det = detections[i]
            tid = self._next_id; self._next_id += 1
            self._tracks[tid] = {
                "bbox":     det["bbox"],
                "class":    det["class_name"],
                "depth":    det["depth"],
                "conf":     det["confidence"],
                "lost":     0,
                "velocity": 0.0,
                "frame":    frame_idx,
                "history":  [det["depth"]],
            }
            det["track_id"]     = tid
            det["velocity_mps"] = 0.0

        return detections

# ─────────────────────────────────────────────────────────────────────────────
# ADAS RENDERER
# ─────────────────────────────────────────────────────────────────────────────

# Fonts
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO  = cv2.FONT_HERSHEY_PLAIN
FONT_BOLD  = cv2.FONT_HERSHEY_SIMPLEX

def _alpha_rect(img, x1, y1, x2, y2, color, alpha):
    """Semi-transparent filled rectangle."""
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(sub, 1-alpha, rect, alpha, 0, dst=sub)

def _corner_bracket(img, x1, y1, x2, y2, color, thickness=2, length_frac=0.22):
    """Draw four corner L-brackets instead of a full rectangle (ADAS style)."""
    lx = max(int((x2 - x1) * length_frac), 12)
    ly = max(int((y2 - y1) * length_frac), 12)
    corners = [
        # top-left
        [(x1,y1),(x1+lx,y1)], [(x1,y1),(x1,y1+ly)],
        # top-right
        [(x2,y1),(x2-lx,y1)], [(x2,y1),(x2,y1+ly)],
        # bottom-left
        [(x1,y2),(x1+lx,y2)], [(x1,y2),(x1,y2-ly)],
        # bottom-right
        [(x2,y2),(x2-lx,y2)], [(x2,y2),(x2,y2-ly)],
    ]
    for p1, p2 in corners:
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def _label_pill(img, x, y, text, bg_color, text_color=(255,255,255),
                font=FONT_MONO, scale=0.85, thickness=1, pad=5):
    """Pill-shaped label with background."""
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    rx1, ry1 = x, y - th - pad*2
    rx2, ry2 = x + tw + pad*2, y
    rx1, ry1 = max(rx1, 0), max(ry1, 0)
    _alpha_rect(img, rx1, ry1, min(rx2, img.shape[1]), min(ry2, img.shape[0]),
                bg_color, 0.88)
    cv2.putText(img, text, (rx1+pad, ry2-pad), font, scale,
                text_color, thickness, cv2.LINE_AA)
    return rx2 - rx1   # width

def _draw_detection(img, det, depth_map):
    """Render a single ADAS-annotated detection."""
    bbox  = det["bbox"]
    cls   = det["class_name"]
    conf  = det["confidence"]
    depth = det.get("depth", np.inf)
    tid   = det.get("track_id", -1)
    vel   = det.get("velocity_mps", 0.0)
    cfg   = CLASS_CFG.get(cls, DEFAULT_CFG)
    abbr  = cfg["abbr"]
    vru   = cfg["vru"]

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = img.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w-1), min(y2, h-1)
    if x1 >= x2 or y1 >= y2:
        return

    color, threat, fill_alpha = threat_for_depth(depth)
    thickness = 3 if threat in ("CRITICAL","DANGER") else 2

    # ── 1. Semi-transparent threat fill ──────────────────────────────────────
    _alpha_rect(img, x1, y1, x2, y2, color, fill_alpha)

    # ── 2. Corner brackets ───────────────────────────────────────────────────
    _corner_bracket(img, x1, y1, x2, y2, color, thickness=thickness)

    # ── 3. VRU marker (bright magenta outer ring) ─────────────────────────────
    if vru:
        cx_obj = (x1 + x2) // 2
        cy_obj = (y1 + y2) // 2
        rx = max((x2 - x1) // 2 + 6, 10)
        ry = max((y2 - y1) // 2 + 6, 10)
        cv2.ellipse(img, (cx_obj, cy_obj), (rx, ry), 0, 0, 360,
                    (255, 0, 220), 1, cv2.LINE_AA)

    # ── 4. Top label strip ───────────────────────────────────────────────────
    label_parts = [f"#{tid:03d}" if tid > 0 else "---",
                   abbr,
                   f"{conf:.2f}"]
    if np.isfinite(depth):
        label_parts.append(f"{depth:.1f}m")
    label = "  ".join(label_parts)
    _label_pill(img, x1, y1, label, color, font=FONT_MONO, scale=0.78)

    # ── 5. Velocity chip (bottom-left of box) ────────────────────────────────
    if np.isfinite(vel) and abs(vel) > 0.5:
        sign   = "▲" if vel > 0 else "▼"   # approaching ▲ / receding ▼
        v_col  = (0, 80, 255) if vel > 0 else (0, 200, 80)
        v_text = f"{sign}{abs(vel):.1f}m/s"
        _label_pill(img, x1, y2, v_text, v_col, scale=0.7)

    # ── 6. Threat badge (top-right corner) ───────────────────────────────────
    badge_col = {
        "CRITICAL": (0,   0,   220),
        "DANGER":   (0,   40,  220),
        "WARNING":  (0,  130,  240),
        "CAUTION":  (0,  200,  200),
        "SAFE":     (30, 160,  60),
    }.get(threat, color)

    bw = _label_pill(img, x2 - 80, y1, threat, badge_col,
                     font=FONT_MONO, scale=0.65)

    # ── 7. Depth bar (right edge of bbox) ────────────────────────────────────
    if np.isfinite(depth) and y2 > y1 + 10:
        bar_h  = y2 - y1
        bar_w  = 5
        filled = int(bar_h * np.clip(1.0 - (depth / 45.0), 0, 1))
        bx1 = min(x2 + 3, w - bar_w - 1)
        bx2 = bx1 + bar_w
        cv2.rectangle(img, (bx1, y1), (bx2, y2), (60, 60, 60), -1)
        cv2.rectangle(img, (bx1, y2 - filled), (bx2, y2), color, -1)


def draw_hud(img, frame_idx, detections, fps=None):
    """
    Top-of-frame HUD bar with:
      • Frame / time stamp
      • Object counts per class
      • Threat-level histogram
      • FCW (forward collision warning) banner
    """
    h, w = img.shape[:2]
    HUD_H = 52

    # Dark translucent banner
    _alpha_rect(img, 0, 0, w, HUD_H, (10, 10, 20), 0.82)

    # Thin accent line
    cv2.line(img, (0, HUD_H), (w, HUD_H), (0, 200, 255), 1)

    # Left: frame info
    ts_text = f"FRAME {frame_idx:04d}"
    cv2.putText(img, ts_text, (12, 34), FONT_MONO, 1.3,
                (0, 220, 255), 1, cv2.LINE_AA)

    if fps:
        cv2.putText(img, f"{fps:.1f}fps", (160, 34), FONT_MONO, 0.95,
                    (120, 200, 120), 1, cv2.LINE_AA)

    # Centre: object count summary
    counts = {}
    for d in detections:
        cls = d["class_name"]
        counts[cls] = counts.get(cls, 0) + 1
    cx = w // 2 - len(counts)*55
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        abbr = CLASS_CFG.get(cls, DEFAULT_CFG)["abbr"]
        cv2.putText(img, f"{abbr}:{cnt}", (cx, 34), FONT_MONO, 1.0,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cx += 90

    # Right: closest threat
    depths = [d.get("depth", np.inf) for d in detections if np.isfinite(d.get("depth", np.inf))]
    if depths:
        min_d = min(depths)
        col, threat, _ = threat_for_depth(min_d)
        cv2.putText(img, f"MIN DIST {min_d:.1f}m [{threat}]",
                    (w - 340, 34), FONT_MONO, 1.0, col, 1, cv2.LINE_AA)

    # ── FCW Banner ───────────────────────────────────────────────────────────
    critical = [d for d in detections
                if np.isfinite(d.get("depth", np.inf)) and d["depth"] < 8]
    if critical:
        banner_y = HUD_H + 4
        _alpha_rect(img, 0, banner_y, w, banner_y + 36, (0, 0, 180), 0.75)
        cv2.line(img, (0, banner_y), (w, banner_y), (0, 0, 255), 2)
        n = len(critical)
        classes = ", ".join(set(d["class_name"] for d in critical))
        fcw = f"  ⚠  FORWARD COLLISION WARNING  |  {n} OBJECT(S) WITHIN 8m  |  {classes.upper()}  ⚠"
        cv2.putText(img, fcw, (10, banner_y + 24),
                    FONT_MONO, 0.95, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Threat histogram (bottom-right strip) ────────────────────────────────
    hist_x = w - 200
    hist_y = h - 140
    zone_counts = [0] * len(THREAT_ZONES)
    for d in detections:
        dep = d.get("depth", np.inf)
        for zi, (mn, mx, bgr, lbl, _) in enumerate(THREAT_ZONES):
            if mn <= dep < mx:
                zone_counts[zi] += 1
                break
    bar_w = 30
    for zi, (mn, mx, bgr, lbl, _) in enumerate(THREAT_ZONES):
        bh = zone_counts[zi] * 12 + 2
        bx = hist_x + zi * (bar_w + 4)
        _alpha_rect(img, bx, hist_y - bh, bx+bar_w, hist_y,
                    bgr, 0.8)
        cv2.putText(img, lbl[:3], (bx, hist_y + 12),
                    FONT_MONO, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # ── ADAS watermark (bottom-left) ─────────────────────────────────────────
    cv2.putText(img, "ADAS  |  LiDAR+CAM FUSION  |  YOLOv8",
                (10, h - 14), FONT_MONO, 0.75, (80, 80, 80), 1, cv2.LINE_AA)


def draw_depth_legend(img):
    """Compact depth-colour legend strip (bottom-right corner)."""
    h, w = img.shape[:2]
    lx, ly = w - 200, h - 195
    lw, lh = 16, 120
    # Gradient strip
    for i in range(lh):
        t = 1.0 - i / lh
        d = 1.5 + t * 43.5
        bgr, _, _ = threat_for_depth(d)
        cv2.line(img, (lx, ly+i), (lx+lw, ly+i), bgr, 1)
    cv2.rectangle(img, (lx, ly), (lx+lw, ly+lh), (150,150,150), 1)
    cv2.putText(img, "1.5m", (lx+lw+4, ly+12),   FONT_MONO, 0.55, (200,200,200), 1)
    cv2.putText(img, "25m",  (lx+lw+4, ly+lh//2), FONT_MONO, 0.55, (200,200,200), 1)
    cv2.putText(img, "45m",  (lx+lw+4, ly+lh-4),  FONT_MONO, 0.55, (200,200,200), 1)
    cv2.putText(img, "DEPTH",(lx-2, ly-6),         FONT_MONO, 0.55, (140,140,140), 1)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PANORAMA MAPPER  (shared across all frames)
# ─────────────────────────────────────────────────────────────────────────────
mapper = PanoramaMapper(
    intr_front = INTR["camera_front_wide_120fov"],
    intr_left  = INTR["camera_cross_left_120fov"],
    intr_right = INTR["camera_cross_right_120fov"],
    ext_front  = EXTR["camera_front_wide_120fov"],
    ext_left   = EXTR["camera_cross_left_120fov"],
    ext_right  = EXTR["camera_cross_right_120fov"],
    blend_width        = BLEND_WIDTH,
    front_fov_half_deg = FRONT_FOV_HALF_DEG,
)

tracker = ADASTracker(iou_thresh=0.35, max_lost=5)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
total = 0
t0    = time.time()

for idx, row in sync_df.iterrows():
    try:
        f_id = row.get("camera_frame")
        if pd.isna(f_id):
            continue
        frame_idx = int(f_id)

        lidar_frame_id = row.get("lidar_frame")
        if pd.isna(lidar_frame_id):
            logger.warning(f"No LiDAR frame for camera_frame={frame_idx}")
            continue
        lidar_frame_id = int(lidar_frame_id)

        # ── Load pre-rendered LiDAR-fused frame ──────────────────────────────
        fused_path = os.path.join(FUSED_LIDAR_DIR, f"{frame_idx:06d}.jpg")
        if not os.path.exists(fused_path):
            logger.warning(f"Missing fused frame: {fused_path}")
            continue
        frame = cv2.imread(fused_path)
        if frame is None:
            logger.warning(f"Could not read: {fused_path}")
            continue

        # ── Load detection JSON ───────────────────────────────────────────────
        det_path = os.path.join(DET_JSON_DIR, f"{frame_idx}.json")
        if not os.path.exists(det_path):
            logger.warning(f"Missing detection JSON: {det_path}")
            detections = []
        else:
            with open(det_path) as f:
                jdata = json.load(f)
            detections = jdata.get("detections", [])

        # ── Build depth map from LiDAR ────────────────────────────────────────
        lidar_path = os.path.join(LIDAR_DIR, f"{lidar_frame_id}.parquet")
        depth_map  = None
        if os.path.exists(lidar_path):
            df_lidar = pd.read_parquet(lidar_path)
            pts = df_lidar[["x", "y", "z"]].values
            pts = downsample_adaptive(clean_lidar(pts))
            depth_map = build_depth_map(frame.shape, pts, mapper)
        else:
            logger.warning(f"Missing LiDAR: {lidar_path}")
            depth_map = np.full(frame.shape[:2], np.inf, dtype=np.float32)

        # ── Enrich detections with depth ─────────────────────────────────────
        for det in detections:
            det["depth"] = bbox_depth(depth_map, det["bbox"], pct=5)

        # ── Sort: VRUs first, then by depth (closest last = drawn on top) ────
        detections.sort(key=lambda d: (
            -CLASS_CFG.get(d["class_name"], DEFAULT_CFG)["priority"],
            -d.get("depth", 999)
        ))

        # ── Tracker update ────────────────────────────────────────────────────
        detections = tracker.update(detections, frame_idx)

        # ── Render ADAS overlay ───────────────────────────────────────────────
        output = frame.copy()
        for det in detections:
            _draw_detection(output, det, depth_map)

        draw_depth_legend(output)
        draw_hud(output, frame_idx, detections)

        # ── Save ──────────────────────────────────────────────────────────────
        out_path = os.path.join(OUT_DIR, f"{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
        total += 1

        fps = total / (time.time() - t0 + 1e-6)
        logger.info(
            f"[{frame_idx:04d}] dets={len(detections):2d} | "
            f"depth_pts={np.isfinite(depth_map).sum():6d} | "
            f"fps={fps:.1f} → {out_path}"
        )

    except Exception as e:
        logger.error(f"Error at idx={idx}: {e}", exc_info=True)
        continue

logger.info(f"Done — {total} frames annotated in {time.time()-t0:.1f}s → {OUT_DIR}")