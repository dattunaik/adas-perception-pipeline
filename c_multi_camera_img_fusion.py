import os
import cv2
import numpy as np
import pandas as pd
import logging
import json

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ADAS-FUSION-V3")

# ============================================================
# CONFIG
# ============================================================
CLIP_ID = "0a948f59-0a06-41a2-8e20-ac3a39ff4d61"

BASE_FRAMES = "data_processed/extracted_frames"
SYNC_PATH   = "data_processed/sync/camera_synced.parquet"

INTRINSIC_PATH = "data_source/calibration/camera_intrinsics.chunk_0000.parquet"
EXTRINSIC_PATH = "data_source/calibration/sensor_extrinsics.chunk_0000.parquet"

OUT_DIR   = "data_processed/final_fusion_v3"
DEBUG_DIR = "data_processed/fusion_debug_v3"

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

FRONT_DIR = f"{BASE_FRAMES}/camera_front_wide_120fov.chunk_0000/{CLIP_ID}/frames"
LEFT_DIR  = f"{BASE_FRAMES}/camera_cross_left_120fov.chunk_0000/{CLIP_ID}/frames"
RIGHT_DIR = f"{BASE_FRAMES}/camera_cross_right_120fov.chunk_0000/{CLIP_ID}/frames"

FRONT_FOV_HALF_DEG = 60.0   # half of 120 deg front camera FOV
BLEND_WIDTH        = 80     # crossfade width in pixels

# ============================================================
# LOAD DATA
# ============================================================
sync_df = pd.read_parquet(SYNC_PATH)
sync_df = sync_df[sync_df["clip_id"] == CLIP_ID].reset_index(drop=True)

intrinsics = pd.read_parquet(INTRINSIC_PATH).reset_index()
extrinsics = pd.read_parquet(EXTRINSIC_PATH).reset_index()

logger.info(f"Total frames: {len(sync_df)}")

# ============================================================
# CALIBRATION
# ============================================================
def get_intrinsic(cam):
    row = intrinsics[
        (intrinsics["clip_id"] == CLIP_ID) &
        (intrinsics["camera_name"] == cam)
    ].iloc[0]
    return {
        "fx": float(row["fw_poly_1"]),
        "fy": float(row["fw_poly_1"]),
        "cx": float(row["cx"]),
        "cy": float(row["cy"])
    }

def get_extrinsic(cam):
    row = extrinsics[
        (extrinsics["clip_id"] == CLIP_ID) &
        (extrinsics["sensor_name"] == cam)
    ].iloc[0]
    return {
        "qx": float(row["qx"]),
        "qy": float(row["qy"]),
        "qz": float(row["qz"]),
        "qw": float(row["qw"]),
        "x":  float(row["x"]),
        "y":  float(row["y"]),
        "z":  float(row["z"])
    }

K_FRONT = get_intrinsic("camera_front_wide_120fov")
K_LEFT  = get_intrinsic("camera_cross_left_120fov")
K_RIGHT = get_intrinsic("camera_cross_right_120fov")

EXT_FRONT = get_extrinsic("camera_front_wide_120fov")
EXT_LEFT  = get_extrinsic("camera_cross_left_120fov")
EXT_RIGHT = get_extrinsic("camera_cross_right_120fov")

# ============================================================
# GEOMETRY HELPERS
# ============================================================
def quat_to_R(ext):
    """
    Build 3x3 rotation matrix from quaternion in extrinsic dict.
    R transforms a point from camera frame to vehicle frame:
        p_vehicle = R @ p_camera
    """
    qx, qy, qz, qw = ext["qx"], ext["qy"], ext["qz"], ext["qw"]
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def non_overlap_cols(side_ext, side_intr, front_ext, img_w, side):
    """
    Returns (col_a, col_b): the column slice of the side camera whose
    angular content does NOT overlap with the front camera's 120 deg FOV.

    Steps
    -----
    1. Build the front camera edge ray at +/-60 deg in front-cam frame.
       Camera axes: x=right, y=down, z=forward.
       Horizontal edge ray: [sin(angle), 0, cos(angle)]
    2. Rotate into vehicle frame (R_front), then into side-cam frame (R_side.T).
    3. Map horizontal angle to pixel: col = cx + fx * atan2(x, z)
       (fisheye linear model — valid for moderate angles).
    4. Non-overlapping strip is on the OUTSIDE of that column.

    Falls back to w//4 if geometry fails.
    """
    try:
        R_front = quat_to_R(front_ext)
        R_side  = quat_to_R(side_ext)

        angle_fc = np.radians(-FRONT_FOV_HALF_DEG if side == "left"
                               else  FRONT_FOV_HALF_DEG)
        ray_fc = np.array([np.sin(angle_fc), 0.0, np.cos(angle_fc)])

        ray_vc = R_front @ ray_fc      # camera frame -> vehicle frame
        ray_sc = R_side.T @ ray_vc    # vehicle frame -> side-cam frame

        h_angle = np.arctan2(float(ray_sc[0]), float(ray_sc[2]))
        col = int(side_intr["cx"] + side_intr["fx"] * h_angle)
        col = int(np.clip(col, img_w // 8, img_w * 7 // 8))

        logger.info(f"  [{side}] non-overlap boundary col={col}/{img_w} "
                    f"(h_angle={np.degrees(h_angle):.1f} deg)")

        return (0, col) if side == "left" else (col, img_w)

    except Exception as e:
        logger.warning(f"non_overlap_cols failed ({e}); using w//4 fallback")
        return (0, img_w // 4) if side == "left" else (img_w * 3 // 4, img_w)


_strip_cols_cache = {}

def get_strip_cols(side, img_w):
    key = (side, img_w)
    if key not in _strip_cols_cache:
        ext   = EXT_LEFT  if side == "left" else EXT_RIGHT
        intr  = K_LEFT    if side == "left" else K_RIGHT
        _strip_cols_cache[key] = non_overlap_cols(ext, intr, EXT_FRONT, img_w, side)
    return _strip_cols_cache[key]


# ============================================================
# COLOR MATCHING — stable, seam-local, EMA-smoothed
# ============================================================
_color_ema: dict = {}

def match_color_stable(src, ref, cam_id: str) -> np.ndarray:
    """
    Conservative LAB colour match with three stability improvements:

    1. SEAM-LOCAL STATS — compute means/stds only from the strip columns
       that are adjacent to the front camera (inner 25% of the strip width).
       Global stats pulled the correction toward the dark outer edge of the
       side camera and caused the pink/green cast.

    2. EMA SMOOTHING (α=0.12) — running average of the per-channel
       correction parameters across frames.  Prevents the frame-to-frame
       colour jumping seen when each frame computed independent stats.

    3. CONSERVATIVE RATIOS —
       • L channel: std ratio clamped to [0.80, 1.20] to avoid wash-out.
       • a/b channels: only shift the mean (no std rescaling) at 40% strength.
         This removes the colour cast without introducing new tints.
    """
    EMA_ALPHA  = 0.12
    inner_frac = 0.25   # fraction of strip width used for stat sampling

    w = src.shape[1]
    inner = max(int(w * inner_frac), 30)

    # Sample from the edge that faces the front camera
    src_region = src[:, -inner:] if cam_id == "left" else src[:, :inner]
    ref_region = ref[:,  :inner] if cam_id == "left" else ref[:, -inner:]

    src_full_lab   = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_region_lab = cv2.cvtColor(src_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_region_lab = cv2.cvtColor(ref_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    params: dict = {}
    for ch in range(3):
        params[f"sm{ch}"] = float(src_region_lab[:, :, ch].mean())
        params[f"ss{ch}"] = float(src_region_lab[:, :, ch].std()) + 1e-6
        params[f"rm{ch}"] = float(ref_region_lab[:, :, ch].mean())
        params[f"rs{ch}"] = float(ref_region_lab[:, :, ch].std()) + 1e-6

    if cam_id not in _color_ema:
        _color_ema[cam_id] = params.copy()
    else:
        for k, v in params.items():
            _color_ema[cam_id][k] = EMA_ALPHA * v + (1 - EMA_ALPHA) * _color_ema[cam_id][k]

    p = _color_ema[cam_id]
    out = src_full_lab.copy()

    # L: luminance — match mean+std, clamped ratio
    ratio_L = float(np.clip(p["rs0"] / p["ss0"], 0.80, 1.20))
    out[:, :, 0] = (out[:, :, 0] - p["sm0"]) * ratio_L + p["rm0"]

    # a/b: mean shift only at 40% strength — removes cast, avoids new tints
    for ch in (1, 2):
        shift = float(np.clip(p[f"rm{ch}"] - p[f"sm{ch}"], -20, 20))
        out[:, :, ch] += shift * 0.40

    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ============================================================
# VIGNETTE CORRECTION
# ============================================================
def correct_vignette(img: np.ndarray, side: str) -> np.ndarray:
    """
    Compensate for lens fall-off on the outer edge of each side strip.
    The outer edge (away from the front camera) is darkest; we apply a
    1D power-law ramp: up to +25% brightness at the outermost column,
    tapering smoothly to 0 at the inner (seam-adjacent) edge.
    """
    h, w = img.shape[:2]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    # Left strip: left edge is outer.  Right strip: right edge is outer.
    ramp = (1.0 - x) ** 1.5 if side == "left" else x ** 1.5
    correction = (1.0 + 0.25 * ramp).reshape(1, w, 1)
    return np.clip(img.astype(np.float32) * correction, 0, 255).astype(np.uint8)


# ============================================================
# HIGHLIGHT PROTECTION  (right-side wash-out / glow)
# ============================================================
def protect_highlights(img: np.ndarray, knee: float = 210.0) -> np.ndarray:
    """
    Soft-compress luminance above `knee` to suppress the washed-out glow
    on the right strip caused by a strong light source in that camera's FOV.
    Pixels below knee are untouched; above knee they are gently rolled off
    with a sqrt-knee curve so very bright spots stay visually bright but
    no longer blow out to white.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    above = L > knee
    L[above] = knee + (255.0 - knee) * np.sqrt((L[above] - knee) / (255.0 - knee)) * 0.55
    lab[:, :, 0] = np.clip(L, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ============================================================
# VERTICAL ALIGNMENT (phase correlation)
# ============================================================
_vert_ema: dict = {}

def align_vertical(strip, front, side):
    """
    Detect and correct a small vertical pixel offset via phase correlation
    on a 60-column seam-edge band.

    The raw shift is smoothed with an EMA (α=0.25) so the strip doesn't
    jump up/down frame-to-frame when the correlator returns a noisy estimate.
    Hard clamp at ±40 px guards against complete misses on low-texture frames.
    """
    EMA_ALPHA = 0.25
    BW = 60
    try:
        band_s = strip[:, -BW:] if side == "left" else strip[:, :BW]
        band_f = front[:,  :BW] if side == "left" else front[:, -BW:]

        g_s = cv2.cvtColor(band_s, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g_f = cv2.cvtColor(band_f, cv2.COLOR_BGR2GRAY).astype(np.float32)

        (_, dy_raw), _ = cv2.phaseCorrelate(g_s, g_f)
        dy_raw = float(np.clip(dy_raw, -40, 40))

        if side not in _vert_ema:
            _vert_ema[side] = dy_raw
        else:
            _vert_ema[side] = EMA_ALPHA * dy_raw + (1 - EMA_ALPHA) * _vert_ema[side]

        dy = int(round(_vert_ema[side]))
        if dy == 0:
            return strip

        h, w = strip.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, dy]])
        return cv2.warpAffine(strip, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return strip


# ============================================================
# SEAM BLEND — sigmoid S-curve (human-eye peripheral transition)
# ============================================================
def alpha_blend_seam(img1, img2, bw):
    """
    Stitch img1 (left) and img2 (right) with a sigmoid crossfade.

    A sigmoid S-curve (logistic function over [-5, +5]) transitions fast
    in the middle and tapers gently at the outer edges — matching how human
    peripheral vision blends two fields.  A linear ramp produces a visible
    haze across the full blend zone; the sigmoid keeps the central imagery
    crisp while hiding the hard edge in the soft shoulder on either side.

    Output width = img1.w + img2.w - bw.
    alpha shape (1, bw, 1) broadcasts over (h, bw, 3).
    """
    h      = img1.shape[0]
    w1, w2 = img1.shape[1], img2.shape[1]
    out_w  = w1 + w2 - bw

    t     = np.linspace(-5.0, 5.0, bw, dtype=np.float32)
    alpha = (1.0 / (1.0 + np.exp(-t))).reshape(1, bw, 1)

    out = np.empty((h, out_w, 3), dtype=np.float32)
    out[:, :w1 - bw]   = img1[:, :w1 - bw]
    out[:, w1 - bw:w1] = (img1[:, w1 - bw:].astype(np.float32) * (1 - alpha)
                         + img2[:, :bw].astype(np.float32)      * alpha)
    out[:, w1:]         = img2[:, bw:]

    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# TEMPORAL SMOOTH
# ============================================================
prev_frame = None
def temporal_smooth(curr, alpha=0.75):
    global prev_frame
    if prev_frame is None:
        prev_frame = curr.astype(np.float32)
        return curr
    out = alpha * curr.astype(np.float32) + (1 - alpha) * prev_frame
    prev_frame = out
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# LOAD IMAGE
# ============================================================
def load_frame(base, frame_id):
    path = os.path.join(base, f"{CLIP_ID}_frame_{int(frame_id):06d}.jpg")
    if not os.path.exists(path):
        return None, path
    return cv2.imread(path), path


# ============================================================
# CORE FUSION
# ============================================================
def fuse(front, left, right):
    h, w = front.shape[:2]

    # 1. SCALE CORRECTION
    scale_left  = K_FRONT["fx"] / K_LEFT["fx"]
    scale_right = K_FRONT["fx"] / K_RIGHT["fx"]
    left  = cv2.resize(left,  None, fx=scale_left,  fy=scale_left,
                       interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(right, None, fx=scale_right, fy=scale_right,
                       interpolation=cv2.INTER_LINEAR)

    # 2. MATCH HEIGHT
    left  = cv2.resize(left,  (left.shape[1],  h), interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(right, (right.shape[1], h), interpolation=cv2.INTER_LINEAR)

    # 3. GEOMETRIC NON-OVERLAP CROP
    #    Uses extrinsics to find exactly which columns fall outside the front
    #    camera's FOV — eliminating the doubled-content artifact from w//2.
    lw = left.shape[1]
    rw = right.shape[1]
    l_a, l_b = get_strip_cols("left",  lw)
    r_a, r_b = get_strip_cols("right", rw)
    left_strip  = left[:,  l_a:l_b]
    right_strip = right[:, r_a:r_b]

    # 4. VIGNETTE CORRECTION — compensate outer-edge lens fall-off
    left_strip  = correct_vignette(left_strip,  "left")
    right_strip = correct_vignette(right_strip, "right")

    # 5. COLOR / EXPOSURE MATCHING — EMA-stabilised, seam-local, conservative
    left_strip  = match_color_stable(left_strip,  front, "left")
    right_strip = match_color_stable(right_strip, front, "right")

    # 6. HIGHLIGHT PROTECTION — suppress right-side wash-out / glow
    right_strip = protect_highlights(right_strip, knee=210.0)

    # 7. VERTICAL ALIGNMENT — EMA-stabilised phase correlation
    left_strip  = align_vertical(left_strip,  front, "left")
    right_strip = align_vertical(right_strip, front, "right")

    # 8. PANORAMA STITCHING — sigmoid seam blend
    bw = BLEND_WIDTH
    left_front = alpha_blend_seam(left_strip, front, bw=bw)
    fused      = alpha_blend_seam(left_front, right_strip, bw=bw)

    return fused


# ============================================================
# MAIN LOOP
# ============================================================
for idx, row in sync_df.iterrows():
    try:
        f_id = row["camera_front_wide_120fov.chunk_0000_frame"]
        l_id = row["camera_cross_left_120fov.chunk_0000_frame"]
        r_id = row["camera_cross_right_120fov.chunk_0000_frame"]

        if pd.isna(f_id) or pd.isna(l_id) or pd.isna(r_id):
            continue

        front, f_path = load_frame(FRONT_DIR, f_id)
        left,  l_path = load_frame(LEFT_DIR,  l_id)
        right, r_path = load_frame(RIGHT_DIR, r_id)

        if front is None or left is None or right is None:
            continue

        fused = fuse(front, left, right)
        fused = temporal_smooth(fused)

        out_path = os.path.join(OUT_DIR, f"{int(f_id):06d}.jpg")
        cv2.imwrite(out_path, fused)

        debug_data = {
            "frame":     int(f_id),
            "timestamp": float(row["timestamp"]),
            "paths":     {"front": f_path, "left": l_path, "right": r_path},
            "intrinsics": {"front": K_FRONT, "left": K_LEFT, "right": K_RIGHT},
            "extrinsics": {"front": EXT_FRONT, "left": EXT_LEFT, "right": EXT_RIGHT},
            "strip_cols": {
                "left":  list(_strip_cols_cache.get(("left",  left.shape[1]),  [])),
                "right": list(_strip_cols_cache.get(("right", right.shape[1]), [])),
            },
        }

        debug_path = os.path.join(DEBUG_DIR, f"{int(f_id):06d}.json")
        with open(debug_path, "w") as dbf:
            json.dump(debug_data, dbf, indent=4)

        logger.info(f"FRAME {int(f_id)} | ts={row['timestamp']:.3f} | "
                    f"front={f_path}")

    except Exception as e:
        logger.error(f"Error at index {idx}: {e}")
        continue

logger.info("Fusion V3 — completed successfully")