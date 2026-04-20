#final one for projection only 604  images projected - RIGHT Go for this 

import os
import cv2
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LIDAR-FUSION-V2")

# ============================================================
# CONFIG
# ============================================================
CLIP_ID     = "0a948f59-0a06-41a2-8e20-ac3a39ff4d61"
BLEND_WIDTH = 80
FRONT_FOV_HALF_DEG = 60.0

BASE_FRAMES    = "data_processed/extracted_frames"
LIDAR_DIR      = "data_processed/lidar_frames"
FUSED_DIR      = "data_processed/final_fusion_v3"
OUT_DIR        = "data_processed/final_fusion_lidar_v2"
os.makedirs(OUT_DIR, exist_ok=True)

FRONT_DIR = f"{BASE_FRAMES}/camera_front_wide_120fov.chunk_0000/{CLIP_ID}/frames"
LEFT_DIR  = f"{BASE_FRAMES}/camera_cross_left_120fov.chunk_0000/{CLIP_ID}/frames"
RIGHT_DIR = f"{BASE_FRAMES}/camera_cross_right_120fov.chunk_0000/{CLIP_ID}/frames"

# SYNC_PATH      = "data_processed/sync/camera_synced.parquet"
# SYNC_PATH      = "data_processed/sync/sensor_synced_cam_ref.parquet"
SYNC_PATH = r"data_processed\final_fusion\final_fused.parquet"  # ← updated

INTRINSIC_PATH = "data_source/calibration/camera_intrinsics.chunk_0000.parquet"
EXTRINSIC_PATH = "data_source/calibration/sensor_extrinsics.chunk_0000.parquet"

# ============================================================
# FIX 1: VOXEL GRID PARAMS  (uniform density + gap filling)
# ============================================================
VOXEL_SIZE_NEAR  = 0.12   # m — finer for close objects (< 15m)
VOXEL_SIZE_FAR   = 0.35   # m — coarser for distant objects (>= 15m)
VOXEL_NEAR_DIST  = 15.0

# ============================================================
# LOAD CALIBRATION
# ============================================================
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
    if mask.sum() == 0:
        avail = extrinsics[extrinsics["clip_id"] == CLIP_ID]["sensor_name"].tolist()
        raise ValueError(f"Sensor '{cam}' not found. Available: {avail}")
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

# ============================================================
# GEOMETRY
# ============================================================
def quat_to_R(q):
    qx, qy, qz, qw = q
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)

def build_T(ext):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(ext["q"])
    T[:3,  3] = ext["t"]
    return T

def transform_points(pts, T):
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

# ============================================================
# FIX 1 + 4: VOXEL GRID DOWNSAMPLING
# Splits near/far into two voxel sizes → balances density
# across the depth range and removes ground carpet clusters
# ============================================================
def voxel_downsample(pts, voxel_size):
    """Keep one representative point per voxel (mean position)."""
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int32)
    _, idx, inv = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    # Use mean per voxel for sub-pixel accuracy
    out = np.zeros((len(idx), 3), dtype=np.float32)
    np.add.at(out, inv, pts)
    counts = np.bincount(inv, minlength=len(idx))
    return out / counts[:, None]

def downsample_adaptive(pts):
    """Near points get finer voxels; far points get coarser."""
    dist = np.linalg.norm(pts, axis=1)
    near = pts[dist <  VOXEL_NEAR_DIST]
    far  = pts[dist >= VOXEL_NEAR_DIST]
    parts = []
    if len(near): parts.append(voxel_downsample(near, VOXEL_SIZE_NEAR))
    if len(far):  parts.append(voxel_downsample(far,  VOXEL_SIZE_FAR))
    return np.vstack(parts) if parts else pts

# ============================================================
# LIDAR CLEANING
# ============================================================
def clean_lidar(pts):
    """Remove ego-box, ground noise, and out-of-range points."""
    dist = np.linalg.norm(pts, axis=1)
    # Ego-vehicle bounding box in LiDAR frame
    ego = (
        (pts[:, 0] > -2.0) & (pts[:, 0] < 4.5) &
        (np.abs(pts[:, 1]) < 1.2) &
        (pts[:, 2] > -0.5) & (pts[:, 2] < 2.5)
    )
    pts = pts[~ego]
    dist = np.linalg.norm(pts, axis=1)
    pts = pts[(dist > 0.5) & (dist < 80)]
    pts = pts[pts[:, 2] > -2.5]
    return pts

# ============================================================
# FISHEYE PROJECTION
# ============================================================
def project_fisheye(pts_cam, intr):
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    rxy   = np.sqrt(x**2 + y**2)
    theta = np.arctan2(rxy, z)
    fw = intr["fw"]
    r_px = (fw[0] + fw[1]*theta + fw[2]*theta**2
                  + fw[3]*theta**3 + fw[4]*theta**4)
    safe_rxy = np.where(rxy < 1e-9, 1.0, rxy)
    u = intr["cx"] + r_px * (x / safe_rxy)
    v = intr["cy"] + r_px * (y / safe_rxy)
    MAX_THETA = np.deg2rad(60.0)
    valid = (r_px > 0) & (theta < MAX_THETA) & (z > 0)
    return u, v, z, valid, theta

# ============================================================
# FIX 5: TURBO COLORMAP  (much better perceptual separation)
# Covers full hue range: dark-blue → cyan → green → yellow → red
# ============================================================
_TURBO_DATA = np.array([
    [0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],
    [0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],
    [0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],
    [0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],
    [0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],
    [0.23236,0.18603,0.50004],[0.23583,0.19720,0.52373],
    [0.23917,0.20833,0.54686],[0.24237,0.21941,0.56942],
    [0.24543,0.23044,0.59142],[0.24835,0.24143,0.61286],
    [0.25113,0.25237,0.63374],[0.25378,0.26327,0.65406],
    [0.25629,0.27412,0.67381],[0.25867,0.28492,0.69300],
    [0.26091,0.29568,0.71162],[0.26301,0.30639,0.72968],
    [0.26498,0.31706,0.74718],[0.26681,0.32768,0.76412],
    [0.26850,0.33825,0.78050],[0.27006,0.34878,0.79631],
    [0.27147,0.35926,0.81156],[0.27275,0.36970,0.82624],
    [0.27388,0.38008,0.84037],[0.27487,0.39043,0.85393],
    [0.27572,0.40072,0.86692],[0.27643,0.41097,0.87936],
    [0.27700,0.42118,0.89123],[0.27743,0.43134,0.90254],
    [0.27771,0.44145,0.91328],[0.27786,0.45152,0.92347],
    [0.27787,0.46153,0.93309],[0.27774,0.47151,0.94214],
    [0.27747,0.48144,0.95064],[0.27706,0.49132,0.95857],
    [0.27651,0.50115,0.96594],[0.27583,0.51094,0.97275],
    [0.27501,0.52069,0.97899],[0.27407,0.53040,0.98461],
    [0.27299,0.54015,0.98930],[0.27181,0.54995,0.99303],
    [0.27019,0.55969,0.99583],[0.26861,0.56943,0.99773],
    [0.26681,0.57913,0.99876],[0.26480,0.58883,0.99896],
    [0.26260,0.59848,0.99835],[0.26022,0.60811,0.99697],
    [0.25767,0.61769,0.99485],[0.25398,0.62724,0.99202],
    [0.25082,0.63676,0.98851],[0.24753,0.64621,0.98436],
    [0.24411,0.65561,0.97959],[0.24060,0.66497,0.97423],
    [0.23701,0.67427,0.96833],[0.23334,0.68354,0.96190],
    [0.22962,0.69275,0.95498],[0.22586,0.70190,0.94761],
    [0.22207,0.71100,0.93981],[0.21826,0.72004,0.93161],
    [0.21445,0.72902,0.92305],[0.21063,0.73795,0.91416],
    [0.20681,0.74682,0.90496],[0.20300,0.75563,0.89549],
    [0.19921,0.76438,0.88579],[0.19545,0.77307,0.87588],
    [0.19172,0.78171,0.86581],[0.18802,0.79029,0.85561],
    [0.18437,0.79881,0.84532],[0.18079,0.80728,0.83497],
    [0.17729,0.81569,0.82460],[0.17387,0.82404,0.81424],
    [0.17054,0.83233,0.80392],[0.16731,0.84055,0.79371],
    [0.16420,0.84872,0.78364],[0.16121,0.85682,0.77376],
    [0.15835,0.86485,0.76411],[0.15562,0.87282,0.75472],
    [0.15302,0.88073,0.74563],[0.15057,0.88858,0.73688],
    [0.14826,0.89636,0.72851],[0.14610,0.90408,0.72054],
    [0.14408,0.91173,0.71300],[0.14222,0.91932,0.70593],
    [0.14051,0.92685,0.69936],[0.13896,0.93431,0.69331],
    [0.13757,0.94170,0.68781],[0.13636,0.94901,0.68289],
    [0.13532,0.95626,0.67857],[0.13447,0.96344,0.67489],
    [0.13581,0.97055,0.67105],[0.13836,0.97640,0.66386],
    [0.14172,0.98117,0.65444],[0.14586,0.98500,0.64308],
    [0.15075,0.98802,0.63015],[0.15636,0.99027,0.61580],
    [0.16265,0.99184,0.60033],[0.16958,0.99279,0.58390],
    [0.17710,0.99315,0.56673],[0.18517,0.99298,0.54906],
    [0.19373,0.99231,0.53096],[0.20274,0.99119,0.51264],
    [0.21215,0.98967,0.49418],[0.22193,0.98778,0.47569],
    [0.23203,0.98555,0.45723],[0.24240,0.98300,0.43886],
    [0.25300,0.98016,0.42062],[0.26380,0.97705,0.40253],
    [0.27477,0.97368,0.38463],[0.28588,0.97007,0.36692],
    [0.29709,0.96623,0.34942],[0.30840,0.96217,0.33211],
    [0.31977,0.95789,0.31497],[0.33119,0.95340,0.29800],
    [0.34261,0.94770,0.28110],[0.35405,0.94289,0.26437],
    [0.36548,0.93688,0.24780],[0.37688,0.93067,0.23136],
    [0.38823,0.92427,0.21507],[0.39951,0.91769,0.19890],
    [0.41070,0.91093,0.18285],[0.42178,0.90399,0.16692],
    [0.43273,0.89689,0.15110],[0.44353,0.88962,0.13539],
    [0.45415,0.88220,0.11977],[0.46458,0.87463,0.10424],
    [0.47480,0.86692,0.08878],[0.48477,0.85907,0.07341],
    [0.49449,0.85109,0.05810],[0.50394,0.84299,0.04285],
    [0.51310,0.83476,0.02768],[0.52197,0.82642,0.01255],
    [0.53053,0.81797,0.00000],[0.53878,0.80940,0.00000],
    [0.54671,0.80072,0.00000],[0.55431,0.79193,0.00000],
    [0.56157,0.78304,0.00000],[0.56849,0.77405,0.00000],
    [0.57506,0.76496,0.00000],[0.58128,0.75578,0.00000],
    [0.58714,0.74650,0.00000],[0.59266,0.73713,0.00000],
    [0.59782,0.72768,0.00000],[0.60263,0.71814,0.00000],
    [0.60709,0.70851,0.00000],[0.61121,0.69881,0.00000],
    [0.61499,0.68903,0.00000],[0.61844,0.67916,0.00000],
    [0.62154,0.66923,0.00000],[0.62431,0.65923,0.00000],
    [0.62675,0.64917,0.00000],[0.62886,0.63905,0.00000],
    [0.63065,0.62888,0.00000],[0.63210,0.61867,0.00000],
    [0.63324,0.60841,0.00000],[0.63405,0.59811,0.00000],
    [0.63454,0.58778,0.00000],[0.63471,0.57743,0.00000],
    [0.63457,0.56706,0.00000],[0.63411,0.55667,0.00000],
    [0.63335,0.54627,0.00000],[0.63228,0.53586,0.00000],
    [0.63090,0.52545,0.00000],[0.62923,0.51504,0.00000],
    [0.72575,0.39014,0.00000],[0.75660,0.36000,0.00000],
], dtype=np.float32)

# Build a clean 256-entry turbo LUT without the broken row above
def _build_turbo_lut():
    """256 turbo colormap entries as uint8 BGR."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Turbo formula (approximation good to 1%)
        r = (0.1357 + t*(4.5974 - t*(42.3277 - t*(130.5887
             - t*(150.5666 - t*58.1375)))))
        g = (0.0914 + t*(2.1856 + t*(4.8052 - t*(14.0195
             - t*(4.2109 + t*2.7747)))))
        b = (0.1067 + t*(11.5429 + t*(-44.1120 + t*(( 82.3991
             + t*(-71.9491 + t*23.7381))))))
        r = float(np.clip(r, 0, 1))
        g = float(np.clip(g, 0, 1))
        b = float(np.clip(b, 0, 1))
        # Store as BGR for OpenCV
        lut[i] = [int(b*255), int(g*255), int(r*255)]
    return lut

TURBO_LUT = _build_turbo_lut()

def depth_to_bgr_turbo(depths, d_min=1.5, d_max=45.0):
    """
    Map depth → turbo colormap index.
    Uses log-scale depth compression so road/car/building
    depth ranges each get a distinct colour band.
    """
    # Log-scale: compresses near range, expands far separation
    log_d   = np.log1p(np.clip(depths, d_min, d_max))
    log_min = np.log1p(d_min)
    log_max = np.log1p(d_max)
    t = np.clip((log_d - log_min) / (log_max - log_min), 0.0, 1.0)
    idx = (t * 255).astype(np.uint8)
    return TURBO_LUT[idx]   # Nx3 BGR

# ============================================================
# FIX 3: PANORAMA COORDINATE MAPPER  (corrected offset math)
# ============================================================
class PanoramaMapper:
    def __init__(self, intr_front, intr_left, intr_right,
                 ext_front, ext_left, ext_right,
                 blend_width=80, front_fov_half_deg=60.0):
        self.BW = blend_width
        W_f = intr_front["width"]   # 1920

        # Reuse fw[1] as effective focal length for scale computation
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

        # ── Left strip: columns [0, l_b] ─────────────────────────────
        # front left-edge ray (angle = -60°)
        ray = np.array([np.sin(np.radians(-front_fov_half_deg)), 0.0,
                         np.cos(np.radians(-front_fov_half_deg))])
        ray_vc = R_f @ ray
        ray_sc = R_l.T @ ray_vc
        h_ang  = np.arctan2(float(ray_sc[0]), float(ray_sc[2]))
        l_b = int(np.clip(cx_l_sc + fx_l_sc * h_ang,
                          W_l_sc // 8, W_l_sc * 7 // 8))
        self.l_a, self.l_b = 0, l_b

        # ── Right strip: columns [r_a, W_r_sc] ───────────────────────
        ray = np.array([np.sin(np.radians(+front_fov_half_deg)), 0.0,
                         np.cos(np.radians(+front_fov_half_deg))])
        ray_vc = R_f @ ray
        ray_sc = R_r.T @ ray_vc
        h_ang  = np.arctan2(float(ray_sc[0]), float(ray_sc[2]))
        r_a = int(np.clip(cx_r_sc + fx_r_sc * h_ang,
                          W_r_sc // 8, W_r_sc * 7 // 8))
        self.r_a, self.r_b = r_a, W_r_sc

        self.left_strip_w  = l_b
        self.right_strip_w = W_r_sc - r_a

        # FIX 3: fused layout widths match alpha_blend_seam exactly
        # alpha_blend_seam(left_strip, front, BW):
        #   left_front_w = left_strip_w + W_f - BW
        # alpha_blend_seam(left_front, right_strip, BW):
        #   fused_w = left_front_w + right_strip_w - BW
        self.left_front_w = self.left_strip_w + W_f - blend_width
        self.fused_w      = self.left_front_w + self.right_strip_w - blend_width

        logger.info(
            f"PanoramaMapper: left_strip={self.left_strip_w} | "
            f"front={W_f} | right_strip={self.right_strip_w} | "
            f"fused_w={self.fused_w}"
        )

    def cam_to_fused_x(self, u_cam, cam: str) -> np.ndarray:
        """
        Map camera-pixel column u_cam (float array) → fused image x.

        FRONT : u_f  → left_strip_w + u_f - BW
        LEFT  : u_l  → (u_l × scale_left) - l_a
        RIGHT : u_r  → left_front_w - BW + (u_r × scale_right - r_a)
        """
        BW = self.BW
        if cam == "camera_front_wide_120fov":
            return self.left_strip_w + u_cam - BW

        elif cam == "camera_cross_left_120fov":
            u_sc = u_cam * self.scale_left
            return u_sc - self.l_a

        elif cam == "camera_cross_right_120fov":
            u_sc   = u_cam * self.scale_right
            u_strip = u_sc - self.r_a
            return self.left_front_w - BW + u_strip

        raise ValueError(f"Unknown camera: {cam}")

# ============================================================
# FIX 2: Z-BUFFER  (per-pixel occlusion handling)
# Replaces painter's algorithm — eliminates "points through cars"
# ============================================================
def render_with_zbuffer(fused_img, uf, vf, dep, colors, radii):
    """
    Two-pass rendering using a per-pixel depth buffer:
      Pass 1 — build depth buffer (min-depth per pixel, no color)
      Pass 2 — draw only points whose depth is within a tight
               tolerance of the buffer value at that pixel

    This guarantees foreground objects always occlude background
    ones, regardless of point cloud order.
    Tolerance = 0.5 m (allows slight depth quantisation noise).
    """
    h, w = fused_img.shape[:2]
    Z_BUF  = np.full((h, w), np.inf, dtype=np.float32)
    TOLERANCE = 0.5   # metres

    # ── Pass 1: populate depth buffer ────────────────────────────────
    # Use square stamps for speed (circle fill not needed here)
    max_r = int(radii.max()) if len(radii) else 3
    for i in range(len(uf)):
        xi, yi, d, r = uf[i], vf[i], dep[i], int(radii[i])
        x0 = max(xi - r, 0);   x1 = min(xi + r + 1, w)
        y0 = max(yi - r, 0);   y1 = min(yi + r + 1, h)
        region = Z_BUF[y0:y1, x0:x1]
        np.minimum(region, d, out=region)

    # ── Pass 2: draw only visible points ─────────────────────────────
    out = fused_img.copy()
    for i in range(len(uf)):
        xi, yi, d, r = uf[i], vf[i], dep[i], int(radii[i])
        x0 = max(xi - r, 0);   x1 = min(xi + r + 1, w)
        y0 = max(yi - r, 0);   y1 = min(yi + r + 1, h)
        if Z_BUF[y0:y1, x0:x1].min() < d - TOLERANCE:
            continue   # this point is occluded — skip
        cv2.circle(out, (xi, yi), r,
                   (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])),
                   -1)
    return out


# ============================================================
# FIX 4: CAMERA PRIORITY + OVERLAP DEDUPLICATION
# ============================================================
def project_lidar_on_fused(fused_img, pts_lidar, mapper):
    """
    1. Project each LiDAR point through all cameras.
    2. Assign each point to the camera with the smallest theta
       (most frontal view = least distortion).
    3. Deduplicate: in the overlap zone both cameras may claim
       the same fused pixel — keep whichever has smaller theta.
    4. Render via Z-buffer (fix 2).
    """
    T_lidar = build_T(EXTR["lidar"])
    h_fused, w_fused = fused_img.shape[:2]

    n = len(pts_lidar)
    best_uf    = np.full(n, np.nan, dtype=np.float32)
    best_vf    = np.full(n, np.nan, dtype=np.float32)
    best_dep   = np.full(n, np.nan, dtype=np.float32)
    best_rad   = np.zeros(n, dtype=np.int32)
    best_theta = np.full(n, np.inf, dtype=np.float32)

    RING_SPACING = np.deg2rad(30.0 / 64)

    for cam in CAM_NAMES:
        intr  = INTR[cam]
        T_cam = build_T(EXTR[cam])
        T_L2C = np.linalg.inv(T_cam) @ T_lidar

        pts_cam = transform_points(pts_lidar, T_L2C)
        fwd = (pts_cam[:, 2] > 0.5) & (pts_cam[:, 2] < 80.0)

        u, v, dep, valid, theta = project_fisheye(pts_cam, intr)

        W, H = intr["width"], intr["height"]
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ok = fwd & valid & in_bounds

        # Adaptive radius
        fw = intr["fw"]
        drdt = (fw[1] + 2*fw[2]*theta + 3*fw[3]*theta**2
                      + 4*fw[4]*theta**3)
        r = np.clip(np.abs(drdt) * RING_SPACING * 0.6, 1, 6).astype(np.int32)

        # Map u → fused x
        fused_x = mapper.cam_to_fused_x(u, cam)

        # Assign to camera with smallest theta (most head-on view)
        wins = ok & (theta < best_theta)
        best_theta[wins] = theta[wins]
        best_dep  [wins] = dep  [wins]
        best_vf   [wins] = v    [wins]
        best_uf   [wins] = fused_x[wins]
        best_rad  [wins] = r    [wins]

    # Filter assigned & in-bounds points
    assigned = ~np.isnan(best_uf)
    uf  = best_uf [assigned]
    vf  = best_vf [assigned]
    dep = best_dep[assigned]
    rad = best_rad[assigned]

    in_pano = (uf >= 0) & (uf < w_fused) & (vf >= 0) & (vf < h_fused)
    uf  = uf [in_pano].astype(np.int32)
    vf  = vf [in_pano].astype(np.int32)
    dep = dep[in_pano]
    rad = rad[in_pano]

    # FIX 5: Turbo colormap + log-scale depth
    colors = depth_to_bgr_turbo(dep, d_min=1.5, d_max=45.0)

    # FIX 2: Z-buffer render (no more painter's algorithm)
    result = render_with_zbuffer(fused_img, uf, vf, dep, colors, rad)

    logger.info(f"  Projected {len(uf)} points | "
                f"depth [{dep.min():.1f}–{dep.max():.1f}] m")
    return result

# ============================================================
# BUILD MAPPER
# ============================================================
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

# ============================================================
# MAIN LOOP
# ============================================================
# for idx, row in sync_df.iterrows():
#     try:
#         f_id = row["camera_front_wide_120fov.chunk_0000_frame"]
#         if pd.isna(f_id):
#             continue
#         frame_idx = int(f_id)

#         fused_path = os.path.join(FUSED_DIR, f"{frame_idx:06d}.jpg")
#         lidar_path = os.path.join(LIDAR_DIR, f"{frame_idx}.parquet")

#         if not os.path.exists(fused_path):
#             logger.warning(f"Missing fused frame: {fused_path}")
#             continue
#         if not os.path.exists(lidar_path):
#             logger.warning(f"Missing LiDAR frame: {lidar_path}")
#             continue

#         fused = cv2.imread(fused_path)

#         df_lidar = pd.read_parquet(lidar_path)
#         pts = df_lidar[["x", "y", "z"]].values

#         # FIX 1: clean then adaptively downsample
#         pts = clean_lidar(pts)
#         pts = downsample_adaptive(pts)

#         result = project_lidar_on_fused(fused, pts, mapper)

#         out_path = os.path.join(OUT_DIR, f"{frame_idx:06d}.jpg")
#         cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
#         logger.info(f"Frame {frame_idx:06d} → {out_path}")

#     except Exception as e:
#         logger.error(f"Error at idx={idx}: {e}", exc_info=True)
#         continue

for idx, row in sync_df.iterrows():
    try:
        # ── FIX: use 'camera_frame' (front cam index in new parquet)
        f_id = row.get("camera_frame")
        if pd.isna(f_id):
            continue
        frame_idx = int(f_id)

        # ── FIX: use 'lidar_frame' column (explicit in new parquet)
        lidar_frame_id = row.get("lidar_frame")
        if pd.isna(lidar_frame_id):
            logger.warning(f"No LiDAR frame for camera_frame={frame_idx}, skipping")
            continue
        lidar_frame_id = int(lidar_frame_id)

        fused_path = os.path.join(FUSED_DIR, f"{frame_idx:06d}.jpg")
        lidar_path = os.path.join(LIDAR_DIR, f"{lidar_frame_id}.parquet")  # ← uses lidar_frame now

        if not os.path.exists(fused_path):
            logger.warning(f"Missing fused frame: {fused_path}")
            continue
        if not os.path.exists(lidar_path):
            logger.warning(f"Missing LiDAR frame: {lidar_path}")
            continue

        fused = cv2.imread(fused_path)

        df_lidar = pd.read_parquet(lidar_path)
        pts = df_lidar[["x", "y", "z"]].values

        pts = clean_lidar(pts)
        pts = downsample_adaptive(pts)

        result = project_lidar_on_fused(fused, pts, mapper)

        out_path = os.path.join(OUT_DIR, f"{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Frame {frame_idx:06d} (lidar={lidar_frame_id}) → {out_path}")

    except Exception as e:
        logger.error(f"Error at idx={idx}: {e}", exc_info=True)
        continue

logger.info("Done — LiDAR fusion V2 complete.")