import pandas as pd
import numpy as np
import os
import logging
import DracoPy
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LIDAR_PIPELINE")
INPUT_PATH = "data_source/lidar/lidar_top_360fov.chunk_0000/0a948f59-0a06-41a2-8e20-ac3a39ff4d61.lidar_top_360fov.parquet"

# INPUT_PATH = "data_source/lidar/lidar_top_360fov.chunk_0000/xxx.parquet"
OUTPUT_DIR = "data_processed/lidar_cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DECODE
# =========================
def decode_draco(blob):
    try:
        mesh = DracoPy.decode(blob)
        return np.array(mesh.points)
    except:
        return None

# =========================
# PREPROCESS
# =========================
def preprocess(points):

    if len(points) == 0:
        return points

    # Ground remove
    points = points[points[:,2] > -1.2]

    # Ego remove
    points = points[
        ~(
            (points[:,0] > -1.5) & (points[:,0] < 3.0) &
            (points[:,1] > -1.2) & (points[:,1] < 1.2)
        )
    ]

    # Front only
    dist = np.sqrt(points[:,0]**2 + points[:,1]**2)
    points = points[
        (points[:,0] > 0) &
        (dist < 40) &
        (points[:,2] < 3)
    ]

    # Density filter
    if len(points) > 10:
        try:
            nbrs = NearestNeighbors(n_neighbors=5).fit(points)
            d, _ = nbrs.kneighbors(points)
            points = points[d[:,4] < 1.0]
        except:
            pass

    return points

# =========================
# MAIN
# =========================
def main():

    df = pd.read_parquet(INPUT_PATH)

    ts_col = [c for c in df.columns if "time" in c.lower()][0]

    for i, row in enumerate(df.itertuples()):

        blob = getattr(row, "draco_encoded_pointcloud")
        pts = decode_draco(blob)

        if pts is None:
            continue

        pts = preprocess(pts)

        out = pd.DataFrame(pts, columns=["x","y","z"])
        out["frame_id"] = i
        out["timestamp"] = getattr(row, ts_col)

        out.to_parquet(f"{OUTPUT_DIR}/{i}.parquet", index=False)

        if i % 20 == 0:
            logger.info(f"Processed frame {i}")

    print("Script 1 DONE")

if __name__ == "__main__":
    main()