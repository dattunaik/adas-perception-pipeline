"""
ADAS Detection Pipeline
=======================
Production-grade Advanced Driver Assistance System (ADAS) object detection
using YOLOv8 with image enhancement, multi-scale inference, object tracking,
risk assessment, and structured output management.

Author  : ADAS Vision Team
Version : 2.0.0
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# ENUMS
# ──────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """ADAS risk classification for detected objects."""
    CRITICAL = "CRITICAL"   # Immediate collision risk
    HIGH     = "HIGH"       # Close proximity, action needed
    MEDIUM   = "MEDIUM"     # Monitor closely
    LOW      = "LOW"        # Nominal awareness
    CLEAR    = "CLEAR"      # No concern


class ADASClass(str, Enum):
    """ADAS-relevant object classes mapped from COCO labels."""
    CAR           = "car"
    TRUCK         = "truck"
    BUS           = "bus"
    MOTORCYCLE    = "motorcycle"
    BICYCLE       = "bicycle"
    PERSON        = "person"
    TRAFFIC_LIGHT = "traffic light"
    STOP_SIGN     = "stop sign"

    @classmethod
    def values(cls) -> set:
        return {m.value for m in cls}


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """YOLOv8 model inference settings."""
    model_path:       str   = "yolov8m.pt"
    conf_threshold:   float = 0.35
    iou_threshold:    float = 0.50
    input_size:       int   = 1280
    use_tta:          bool  = True   # Test-Time Augmentation
    device:           str   = "cpu"  # "cpu" | "cuda" | "mps"


@dataclass
class EnhancementConfig:
    """Image pre-processing / enhancement settings."""
    alpha:             float = 1.1          # Brightness scale
    beta:              int   = 10           # Brightness offset
    clahe_clip_limit:  float = 2.0
    clahe_tile_size:   Tuple[int, int] = (8, 8)


@dataclass
class RiskConfig:
    """Thresholds for ADAS risk scoring."""
    # Fraction of image height that bbox bottom must exceed
    critical_proximity: float = 0.80
    high_proximity:     float = 0.65
    medium_proximity:   float = 0.45
    # Fraction of image width that bbox must span
    critical_width_frac: float = 0.35
    high_width_frac:     float = 0.20
    # Per-class risk weight (higher = more dangerous)
    class_weights: Dict[str, float] = field(default_factory=lambda: {
        "person":        1.5,
        "motorcycle":    1.3,
        "bicycle":       1.2,
        "car":           1.0,
        "bus":           1.0,
        "truck":         1.0,
        "traffic light": 0.8,
        "stop sign":     0.7,
    })


@dataclass
class PathConfig:
    """All I/O directory settings."""
    clip_id:            str  = "0a948f59-0a06-41a2-8e20-ac3a39ff4d61"
    input_dir:          Path = Path("data_processed/final_fusion_v3")
    output_dir:         Path = Path("data_processed/fused_detections")

    # Derived — populated in __post_init__
    vis_dir:            Path = field(init=False)
    per_frame_json_dir: Path = field(init=False)

    def __post_init__(self):
        self.vis_dir            = self.output_dir / "visualizations"
        self.per_frame_json_dir = self.output_dir / "per_frame_json"

    def make_dirs(self) -> None:
        for d in [self.output_dir, self.vis_dir, self.per_frame_json_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration aggregating all sub-configs."""
    model:       ModelConfig      = field(default_factory=ModelConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    risk:        RiskConfig       = field(default_factory=RiskConfig)
    paths:       PathConfig       = field(default_factory=PathConfig)
    log_interval: int             = 20   # frames between progress logs
    save_vis:     bool            = True
    save_parquet: bool            = True


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def build_logger(name: str = "ADAS-Pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    clip_id:     str
    frame_id:    int
    class_name:  str
    confidence:  float
    bbox:        BoundingBox
    risk_level:  RiskLevel = RiskLevel.CLEAR
    track_id:    Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "clip_id":    self.clip_id,
            "frame_id":   self.frame_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox":       self.bbox.to_list(),
            "risk_level": self.risk_level.value,
            "track_id":   self.track_id,
            "x1":         self.bbox.x1,
            "y1":         self.bbox.y1,
            "x2":         self.bbox.x2,
            "y2":         self.bbox.y2,
        }

@dataclass
class FrameResult:
    clip_id:    str
    frame_id:   int
    detections: List[Detection]
    proc_ms:    float = 0.0

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def has_critical(self) -> bool:
        return any(d.risk_level == RiskLevel.CRITICAL for d in self.detections)

    def to_dict(self) -> dict:
        return {
            "clip_id":       self.clip_id,
            "frame_id":      self.frame_id,
            "proc_ms":       round(self.proc_ms, 2),
            "has_critical":  self.has_critical,
            "detection_count": self.detection_count,
            "detections":    [d.to_dict() for d in self.detections],
        }

# ──────────────────────────────────────────────────────────────────────────────
# IMAGE ENHANCER
# ──────────────────────────────────────────────────────────────────────────────

class ImageEnhancer:
    """
    Applies brightness/contrast normalization and CLAHE to improve detection
    in low-contrast or poorly lit driving scenes.
    """

    def __init__(self, cfg: EnhancementConfig) -> None:
        self.cfg   = cfg
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_size,
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self._enhance(img)

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        # 1. Global brightness + contrast
        img = cv2.convertScaleAbs(img, alpha=self.cfg.alpha, beta=self.cfg.beta)

        # 2. CLAHE in LAB colour space (only L channel)
        lab        = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        l          = self.clahe.apply(l)
        img        = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        return img


# ──────────────────────────────────────────────────────────────────────────────
# RISK ASSESSOR
# ──────────────────────────────────────────────────────────────────────────────

class RiskAssessor:
    """
    Assigns an ADAS RiskLevel to each detection based on:
      - Proximity heuristic (how low in the frame the object appears)
      - Object width relative to frame width (closeness proxy)
      - Per-class danger weight
    """

    # Colour coding for visualisation overlay
    RISK_COLOURS: Dict[RiskLevel, Tuple[int, int, int]] = {
        RiskLevel.CRITICAL: (0,   0,   255),  # Red
        RiskLevel.HIGH:     (0,  128,  255),  # Orange
        RiskLevel.MEDIUM:   (0,  255,  255),  # Yellow
        RiskLevel.LOW:      (0,  255,    0),  # Green
        RiskLevel.CLEAR:    (200, 200,  200),  # Grey
    }

    def __init__(self, cfg: RiskConfig, img_h: int, img_w: int) -> None:
        self.cfg   = cfg
        self.img_h = img_h
        self.img_w = img_w

    def assess(self, det: Detection) -> RiskLevel:
        bbox   = det.bbox
        weight = self.cfg.class_weights.get(det.class_name, 1.0)

        # Normalised proximity (0→top, 1→bottom of frame)
        proximity    = bbox.y2 / self.img_h
        width_frac   = bbox.width / self.img_w

        # Combine signals
        prox_score = proximity * weight
        size_score = width_frac * weight

        if prox_score >= self.cfg.critical_proximity or size_score >= self.cfg.critical_width_frac:
            return RiskLevel.CRITICAL
        if prox_score >= self.cfg.high_proximity or size_score >= self.cfg.high_width_frac:
            return RiskLevel.HIGH
        if prox_score >= self.cfg.medium_proximity:
            return RiskLevel.MEDIUM
        if prox_score > 0:
            return RiskLevel.LOW

        return RiskLevel.CLEAR

    def colour_for(self, risk: RiskLevel) -> Tuple[int, int, int]:
        return self.RISK_COLOURS[risk]


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISER
# ──────────────────────────────────────────────────────────────────────────────

class Visualiser:
    """Draws ADAS-style overlays on frames."""

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    THICKNESS  = 1

    def draw(
        self,
        img:       np.ndarray,
        result:    FrameResult,
        assessor:  RiskAssessor,
    ) -> np.ndarray:
        vis = img.copy()

        for det in result.detections:
            colour = assessor.colour_for(det.risk_level)
            b      = det.bbox
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

            # Label background
            label = f"{det.class_name} {det.confidence:.2f} [{det.risk_level.value}]"
            (tw, th), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, self.THICKNESS)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), self.FONT,
                        self.FONT_SCALE, (0, 0, 0), self.THICKNESS, cv2.LINE_AA)

        # HUD overlay
        self._draw_hud(vis, result)
        return vis

    # ------------------------------------------------------------------
    def _draw_hud(self, vis: np.ndarray, result: FrameResult) -> None:
        """Minimal heads-up display in top-left corner."""
        lines = [
            f"Frame : {result.frame_id:>6}",
            f"Objects: {result.detection_count:>4}",
            f"ProcMs : {result.proc_ms:>6.1f}",
        ]
        if result.has_critical:
            lines.append("!! CRITICAL ALERT !!")

        y_offset = 20
        for line in lines:
            colour = (0, 0, 255) if "CRITICAL" in line else (255, 255, 255)
            cv2.putText(vis, line, (10, y_offset), self.FONT,
                        0.55, colour, 1, cv2.LINE_AA)
            y_offset += 22


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class OutputManager:
    """Handles all I/O: per-frame JSON, aggregate JSON, Parquet, and images."""

    def __init__(self, cfg: PathConfig, logger: logging.Logger) -> None:
        self.cfg    = cfg
        self.logger = logger
        cfg.make_dirs()

    # ------------------------------------------------------------------
    def save_frame_json(self, result: FrameResult) -> None:
        path = self.cfg.per_frame_json_dir / f"{result.frame_id}.json"
        with open(path, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)

    # ------------------------------------------------------------------
    def save_vis(self, frame_name: str, vis: np.ndarray) -> None:
        path = self.cfg.vis_dir / frame_name
        cv2.imwrite(str(path), vis)

    # ------------------------------------------------------------------
    def save_aggregate(
        self,
        all_results: List[FrameResult],
        save_parquet: bool,
    ) -> None:
        clip_id = self.cfg.clip_id

        # ── Aggregate JSON ─────────────────────────────────────────────
        aggregate: Dict[str, list] = {}
        for r in all_results:
            aggregate[str(r.frame_id)] = [d.to_dict() for d in r.detections]

        json_path = self.cfg.output_dir / f"{clip_id}_fused_detections.json"
        with open(json_path, "w") as fh:
            json.dump(aggregate, fh, indent=2)
        self.logger.info(f"Aggregate JSON → {json_path}")

        # ── Parquet ────────────────────────────────────────────────────
        if save_parquet:
            rows = [d.to_dict() for r in all_results for d in r.detections]
            if rows:
                df = pd.DataFrame(rows)
                parquet_path = self.cfg.output_dir / f"{clip_id}_fused_detections.parquet"
                df.to_parquet(parquet_path, index=False)
                self.logger.info(f"Parquet         → {parquet_path}")

        # ── Summary stats ──────────────────────────────────────────────
        self._save_summary(all_results)

    # ------------------------------------------------------------------
    def _save_summary(self, all_results: List[FrameResult]) -> None:
        total_dets    = sum(r.detection_count for r in all_results)
        critical_frames = sum(1 for r in all_results if r.has_critical)
        avg_proc_ms   = (
            sum(r.proc_ms for r in all_results) / len(all_results)
            if all_results else 0.0
        )

        class_counts: Dict[str, int] = {}
        risk_counts:  Dict[str, int] = {}
        for r in all_results:
            for d in r.detections:
                class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
                risk_counts[d.risk_level.value] = risk_counts.get(d.risk_level.value, 0) + 1

        summary = {
            "clip_id":         self.cfg.clip_id,
            "total_frames":    len(all_results),
            "total_detections": total_dets,
            "critical_frames": critical_frames,
            "avg_proc_ms":     round(avg_proc_ms, 2),
            "class_breakdown": class_counts,
            "risk_breakdown":  risk_counts,
        }

        path = self.cfg.output_dir / f"{self.cfg.clip_id}_summary.json"
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2)
        self.logger.info(f"Summary         → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# DETECTION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

class ADASDetectionPipeline:
    """
    Orchestrates the full ADAS detection pipeline:
      1. Load YOLOv8 model
      2. Iterate frames from input directory
      3. Enhance each frame
      4. Run multi-scale inference
      5. Assess risk per detection
      6. Visualise and persist results
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg        = cfg
        self.logger     = build_logger()
        self.enhancer   = ImageEnhancer(cfg.enhancement)
        self.visualiser = Visualiser()
        self.output_mgr = OutputManager(cfg.paths, self.logger)
        self.model      = self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> YOLO:
        self.logger.info(f"Loading YOLO model: {self.cfg.model.model_path}")
        model = YOLO(self.cfg.model.model_path)
        self.logger.info(f"Model loaded — classes: {len(model.names)}")
        return model

    # ------------------------------------------------------------------
    def _collect_frames(self) -> List[str]:
        src = self.cfg.paths.input_dir
        if not src.exists():
            raise FileNotFoundError(f"Input directory not found: {src}")

        frames = sorted(
            f for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not frames:
            raise RuntimeError(f"No image files found in {src}")
        return frames

    # ------------------------------------------------------------------
    @staticmethod
    def _frame_id(filename: str) -> int:
        return int(Path(filename).stem)

    # ------------------------------------------------------------------
    def _infer(self, img: np.ndarray):
        """Run YOLOv8 inference with configured settings."""
        m_cfg = self.cfg.model
        return self.model.predict(
            img,
            conf=m_cfg.conf_threshold,
            iou=m_cfg.iou_threshold,
            imgsz=m_cfg.input_size,
            augment=m_cfg.use_tta,
            device=m_cfg.device,
            verbose=False,
        )[0]

    # ------------------------------------------------------------------
    def _parse_detections(
        self,
        raw_results,
        frame_id: int,
        assessor: RiskAssessor,
    ) -> List[Detection]:
        detections: List[Detection] = []

        if raw_results.boxes is None:
            return detections

        valid_classes = ADASClass.values()

        for box in raw_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf            = float(box.conf[0])
            cls             = int(box.cls[0])
            label           = self.model.names[cls]

            if label not in valid_classes:
                continue

            bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
            det  = Detection(
                clip_id=self.cfg.paths.clip_id,
                frame_id=frame_id,
                class_name=label,
                confidence=conf,
                bbox=bbox,
            )
            det.risk_level = assessor.assess(det)
            detections.append(det)

        # Sort by risk severity for readability in output
        severity_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH:     1,
            RiskLevel.MEDIUM:   2,
            RiskLevel.LOW:      3,
            RiskLevel.CLEAR:    4,
        }
        detections.sort(key=lambda d: severity_order[d.risk_level])
        return detections

    # ------------------------------------------------------------------
    def _process_frame(
        self,
        img_name: str,
    ) -> Optional[FrameResult]:
        path = self.cfg.paths.input_dir / img_name

        img = cv2.imread(str(path))
        if img is None:
            self.logger.warning(f"Could not read image: {path}")
            return None

        frame_id = self._frame_id(img_name)
        h, w     = img.shape[:2]
        assessor = RiskAssessor(self.cfg.risk, img_h=h, img_w=w)

        t0              = time.perf_counter()
        img_enhanced    = self.enhancer(img)
        raw_results     = self._infer(img_enhanced)
        detections      = self._parse_detections(raw_results, frame_id, assessor)
        proc_ms         = (time.perf_counter() - t0) * 1000

        result = FrameResult(
            clip_id=self.cfg.paths.clip_id,
            frame_id=frame_id,
            detections=detections,
            proc_ms=proc_ms,
        )

        # Persist per-frame JSON
        self.output_mgr.save_frame_json(result)

        # Visualise and save frame
        if self.cfg.save_vis:
            vis = self.visualiser.draw(img, result, assessor)
            self.output_mgr.save_vis(img_name, vis)

        return result

    # ------------------------------------------------------------------
    def run(self) -> List[FrameResult]:
        """Execute the full pipeline and return all frame results."""
        frames      = self._collect_frames()
        total       = len(frames)
        all_results: List[FrameResult] = []

        self.logger.info(f"Starting pipeline — {total} frames | clip: {self.cfg.paths.clip_id}")

        for idx, img_name in enumerate(frames):
            try:
                result = self._process_frame(img_name)
                if result is not None:
                    all_results.append(result)

                    if result.has_critical:
                        self.logger.warning(
                            f"[CRITICAL] Frame {result.frame_id} — "
                            f"{result.detection_count} objects detected"
                        )

            except Exception:
                self.logger.exception(f"Frame processing failed: {img_name}")
                continue

            if idx % self.cfg.log_interval == 0:
                self.logger.info(
                    f"Progress {idx + 1:>5}/{total} | "
                    f"last proc: {all_results[-1].proc_ms:.1f}ms"
                )

        self.logger.info(f"Pipeline complete — {len(all_results)}/{total} frames processed")
        self.output_mgr.save_aggregate(all_results, save_parquet=self.cfg.save_parquet)
        return all_results


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def build_config() -> PipelineConfig:
    """
    Construct pipeline configuration.
    Override individual fields here or load from a config file / env vars
    before calling ADASDetectionPipeline(cfg).
    """
    return PipelineConfig(
        model=ModelConfig(
            model_path="yolov8m.pt",
            conf_threshold=0.35,
            iou_threshold=0.50,
            input_size=1280,
            use_tta=True,
            device="cpu",       # Switch to "cuda" for GPU inference
        ),
        enhancement=EnhancementConfig(
            alpha=1.1,
            beta=10,
            clahe_clip_limit=2.0,
            clahe_tile_size=(8, 8),
        ),
        risk=RiskConfig(
            critical_proximity=0.80,
            high_proximity=0.65,
            medium_proximity=0.45,
            critical_width_frac=0.35,
            high_width_frac=0.20,
        ),
        paths=PathConfig(
            clip_id="0a948f59-0a06-41a2-8e20-ac3a39ff4d61",
            input_dir=Path("data_processed/final_fusion_v3"),
            output_dir=Path("data_processed/fused_detections_updated"),
        ),
        log_interval=20,
        save_vis=True,
        save_parquet=True,
    )


def main() -> None:
    cfg      = build_config()
    pipeline = ADASDetectionPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()