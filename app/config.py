"""Application settings (CPU-only pose pipeline)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("POSE_MODEL", "LibreYOLONASn-pose.pt")
    device: str = "cpu"
    conf_threshold: float = float(os.getenv("POSE_CONF", "0.25"))
    iou_threshold: float = float(os.getenv("POSE_IOU", "0.45"))
    vid_stride: int = int(os.getenv("POSE_VID_STRIDE", "3"))
    keypoint_min_conf: float = float(os.getenv("POSE_KPT_MIN_CONF", "0.3"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
    output_dir: str = os.getenv("OUTPUT_DIR", "/tmp/pose_outputs")


settings = Settings()
