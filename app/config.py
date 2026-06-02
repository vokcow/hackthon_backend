"""Application settings (CPU-only pose pipeline)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_WEIGHT = PROJECT_ROOT / "weights" / "LibreYOLONASn-pose.pt"
REMOTE_WEIGHT_NAME = "LibreYOLONASn-pose.pt"


def resolve_model_path() -> str:
    """
    Prefer explicit POSE_MODEL, then bundled weights/ (Git LFS in repo),
    else LibreYOLO auto-download by checkpoint name.
    """
    explicit = os.getenv("POSE_MODEL")
    if explicit:
        return explicit
    if BUNDLED_WEIGHT.is_file():
        return str(BUNDLED_WEIGHT)
    return REMOTE_WEIGHT_NAME


@dataclass(frozen=True)
class Settings:
    model_name: str = resolve_model_path()
    device: str = "cpu"
    conf_threshold: float = float(os.getenv("POSE_CONF", "0.25"))
    iou_threshold: float = float(os.getenv("POSE_IOU", "0.45"))
    vid_stride: int = int(os.getenv("POSE_VID_STRIDE", "3"))
    keypoint_min_conf: float = float(os.getenv("POSE_KPT_MIN_CONF", "0.3"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
    output_dir: str = os.getenv("OUTPUT_DIR", "/tmp/pose_outputs")


settings = Settings()
