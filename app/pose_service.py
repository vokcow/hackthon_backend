"""LibreYOLO YOLO-NAS pose inference (CPU, single person)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.drawing import draw_single_person
from app.torch_compat import patch_libreyolo_weight_loading

patch_libreyolo_weight_loading()

logger = logging.getLogger(__name__)

_model: Any | None = None


@dataclass
class PersonOverlay:
    box_xyxy: np.ndarray | None
    keypoints_xy: np.ndarray
    keypoint_conf: np.ndarray | None


@dataclass
class VideoProcessStats:
    frames_total: int
    frames_inferred: int
    persons_detected_max: int
    elapsed_seconds: float
    effective_fps: float


def get_model():
    """Load LibreYOLO pose model once (singleton)."""
    global _model
    if _model is None:
        from libreyolo import LibreYOLO  # noqa: E402 — after torch_compat patch

        logger.info("Loading pose model %s on %s", settings.model_name, settings.device)
        t0 = time.perf_counter()
        _model = LibreYOLO(settings.model_name, device=settings.device)
        logger.info("Model loaded in %.1fs", time.perf_counter() - t0)
    return _model


def warmup() -> None:
    """Run a dummy inference so the first request is faster."""
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    infer_frame(dummy)


def _select_best_person(result) -> int | None:
    if len(result) == 0:
        return None
    conf = result.boxes.conf
    if conf is None or len(conf) == 0:
        return 0
    return int(conf.argmax().item())


def _result_to_overlay(result, person_idx: int) -> PersonOverlay:
    kpts = result.keypoints
    xy = kpts.xy[person_idx].cpu().numpy()
    kpt_conf = None
    if kpts.conf is not None:
        kpt_conf = kpts.conf[person_idx].cpu().numpy()
    box = result.boxes.xyxy[person_idx].cpu().numpy()
    return PersonOverlay(box_xyxy=box, keypoints_xy=xy, keypoint_conf=kpt_conf)


def infer_frame(frame_bgr: np.ndarray) -> PersonOverlay | None:
    """Run pose on one BGR frame; return highest-confidence person only."""
    model = get_model()
    result = model(
        frame_bgr,
        color_format="bgr",
        conf=settings.conf_threshold,
        iou=settings.iou_threshold,
        device=settings.device,
    )
    idx = _select_best_person(result)
    if idx is None:
        return None
    return _result_to_overlay(result, idx)


def process_video_file(
    input_path: Path,
    output_path: Path,
    *,
    vid_stride: int | None = None,
) -> VideoProcessStats:
    """
    Read a video, infer pose every `vid_stride` frames on CPU,
    draw skeleton for one person, write annotated MP4.
    """
    stride = vid_stride if vid_stride is not None else settings.vid_stride
    stride = max(1, int(stride))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")

    last_overlay: PersonOverlay | None = None
    frame_idx = 0
    frames_inferred = 0
    persons_max = 0
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                overlay = infer_frame(frame)
                if overlay is not None:
                    last_overlay = overlay
                    frames_inferred += 1
                    persons_max = max(persons_max, 1)

            annotated = frame
            if last_overlay is not None:
                annotated = draw_single_person(
                    frame,
                    last_overlay.keypoints_xy,
                    box_xyxy=last_overlay.box_xyxy,
                    keypoint_conf=last_overlay.keypoint_conf,
                    min_conf=settings.keypoint_min_conf,
                )

            writer.write(annotated)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t0
    effective_fps = frame_idx / elapsed if elapsed > 0 else 0.0

    return VideoProcessStats(
        frames_total=frame_idx,
        frames_inferred=frames_inferred,
        persons_detected_max=persons_max,
        elapsed_seconds=round(elapsed, 3),
        effective_fps=round(effective_fps, 3),
    )
