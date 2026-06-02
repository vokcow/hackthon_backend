"""Draw COCO-17 pose skeleton on BGR frames (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

# COCO 17 keypoint pairs for skeleton lines
SKELETON: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

POINT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 200, 255)
BOX_COLOR = (255, 128, 0)


def draw_single_person(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    *,
    box_xyxy: np.ndarray | None = None,
    keypoint_conf: np.ndarray | None = None,
    min_conf: float = 0.3,
) -> np.ndarray:
    """Overlay one person's bbox, keypoints, and skeleton."""
    out = frame.copy()
    kpts = np.asarray(keypoints_xy, dtype=np.float32)

    if box_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)

    conf = None
    if keypoint_conf is not None:
        conf = np.asarray(keypoint_conf, dtype=np.float32)

    for a, b in SKELETON:
        if a >= len(kpts) or b >= len(kpts):
            continue
        if conf is not None:
            if conf[a] < min_conf or conf[b] < min_conf:
                continue
        pa = (int(kpts[a, 0]), int(kpts[a, 1]))
        pb = (int(kpts[b, 0]), int(kpts[b, 1]))
        if pa[0] <= 0 and pa[1] <= 0:
            continue
        if pb[0] <= 0 and pb[1] <= 0:
            continue
        cv2.line(out, pa, pb, LINE_COLOR, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(kpts):
        if conf is not None and conf[i] < min_conf:
            continue
        if x <= 0 and y <= 0:
            continue
        cv2.circle(out, (int(x), int(y)), 4, POINT_COLOR, -1, cv2.LINE_AA)

    return out
