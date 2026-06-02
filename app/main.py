"""FastAPI service: video in → annotated video with pose keypoints out."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.pose_service import get_model, process_video_file, warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}

app = FastAPI(
    title="Pose Video API",
    description=(
        "CPU-only YOLO-NAS pose estimation via LibreYOLO. "
        "Upload a video; receive an annotated MP4 with COCO-17 keypoints "
        "for the single highest-confidence person."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        get_model()
        warmup()
        logger.info("Pose model warmed up")
    except Exception as exc:
        logger.warning("Startup warmup failed (first request may be slow): %s", exc)


@app.get("/health")
def health() -> dict:
    model_path = Path(settings.model_name)
    model_exists = model_path.is_file()
    return {
        "status": "ok",
        "model": settings.model_name,
        "model_exists": model_exists,
        "model_size_mb": round(model_path.stat().st_size / 1e6, 1) if model_exists else None,
        "device": settings.device,
        "vid_stride_default": settings.vid_stride,
    }


@app.post(
    "/api/v1/pose/video",
    response_class=FileResponse,
    summary="Annotate uploaded video with pose keypoints",
)
async def pose_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Input video file"),
    vid_stride: int = Query(
        default=settings.vid_stride,
        ge=1,
        le=30,
        description="Run inference every N frames (higher = faster, lower FPS of updates)",
    ),
    return_stats_header: bool = Query(
        default=False,
        description="If true, also return processing stats as JSON in X-Process-Stats header",
    ),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format {suffix}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    job_id = uuid.uuid4().hex
    work_dir = Path(settings.output_dir) / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / f"input{suffix}"
    output_path = work_dir / "output_annotated.mp4"

    try:
        size = 0
        max_bytes = settings.max_upload_mb * 1024 * 1024
        with input_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_mb} MB limit",
                    )
                out.write(chunk)

        stats = process_video_file(input_path, output_path, vid_stride=vid_stride)
        if not output_path.is_file():
            raise HTTPException(status_code=500, detail="Failed to produce output video")

        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)

        headers = {
            "X-Frames-Total": str(stats.frames_total),
            "X-Frames-Inferred": str(stats.frames_inferred),
            "X-Elapsed-Seconds": str(stats.elapsed_seconds),
            "X-Effective-Fps": str(stats.effective_fps),
        }
        if return_stats_header:
            import json

            headers["X-Process-Stats"] = json.dumps(
                {
                    "frames_total": stats.frames_total,
                    "frames_inferred": stats.frames_inferred,
                    "persons_detected_max": stats.persons_detected_max,
                    "elapsed_seconds": stats.elapsed_seconds,
                    "effective_fps": stats.effective_fps,
                    "vid_stride": vid_stride,
                }
            )

        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"pose_{Path(file.filename).stem}.mp4",
            headers=headers,
        )
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except ValueError as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.exception("Video processing failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "pose-video-api",
            "docs": "/docs",
            "health": "/health",
            "process_video": "POST /api/v1/pose/video",
        }
    )
