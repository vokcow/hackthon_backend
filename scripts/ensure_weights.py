#!/usr/bin/env python3
"""
Always materialize weights/LibreYOLONASn-pose.pt for deploy builds.

1. git lfs pull when .git is available (Railway checkout before Docker).
2. Always download the full checkpoint from Deci CDN (overwrites pointers/copies).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

MIN_BYTES = 1_000_000
WEIGHT_PATH = Path("weights/LibreYOLONASn-pose.pt")
WEIGHT_NAME = "LibreYOLONASn-pose.pt"
SIZE_CODE = "n"


def git_lfs_pull() -> bool:
    """Run git lfs pull if this is a git checkout with git-lfs installed."""
    if not Path(".git").is_dir():
        print("No .git directory — skipping git lfs pull (normal inside Docker).")
        return False

    if not shutil.which("git"):
        print("git not found — skipping git lfs pull")
        return False

    if not shutil.which("git-lfs"):
        print("git-lfs not found — skipping git lfs pull")
        return False

    print("Running: git lfs install && git lfs pull")
    subprocess.run(["git", "lfs", "install"], check=False)
    result = subprocess.run(["git", "lfs", "pull"], check=False)
    if result.returncode != 0:
        print(f"git lfs pull exited with {result.returncode}", file=sys.stderr)
        return False

    if WEIGHT_PATH.is_file():
        print(f"After git lfs pull: {WEIGHT_PATH} ({WEIGHT_PATH.stat().st_size:,} bytes)")
    return True


def download_full_weights() -> None:
    """Always fetch the full checkpoint (never trust an existing pointer or stale file)."""
    if WEIGHT_PATH.is_file():
        prev = WEIGHT_PATH.stat().st_size
        print(f"Removing existing {WEIGHT_PATH} ({prev:,} bytes) before download")
        WEIGHT_PATH.unlink()

    WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import libreyolo.models.yolonas  # noqa: F401
    from libreyolo.utils.download import download_weights

    print(f"Downloading {WEIGHT_NAME} → {WEIGHT_PATH}")
    download_weights(str(WEIGHT_PATH), SIZE_CODE)


def main() -> int:
    git_lfs_pull()

    try:
        download_full_weights()
    except Exception as exc:
        print(f"FATAL: could not download weights: {exc}", file=sys.stderr)
        return 1

    if not WEIGHT_PATH.is_file():
        print(f"FATAL: missing {WEIGHT_PATH}", file=sys.stderr)
        return 1

    size = WEIGHT_PATH.stat().st_size
    if size < MIN_BYTES:
        print(f"FATAL: {WEIGHT_PATH} is only {size} bytes", file=sys.stderr)
        return 1

    print(f"Weights ready: {WEIGHT_PATH} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
