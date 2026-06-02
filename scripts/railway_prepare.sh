#!/usr/bin/env bash
# Run before Docker build on Railway (also wired in railway.toml buildCommand).
set -euo pipefail

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Installing git-lfs is required on the build host."
  echo "Railway: ensure buildCommand runs or use the Dockerfile LFS stage."
  exit 1
fi

git lfs install
git lfs pull

WEIGHT="weights/LibreYOLONASn-pose.pt"
if [[ ! -f "$WEIGHT" ]]; then
  echo "Missing $WEIGHT"
  exit 1
fi

SIZE=$(wc -c < "$WEIGHT" | tr -d ' ')
if [[ "$SIZE" -lt 1000000 ]]; then
  echo "ERROR: $WEIGHT is only ${SIZE} bytes (Git LFS pointer?)."
  echo "Run: git lfs install && git lfs pull"
  exit 1
fi

echo "Git LFS OK: $WEIGHT (${SIZE} bytes)"
