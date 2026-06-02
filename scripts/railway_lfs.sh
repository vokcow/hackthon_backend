#!/usr/bin/env bash
# Railway buildCommand: always run git lfs pull on the checkout (before docker build).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x /opt/homebrew/bin/git-lfs ]]; then
  export PATH="/opt/homebrew/bin:${PATH}"
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "ERROR: git-lfs is required. Install it on the Railway build environment."
  exit 1
fi

git lfs install
git lfs pull

SIZE=$(wc -c < weights/LibreYOLONASn-pose.pt | tr -d ' ')
echo "After git lfs pull: weights/LibreYOLONASn-pose.pt = ${SIZE} bytes"
if [[ "$SIZE" -lt 1000000 ]]; then
  echo "ERROR: weights file still looks like an LFS pointer (${SIZE} bytes)."
  exit 1
fi
