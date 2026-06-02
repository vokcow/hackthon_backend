#!/usr/bin/env bash
# Fix push after migrating weights to Git LFS (rewrites history → needs force-with-lease once).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=lfs_path.sh
source "$(dirname "$0")/lfs_path.sh"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Install Git LFS first: brew install git-lfs && git lfs install"
  exit 1
fi

git lfs install
git lfs pull

if ! git lfs ls-files | grep -q 'weights/LibreYOLONASn-pose.pt'; then
  echo "weights/LibreYOLONASn-pose.pt is not tracked by LFS. Run:"
  echo "  git lfs migrate import --include='weights/*.pt' --everything --yes"
  exit 1
fi

git config http.postBuffer 524288000

echo "Pushing (force-with-lease required after LFS migrate rewrites commits)..."
git push --force-with-lease origin main

echo "Done. Clone elsewhere with: git lfs install && git clone ... && git lfs pull"
