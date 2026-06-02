# Model weights (Git LFS)

This directory stores the **full** pose checkpoint used in production:

- `LibreYOLONASn-pose.pt` — YOLO-NAS nano pose (COCO-17 keypoints)

## Clone with LFS

```bash
git lfs install
git clone <repo-url>
cd <repo>
git lfs pull
```

If you already cloned without LFS:

```bash
git lfs install
git lfs pull
```

Verify the file is real (not a tiny pointer stub):

```bash
ls -lh weights/LibreYOLONASn-pose.pt   # expect ~tens of MB, not ~130 bytes
```

## First-time setup (maintainers)

```bash
brew install git-lfs   # macOS; see https://git-lfs.com on Linux/Windows
git lfs install
git add .gitattributes weights/LibreYOLONASn-pose.pt
git commit -m "Track pose weights with Git LFS"
git push
```

Enable **Git LFS** in your GitHub repository settings if prompted.
