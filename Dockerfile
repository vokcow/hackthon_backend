# libreyolo requires Python >=3.10
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Always download full weights (git lfs pull inside script if .git exists; else CDN)
COPY scripts/ensure_weights.py ./scripts/ensure_weights.py
RUN python scripts/ensure_weights.py

COPY app ./app

ENV POSE_MODEL=/app/weights/LibreYOLONASn-pose.pt \
    OUTPUT_DIR=/tmp/pose_outputs \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
