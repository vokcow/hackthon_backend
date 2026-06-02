# libreyolo requires Python >=3.10
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Git metadata + weights (pointers until lfs pull)
COPY .gitattributes ./
COPY .git ./.git
COPY weights ./weights

# Materialize LFS (Railway does not pull LFS before docker build by default)
RUN git lfs pull \
    && SIZE=$(wc -c < weights/LibreYOLONASn-pose.pt | tr -d ' ') \
    && if [ "$SIZE" -lt 1000000 ]; then \
         echo "FATAL: weights/LibreYOLONASn-pose.pt is ${SIZE} bytes — LFS pull failed."; \
         echo "Ensure the repo is cloned with LFS or run scripts/railway_prepare.sh before build."; \
         exit 1; \
       fi \
    && echo "Weights ready: ${SIZE} bytes" \
    && rm -rf .git

COPY app ./app

ENV POSE_MODEL=/app/weights/LibreYOLONASn-pose.pt \
    OUTPUT_DIR=/tmp/pose_outputs \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
