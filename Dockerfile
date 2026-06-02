# libreyolo requires Python >=3.10
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from libreyolo import LibreYOLO; LibreYOLO('LibreYOLONASn-pose.pt')"

COPY app ./app

ENV POSE_MODEL=LibreYOLONASn-pose.pt \
    OUTPUT_DIR=/tmp/pose_outputs \
    PORT=8000

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
