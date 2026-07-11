FROM python:3.11-slim

LABEL org.opencontainers.image.title="Hawk VCA" \
      org.opencontainers.image.description="Video captioning agent for AMD ACT II." \
      org.opencontainers.image.source="https://github.com/GioChkhaidze/hawk-vca" \
      org.opencontainers.image.version="v8.1" \
      org.opencontainers.image.licenses="MIT"

ARG CAPTION_PROXY_URL=
ARG CAPTION_PROXY_ACCESS_ID=
ARG CAPTION_PROXY_TIMEOUT_SECONDS=135

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CAPTION_PROXY_URL=${CAPTION_PROXY_URL}
ENV CAPTION_PROXY_ACCESS_ID=${CAPTION_PROXY_ACCESS_ID}
ENV CAPTION_PROXY_TIMEOUT_SECONDS=${CAPTION_PROXY_TIMEOUT_SECONDS}
ENV STORYBOARD_PERCEPTION_ENABLED=true
ENV STORYBOARD_MAX_FRAMES=16

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/main.py"]
