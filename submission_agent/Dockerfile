FROM mwader/static-ffmpeg:7.1.1@sha256:11a44711684c0b9f754c047dcd64235b8b52deab251bd0e0a86f22faa160749c AS media

FROM python:3.11-alpine@sha256:25976e9d34a0fab1f278cae931f34c8303d97bf0c0d7f85b6b4dcf641d7702a4

LABEL org.opencontainers.image.title="Hawk VCA" \
      org.opencontainers.image.description="Video captioning agent for AMD ACT II." \
      org.opencontainers.image.source="https://github.com/GioChkhaidze/hawk-vca" \
      org.opencontainers.image.version="v9.6" \
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
ENV STORYBOARD_MAX_FRAMES=24

COPY --from=media /ffmpeg /usr/local/bin/ffmpeg
COPY --from=media /ffprobe /usr/local/bin/ffprobe

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/main.py"]
