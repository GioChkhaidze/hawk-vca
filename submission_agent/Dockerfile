FROM python:3.11-slim

LABEL org.opencontainers.image.title="Hawk VCA" \
      org.opencontainers.image.description="Layered video perception and style rendering for captions in four modes." \
      org.opencontainers.image.source="https://github.com/GioChkhaidze/hawk-vca" \
      org.opencontainers.image.version="v3" \
      org.opencontainers.image.licenses="MIT"

ARG CAPTION_PROXY_URL=
ARG CAPTION_PROXY_ACCESS_ID=
ARG CAPTION_PROXY_TIMEOUT_SECONDS=90

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CAPTION_PROXY_URL=${CAPTION_PROXY_URL}
ENV CAPTION_PROXY_ACCESS_ID=${CAPTION_PROXY_ACCESS_ID}
ENV CAPTION_PROXY_TIMEOUT_SECONDS=${CAPTION_PROXY_TIMEOUT_SECONDS}

WORKDIR /app

COPY src/ ./src/

CMD ["python", "src/main.py"]
