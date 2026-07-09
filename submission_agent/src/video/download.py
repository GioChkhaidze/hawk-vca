import base64
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_VIDEO_EXTENSION = ".video"
CHUNK_SIZE = 1024 * 1024
VIDEO_EXTENSIONS = {
  ".avi",
  ".m4v",
  ".mkv",
  ".mov",
  ".mp4",
  ".mpeg",
  ".mpg",
  ".webm",
}


class DownloadError(Exception):
  pass


def download_video(video_url: str, destination_dir: Path, timeout_seconds: float, max_bytes: int) -> Path:
  parsed_url = urllib.parse.urlparse(video_url)
  if parsed_url.scheme not in {"http", "https"}:
    raise DownloadError("video_url must use HTTP or HTTPS")

  destination_dir.mkdir(parents=True, exist_ok=True)
  extension = _extension_from_url(parsed_url)
  final_path = destination_dir / f"video{extension}"
  temp_path = destination_dir / f".video{extension}.tmp"

  try:
    request = urllib.request.Request(video_url, headers={"User-Agent": "submission_agent/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
      status = getattr(response, "status", 200)
      if status >= 400:
        raise DownloadError(f"video download failed with HTTP {status}")

      content_length = response.headers.get("Content-Length")
      if content_length is not None and int(content_length) > max_bytes:
        raise DownloadError("video download exceeds max byte limit")

      total_bytes = 0
      with temp_path.open("wb") as output_file:
        while True:
          chunk = response.read(CHUNK_SIZE)
          if not chunk:
            break
          total_bytes += len(chunk)
          if total_bytes > max_bytes:
            raise DownloadError("video download exceeds max byte limit")
          output_file.write(chunk)

    os.replace(temp_path, final_path)
    return final_path
  except DownloadError:
    raise
  except (OSError, ValueError, urllib.error.URLError) as exc:
    raise DownloadError(f"video download failed: {exc}") from exc
  finally:
    if temp_path.exists():
      temp_path.unlink()


def video_data_url(video_path: Path) -> str:
  media_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
  encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
  return f"data:{media_type};base64,{encoded}"


def _extension_from_url(parsed_url: urllib.parse.ParseResult) -> str:
  suffix = Path(urllib.parse.unquote(parsed_url.path)).suffix.lower()
  return suffix if suffix in VIDEO_EXTENSIONS else DEFAULT_VIDEO_EXTENSION
