import subprocess
from pathlib import Path
from typing import Callable


ANALYSIS_WIDTH = 854
ANALYSIS_HEIGHT = 480
ANALYSIS_MAX_RATE = "700k"
ANALYSIS_BUFFER_SIZE = "1400k"


class AnalysisVideoError(Exception):
  pass


Runner = Callable[..., subprocess.CompletedProcess[str]]


def prepare_analysis_video(
  source_path: Path, destination_dir: Path, timeout_seconds: float, max_bytes: int,
  runner: Runner = subprocess.run,
) -> Path:
  """Return the original video when bounded, otherwise make a silent 480p analysis copy."""
  if timeout_seconds <= 0:
    raise AnalysisVideoError("analysis-video timeout must be positive")
  try:
    source_size = source_path.stat().st_size
  except OSError as exc:
    raise AnalysisVideoError(f"analysis-video source is unavailable: {exc}") from exc
  if 0 < source_size <= max_bytes:
    return source_path

  destination_dir.mkdir(parents=True, exist_ok=True)
  output_path = destination_dir / "native-analysis.mp4"
  command = [
    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source_path),
    "-map", "0:v:0", "-an",
    "-vf", (
      f"scale={ANALYSIS_WIDTH}:{ANALYSIS_HEIGHT}:force_original_aspect_ratio=decrease,"
      "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    ),
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
    "-maxrate", ANALYSIS_MAX_RATE, "-bufsize", ANALYSIS_BUFFER_SIZE,
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(output_path),
  ]
  try:
    completed = runner(
      command, capture_output=True, text=True, timeout=timeout_seconds, check=False,
    )
  except (OSError, subprocess.SubprocessError) as exc:
    raise AnalysisVideoError(f"analysis-video transcode failed: {exc}") from exc
  if completed.returncode != 0:
    error = (completed.stderr or "ffmpeg failed").strip()[:240]
    raise AnalysisVideoError(f"analysis-video transcode failed: {error}")
  try:
    output_size = output_path.stat().st_size
  except OSError as exc:
    raise AnalysisVideoError(f"analysis-video output is unavailable: {exc}") from exc
  if output_size <= 0 or output_size > max_bytes:
    output_path.unlink(missing_ok=True)
    raise AnalysisVideoError("analysis-video output is empty or oversized")
  return output_path
