import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


MIN_STORYBOARD_FRAMES = 8
MAX_STORYBOARD_FRAMES = 16
MAX_FRAME_BYTES = 512 * 1024


class StoryboardError(Exception):
  pass


@dataclass(frozen=True)
class StoryboardFrame:
  frame_id: str
  data_url: str


Runner = Callable[..., subprocess.CompletedProcess[str]]


def extract_storyboard(
  video_path: Path, destination_dir: Path, max_frames: int = MAX_STORYBOARD_FRAMES,
  runner: Runner = subprocess.run,
) -> tuple[float, list[StoryboardFrame]]:
  max_frames = min(MAX_STORYBOARD_FRAMES, max(MIN_STORYBOARD_FRAMES, max_frames))
  destination_dir.mkdir(parents=True, exist_ok=True)
  duration = probe_duration(video_path, runner=runner)

  uniform_count = max_frames - 4
  interval = max(duration / uniform_count, 0.25)
  _run(runner, [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
    "-vf", f"fps=1/{interval:.6f},scale=640:-2:force_original_aspect_ratio=decrease",
    "-frames:v", str(uniform_count), "-strict", "unofficial", "-q:v", "6",
    str(destination_dir / "uniform_%03d.jpg"),
  ], 25)

  scene_slots = min(2, max_frames - uniform_count)
  if scene_slots:
    _run(runner, [
      "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
      "-vf", "select=gt(scene\,0.32),scale=640:-2:force_original_aspect_ratio=decrease",
      "-vsync", "vfr", "-frames:v", str(scene_slots), "-strict", "unofficial", "-q:v", "6",
      str(destination_dir / "scene_%03d.jpg"),
    ], 25, allow_failure=True)

  burst_slots = max_frames - uniform_count - scene_slots
  for index, fraction in enumerate((0.33, 0.66)[:burst_slots], start=1):
    _run(runner, [
      "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{duration * fraction:.3f}",
      "-i", str(video_path), "-vf", "fps=4,scale=640:-2:force_original_aspect_ratio=decrease",
      "-frames:v", "1", "-strict", "unofficial", "-q:v", "6",
      str(destination_dir / f"burst_{index:03d}.jpg"),
    ], 15, allow_failure=True)

  paths = sorted(destination_dir.glob("uniform_*.jpg"))
  paths += sorted(destination_dir.glob("scene_*.jpg"))
  paths += sorted(destination_dir.glob("burst_*.jpg"))
  if len(paths) < MIN_STORYBOARD_FRAMES:
    raise StoryboardError(f"storyboard produced only {len(paths)} frames")

  frames = []
  for index, path in enumerate(paths[:max_frames], start=1):
    data = path.read_bytes()
    if not data or len(data) > MAX_FRAME_BYTES:
      raise StoryboardError("storyboard frame is empty or oversized")
    encoded = base64.b64encode(data).decode("ascii")
    frames.append(StoryboardFrame(f"f{index:02d}", f"data:image/jpeg;base64,{encoded}"))
  return duration, frames


def probe_duration(video_path: Path, runner: Runner = subprocess.run) -> float:
  completed = _run(runner, [
    "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video_path),
  ], 12)
  try:
    duration = float(json.loads(completed.stdout)["format"]["duration"])
  except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise StoryboardError("ffprobe returned an invalid duration") from exc
  if not 0.1 <= duration <= 600:
    raise StoryboardError("video duration is outside the supported range")
  return duration


def _run(
  runner: Runner, command: list[str], timeout: float, allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
  try:
    completed = runner(command, capture_output=True, text=True, timeout=timeout, check=False)
  except (OSError, subprocess.SubprocessError) as exc:
    if allow_failure:
      return subprocess.CompletedProcess(command, 1, "", str(exc))
    raise StoryboardError(f"media command failed: {exc}") from exc
  if completed.returncode != 0 and not allow_failure:
    error = (completed.stderr or "media command failed").strip()[:240]
    raise StoryboardError(error)
  return completed
