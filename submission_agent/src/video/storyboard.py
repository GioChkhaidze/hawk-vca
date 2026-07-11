import base64
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


MIN_STORYBOARD_FRAMES = 8
MAX_STORYBOARD_FRAMES = 24
MAX_FRAME_BYTES = 512 * 1024
THUMBNAIL_BYTES = 16 * 16
NEAR_DUPLICATE_BITS = 8


class StoryboardError(Exception):
  pass


@dataclass(frozen=True)
class StoryboardFrame:
  frame_id: str
  data_url: str
  timestamp_seconds: float = 0.0
  kind: str = "uniform"


@dataclass(frozen=True)
class _Candidate:
  timestamp: float
  kind: str
  width: int = 640


Runner = Callable[..., subprocess.CompletedProcess[str]]


def extract_storyboard(
  video_path: Path, destination_dir: Path, max_frames: int = MAX_STORYBOARD_FRAMES,
  runner: Runner = subprocess.run,
) -> tuple[float, list[StoryboardFrame]]:
  max_frames = min(MAX_STORYBOARD_FRAMES, max(MIN_STORYBOARD_FRAMES, max_frames))
  destination_dir.mkdir(parents=True, exist_ok=True)
  duration = probe_duration(video_path, runner=runner)
  scene_times = _scene_change_times(video_path, duration, runner)
  desired = _desired_frame_count(duration, len(scene_times), max_frames)
  candidates = _candidate_pool(duration, scene_times, desired)
  extracted = []
  fingerprints: list[int] = []
  for index, candidate in enumerate(candidates):
    frame = _extract_candidate(video_path, destination_dir, candidate, index, runner)
    if frame is None:
      continue
    storyboard_frame, fingerprint = frame
    if any((fingerprint ^ previous).bit_count() <= NEAR_DUPLICATE_BITS for previous in fingerprints):
      continue
    extracted.append(storyboard_frame)
    fingerprints.append(fingerprint)

  frames = _choose_frames(extracted, desired)
  if len(frames) < MIN_STORYBOARD_FRAMES:
    raise StoryboardError(f"storyboard produced only {len(frames)} distinct frames")
  return duration, frames


def _scene_change_times(video_path: Path, duration: float, runner: Runner) -> list[float]:
  completed = _run(runner, [
    "ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(video_path), "-an",
    "-vf", "select='gt(scene,0.30)',showinfo", "-f", "null", "-",
  ], 25, allow_failure=True)
  values = []
  for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", completed.stderr or ""):
    timestamp = float(match.group(1))
    if 0.05 < timestamp < duration - 0.05:
      values.append(round(timestamp, 3))
  return _dedupe_timestamps(values, minimum_gap=0.35)


def _desired_frame_count(duration: float, scene_count: int, maximum: int) -> int:
  if duration <= 10:
    target = 12
  elif duration <= 30:
    target = 16
  elif duration <= 60:
    target = 20
  elif duration <= 120:
    target = 22
  else:
    target = 24
  if scene_count <= 1:
    target = min(target, 12 if duration <= 30 else 16)
  elif scene_count >= 8:
    target = 24
  return min(maximum, max(MIN_STORYBOARD_FRAMES, target))


def _candidate_pool(duration: float, scene_times: list[float], desired: int) -> list[_Candidate]:
  uniform_count = desired + 8
  candidates = [
    _Candidate(duration * (index + 0.5) / uniform_count, "uniform")
    for index in range(uniform_count)
  ]
  selected_scenes = _spread(scene_times, min(8, max(2, desired // 3)))
  candidates.extend(_Candidate(timestamp, "scene") for timestamp in selected_scenes)

  burst_anchors = _spread(scene_times, 2) if scene_times else [duration / 3, duration * 2 / 3]
  burst_offset = min(0.35, max(0.12, duration / 240))
  for anchor in burst_anchors:
    for offset in (-burst_offset, burst_offset):
      candidates.append(_Candidate(min(duration - 0.02, max(0.02, anchor + offset)), "burst"))

  key_count = 2 if duration <= 30 else 3 if duration <= 60 else 4
  key_anchors = _spread(scene_times, key_count) if scene_times else [
    duration * (index + 1) / (key_count + 1) for index in range(key_count)
  ]
  candidates.extend(_Candidate(timestamp, "key", 960) for timestamp in key_anchors)

  preferred = {"uniform": 0, "burst": 1, "scene": 2, "key": 3}
  by_millisecond: dict[int, _Candidate] = {}
  for candidate in candidates:
    millisecond = int(round(candidate.timestamp * 1_000))
    current = by_millisecond.get(millisecond)
    if current is None or preferred[candidate.kind] > preferred[current.kind]:
      by_millisecond[millisecond] = candidate
  return sorted(by_millisecond.values(), key=lambda item: item.timestamp)


def _extract_candidate(
  video_path: Path, destination_dir: Path, candidate: _Candidate, index: int, runner: Runner,
) -> tuple[StoryboardFrame, int] | None:
  image_path = destination_dir / f"candidate_{index:03d}.jpg"
  gray_path = destination_dir / f"candidate_{index:03d}.gray"
  filter_graph = (
    f"[0:v]split=2[main][thumb];"
    f"[main]scale={candidate.width}:-2:force_original_aspect_ratio=decrease[out];"
    "[thumb]scale=16:16,format=gray[gray]"
  )
  completed = _run(runner, [
    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss",
    f"{candidate.timestamp:.3f}", "-i", str(video_path), "-filter_complex", filter_graph,
    "-map", "[out]", "-frames:v", "1", "-q:v", "6", "-y", str(image_path),
    "-map", "[gray]", "-frames:v", "1", "-f", "rawvideo", "-y", str(gray_path),
  ], 12, allow_failure=True)
  if completed.returncode != 0 or not image_path.exists() or not gray_path.exists():
    return None
  data = image_path.read_bytes()
  gray = gray_path.read_bytes()
  if not data or len(data) > MAX_FRAME_BYTES or len(gray) != THUMBNAIL_BYTES:
    return None
  average = sum(gray) / len(gray)
  fingerprint = 0
  for value in gray:
    fingerprint = (fingerprint << 1) | int(value >= average)
  timestamp_ms = int(round(candidate.timestamp * 1_000))
  encoded = base64.b64encode(data).decode("ascii")
  return (
    StoryboardFrame(
      f"t{timestamp_ms:06d}", f"data:image/jpeg;base64,{encoded}",
      candidate.timestamp, candidate.kind,
    ),
    fingerprint,
  )


def _choose_frames(frames: list[StoryboardFrame], desired: int) -> list[StoryboardFrame]:
  if len(frames) <= desired:
    return sorted(frames, key=lambda item: item.timestamp_seconds)
  priority = {"key": 0, "scene": 1, "burst": 2, "uniform": 3}
  ranked = sorted(frames, key=lambda item: (priority[item.kind], item.timestamp_seconds))
  selected = ranked[:desired]
  return sorted(selected, key=lambda item: item.timestamp_seconds)


def _spread(values: list[float], limit: int) -> list[float]:
  if not values or limit <= 0:
    return []
  if len(values) <= limit:
    return list(values)
  if limit == 1:
    return [values[len(values) // 2]]
  indexes = {round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)}
  return [values[index] for index in sorted(indexes)]


def _dedupe_timestamps(values: list[float], minimum_gap: float) -> list[float]:
  result = []
  for value in sorted(set(values)):
    if not result or value - result[-1] >= minimum_gap:
      result.append(value)
  return result


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
