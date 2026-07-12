import base64
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


MIN_STORYBOARD_FRAMES = 1
MAX_STORYBOARD_FRAMES = 24
MAX_FRAME_BYTES = 512 * 1024
THUMBNAIL_BYTES = 16 * 16
NEAR_DUPLICATE_BITS = 8
BATCH_SAMPLE_FPS = 8
BATCH_EXTRACTION_TIMEOUT_SECONDS = 35
SEQUENTIAL_FALLBACK_BUDGET_SECONDS = 20
SEQUENTIAL_FRAME_TIMEOUT_SECONDS = 4


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
  started = time.perf_counter()
  max_frames = min(MAX_STORYBOARD_FRAMES, max(MIN_STORYBOARD_FRAMES, max_frames))
  destination_dir.mkdir(parents=True, exist_ok=True)
  probe_started = time.perf_counter()
  duration = probe_duration(video_path, runner=runner)
  probe_seconds = time.perf_counter() - probe_started
  scene_started = time.perf_counter()
  scene_times = _scene_change_times(video_path, duration, runner)
  scene_seconds = time.perf_counter() - scene_started
  desired = _desired_frame_count(duration, len(scene_times), max_frames)
  candidates = _quantize_candidates(_candidate_pool(duration, scene_times, desired), duration)
  extraction_started = time.perf_counter()
  extracted = _extract_candidates_batch(video_path, destination_dir, candidates, runner)
  extraction_mode = "batch"
  if not extracted:
    extraction_mode = "sequential_fallback"
    extracted = _extract_candidates_sequential(
      video_path, destination_dir, _fallback_candidate_plan(candidates, desired), runner,
    )
  extraction_seconds = time.perf_counter() - extraction_started

  distinct = []
  fingerprints: list[int] = []
  for storyboard_frame, fingerprint in sorted(
    extracted, key=lambda item: item[0].timestamp_seconds,
  ):
    if any((fingerprint ^ previous).bit_count() <= NEAR_DUPLICATE_BITS for previous in fingerprints):
      continue
    distinct.append(storyboard_frame)
    fingerprints.append(fingerprint)

  frames = _choose_frames(distinct, desired)
  if len(frames) < MIN_STORYBOARD_FRAMES:
    raise StoryboardError(f"storyboard produced only {len(frames)} distinct frames")
  print(
    f"STORYBOARD_TIMING duration_seconds={duration:.3f} probe_seconds={probe_seconds:.3f} "
    f"scene_seconds={scene_seconds:.3f} extraction_seconds={extraction_seconds:.3f} "
    f"total_seconds={time.perf_counter() - started:.3f} scene_count={len(scene_times)} "
    f"candidate_count={len(candidates)} distinct_count={len(distinct)} "
    f"selected_count={len(frames)} extraction_mode={extraction_mode}",
    file=sys.stderr,
  )
  return duration, frames


def _scene_change_times(video_path: Path, duration: float, runner: Runner) -> list[float]:
  completed = _run(runner, [
    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-i", str(video_path), "-an",
    "-vf", (
      "fps=4,scale=320:-2:force_original_aspect_ratio=decrease,"
      "select='gt(scene,0.30)',showinfo"
    ),
    "-f", "null", "-",
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


def _quantize_candidates(candidates: list[_Candidate], duration: float) -> list[_Candidate]:
  """Map requested times to the stable 8 fps stream extracted by one ffmpeg decode."""
  maximum_index = max(0, math.floor(duration * BATCH_SAMPLE_FPS) - 1)
  preferred = {"uniform": 0, "burst": 1, "scene": 2, "key": 3}
  by_frame_index: dict[int, _Candidate] = {}
  for candidate in candidates:
    frame_index = min(maximum_index, max(0, round(candidate.timestamp * BATCH_SAMPLE_FPS)))
    sampled = _Candidate(frame_index / BATCH_SAMPLE_FPS, candidate.kind, candidate.width)
    current = by_frame_index.get(frame_index)
    if current is None or preferred[sampled.kind] > preferred[current.kind]:
      by_frame_index[frame_index] = sampled
  return [by_frame_index[index] for index in sorted(by_frame_index)]


def _extract_candidates_batch(
  video_path: Path, destination_dir: Path, candidates: list[_Candidate], runner: Runner,
) -> list[tuple[StoryboardFrame, int]]:
  groups = [
    ("normal", 640, [candidate for candidate in candidates if candidate.width < 960]),
    ("key", 960, [candidate for candidate in candidates if candidate.width >= 960]),
  ]
  groups = [group for group in groups if group[2]]
  if not groups:
    return []

  graph = [f"[0:v]fps={BATCH_SAMPLE_FPS}"]
  if len(groups) == 1:
    graph.append(f"[{groups[0][0]}_source];")
  else:
    outputs = "".join(f"[{label}_source]" for label, _, _ in groups)
    graph.append(f",split={len(groups)}{outputs};")

  output_arguments: list[str] = []
  paths: dict[str, tuple[Path, Path]] = {}
  for label, width, group_candidates in groups:
    expression = "+".join(
      f"eq(n\\,{round(candidate.timestamp * BATCH_SAMPLE_FPS)})"
      for candidate in group_candidates
    )
    graph.extend([
      f"[{label}_source]select='{expression}',split=2[{label}_main][{label}_thumb];",
      f"[{label}_main]scale={width}:-2:force_original_aspect_ratio=decrease[{label}_out];",
      f"[{label}_thumb]scale=16:16,format=gray[{label}_gray];",
    ])
    image_pattern = destination_dir / f"batch_{label}_%03d.jpg"
    gray_path = destination_dir / f"batch_{label}.gray"
    paths[label] = image_pattern, gray_path
    output_arguments.extend([
      "-map", f"[{label}_out]", "-vsync", "0", "-q:v", "6", "-start_number", "0",
      "-y", str(image_pattern),
      "-map", f"[{label}_gray]", "-vsync", "0", "-f", "rawvideo", "-y", str(gray_path),
    ])
  command = [
    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
    "-filter_complex", "".join(graph), *output_arguments,
  ]

  completed = _run(
    runner, command, BATCH_EXTRACTION_TIMEOUT_SECONDS, allow_failure=True,
  )
  if completed.returncode != 0:
    print(
      f"STORYBOARD_BATCH_FAILED reason=ffmpeg error={(completed.stderr or '')[:240]!r}",
      file=sys.stderr,
    )
    return []

  extracted = []
  for label, _, group_candidates in groups:
    image_pattern, gray_path = paths[label]
    if not gray_path.exists():
      print(f"STORYBOARD_BATCH_FAILED reason=missing_gray group={label}", file=sys.stderr)
      return []
    gray_data = gray_path.read_bytes()
    if len(gray_data) != len(group_candidates) * THUMBNAIL_BYTES:
      print(
        f"STORYBOARD_BATCH_FAILED reason=gray_size group={label} "
        f"expected={len(group_candidates) * THUMBNAIL_BYTES} actual={len(gray_data)}",
        file=sys.stderr,
      )
      return []
    for index, candidate in enumerate(group_candidates):
      image_path = Path(str(image_pattern).replace("%03d", f"{index:03d}"))
      if not image_path.exists():
        print(
          f"STORYBOARD_BATCH_FAILED reason=missing_image group={label} index={index}",
          file=sys.stderr,
        )
        return []
      frame = _frame_from_files(
        candidate, image_path.read_bytes(),
        gray_data[index * THUMBNAIL_BYTES:(index + 1) * THUMBNAIL_BYTES],
      )
      if frame is not None:
        extracted.append(frame)
  return extracted


def _extract_candidates_sequential(
  video_path: Path, destination_dir: Path, candidates: list[_Candidate], runner: Runner,
) -> list[tuple[StoryboardFrame, int]]:
  deadline = time.monotonic() + SEQUENTIAL_FALLBACK_BUDGET_SECONDS
  extracted = []
  for index, candidate in enumerate(candidates):
    remaining = deadline - time.monotonic()
    if remaining < 1:
      break
    frame = _extract_candidate(
      video_path, destination_dir, candidate, index, runner,
      timeout=min(SEQUENTIAL_FRAME_TIMEOUT_SECONDS, remaining),
    )
    if frame is not None:
      extracted.append(frame)
  return extracted


def _fallback_candidate_plan(candidates: list[_Candidate], desired: int) -> list[_Candidate]:
  if len(candidates) <= desired:
    return candidates
  priority = {"key": 0, "scene": 1, "burst": 2, "uniform": 3}
  non_uniform = sorted(
    (candidate for candidate in candidates if candidate.kind != "uniform"),
    key=lambda candidate: (priority[candidate.kind], candidate.timestamp),
  )
  selected = non_uniform[:desired]
  remaining = desired - len(selected)
  if remaining > 0:
    uniforms = [candidate for candidate in candidates if candidate.kind == "uniform"]
    selected.extend(_spread_items(uniforms, remaining))
  return sorted(selected, key=lambda candidate: candidate.timestamp)


def _spread_items(values: list[_Candidate], limit: int) -> list[_Candidate]:
  if not values or limit <= 0:
    return []
  if len(values) <= limit:
    return values
  if limit == 1:
    return [values[len(values) // 2]]
  indexes = {round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)}
  return [values[index] for index in sorted(indexes)]


def _extract_candidate(
  video_path: Path, destination_dir: Path, candidate: _Candidate, index: int, runner: Runner,
  timeout: float = 12,
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
  ], timeout, allow_failure=True)
  if completed.returncode != 0 or not image_path.exists() or not gray_path.exists():
    return None
  return _frame_from_files(candidate, image_path.read_bytes(), gray_path.read_bytes())


def _frame_from_files(
  candidate: _Candidate, data: bytes, gray: bytes,
) -> tuple[StoryboardFrame, int] | None:
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
