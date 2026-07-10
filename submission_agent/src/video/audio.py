import base64
import audioop
import io
import json
import math
import subprocess
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import webrtcvad


SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2
FRAME_MILLISECONDS = 30
FRAME_BYTES = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * FRAME_MILLISECONDS // 1_000
MAX_AUDIO_SECONDS = 130.0
MAX_PCM_BYTES = int(MAX_AUDIO_SECONDS * SAMPLE_RATE * SAMPLE_WIDTH_BYTES)
MAX_DECODED_BYTES = MAX_PCM_BYTES * 2
MIN_AUDIO_SECONDS = 1.5
MIN_SPEECH_SECONDS = 0.6
MIN_SPEECH_RATIO = 0.005
MAX_SPEECH_RATIO = 0.98
WINDOW_FRAMES = 40
MIN_WINDOW_SPEECH_RATIO = 0.5
MIN_STEREO_NEAR_MONO_RATIO = 0.40
NEAR_MONO_SIDE_RATIO = 0.20
ACTIVE_RMS = 100
MIN_DYNAMIC_ENVELOPE_CV = 0.45
MIN_MID_SIDE_VAD_DELTA = 0.15
MIN_MID_SIDE_ENERGY_RATIO = 1.8
MAX_PROBE_SECONDS = 8.0


class AudioError(Exception):
  pass


@dataclass(frozen=True)
class SpeechEvidence:
  audio_data_url: str
  duration_seconds: float
  speech_seconds: float
  speech_ratio: float
  near_mono_ratio: float = 1.0
  envelope_cv: float = 0.0


def collect_speech_evidence(
  video_path: Path, timeout_seconds: float,
) -> SpeechEvidence | None:
  """Return bounded mono audio only when a conservative local VAD finds likely speech."""
  if timeout_seconds <= 0:
    raise AudioError("audio timeout must be positive")
  deadline = time.monotonic() + timeout_seconds
  metadata = _probe_audio_metadata(video_path, deadline)
  if metadata is None or metadata[0] < MIN_AUDIO_SECONDS:
    return None
  duration, source_channels = metadata
  if duration > MAX_AUDIO_SECONDS:
    raise AudioError("audio duration exceeds the supported limit")

  output_channels = 2 if source_channels >= 2 else 1
  decoded = _extract_pcm(video_path, duration, output_channels, deadline)
  if not decoded or len(decoded) > MAX_DECODED_BYTES:
    raise AudioError("decoded audio is empty or oversized")
  near_mono_ratio = 1.0
  if output_channels == 2:
    pcm, near_mono_ratio = _downmix_stereo(decoded)
  else:
    pcm = decoded
  if len(pcm) > MAX_PCM_BYTES:
    raise AudioError("decoded audio is oversized")
  usable_bytes = len(pcm) - (len(pcm) % FRAME_BYTES)
  if usable_bytes < FRAME_BYTES:
    return None
  pcm = pcm[:usable_bytes]

  speech_seconds, speech_ratio, sustained, envelope_cv, speech_flags = _measure_speech(pcm)
  minimum_seconds = max(MIN_SPEECH_SECONDS, min(1.5, duration * 0.01))
  if (speech_seconds < minimum_seconds or speech_ratio < MIN_SPEECH_RATIO
      or not sustained):
    return None
  if speech_ratio > MAX_SPEECH_RATIO and envelope_cv < MIN_DYNAMIC_ENVELOPE_CV:
    return None
  if (output_channels == 2 and near_mono_ratio < MIN_STEREO_NEAR_MONO_RATIO
      and not _has_dominant_stereo_speech(
        decoded[:usable_bytes * 2], speech_flags, speech_ratio, envelope_cv,
      )):
    return None

  wav_bytes = _wav_bytes(pcm)
  encoded = base64.b64encode(wav_bytes).decode("ascii")
  return SpeechEvidence(
    audio_data_url=f"data:audio/wav;base64,{encoded}",
    duration_seconds=duration,
    speech_seconds=speech_seconds,
    speech_ratio=speech_ratio,
    near_mono_ratio=near_mono_ratio,
    envelope_cv=envelope_cv,
  )


def _probe_audio_metadata(video_path: Path, deadline: float) -> tuple[float, int] | None:
  completed = _run([
    "ffprobe", "-v", "error", "-select_streams", "a:0",
    "-show_entries", "stream=codec_type,channels:format=duration", "-of", "json", str(video_path),
  ], min(MAX_PROBE_SECONDS, _remaining(deadline)), text=True)
  try:
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
      return None
    duration = float(payload["format"]["duration"])
    channels = int(streams[0]["channels"])
  except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise AudioError("ffprobe returned invalid audio metadata") from exc
  if not 0.1 <= duration <= 600:
    raise AudioError("audio duration is outside the supported range")
  if channels < 1 or channels > 32:
    raise AudioError("audio channel count is outside the supported range")
  return duration, channels


def _extract_pcm(video_path: Path, duration: float, channels: int, deadline: float) -> bytes:
  completed = _run([
    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
    "-map", "0:a:0", "-vn", "-t", f"{duration:.3f}", "-ac", str(channels), "-ar", str(SAMPLE_RATE),
    "-f", "s16le", "pipe:1",
  ], _remaining(deadline), text=False)
  if not isinstance(completed.stdout, bytes):
    raise AudioError("ffmpeg returned invalid audio data")
  return completed.stdout


def _downmix_stereo(stereo_pcm: bytes) -> tuple[bytes, float]:
  if len(stereo_pcm) % (SAMPLE_WIDTH_BYTES * 2):
    raise AudioError("ffmpeg returned misaligned stereo audio")
  frame_bytes = FRAME_BYTES * 2
  active_frames = 0
  near_mono_frames = 0
  for offset in range(0, len(stereo_pcm) - frame_bytes + 1, frame_bytes):
    frame = stereo_pcm[offset:offset + frame_bytes]
    mid = audioop.tomono(frame, SAMPLE_WIDTH_BYTES, 0.5, 0.5)
    side = audioop.tomono(frame, SAMPLE_WIDTH_BYTES, 0.5, -0.5)
    mid_rms = audioop.rms(mid, SAMPLE_WIDTH_BYTES)
    side_rms = audioop.rms(side, SAMPLE_WIDTH_BYTES)
    if max(mid_rms, side_rms) < ACTIVE_RMS:
      continue
    active_frames += 1
    if mid_rms >= ACTIVE_RMS and side_rms <= mid_rms * NEAR_MONO_SIDE_RATIO:
      near_mono_frames += 1
  ratio = near_mono_frames / active_frames if active_frames else 0.0
  return audioop.tomono(stereo_pcm, SAMPLE_WIDTH_BYTES, 0.5, 0.5), ratio


def _measure_speech(pcm: bytes) -> tuple[float, float, bool, float, list[bool]]:
  vad = webrtcvad.Vad(3)
  frames = [pcm[offset:offset + FRAME_BYTES] for offset in range(0, len(pcm), FRAME_BYTES)]
  flags = [vad.is_speech(frame, SAMPLE_RATE) for frame in frames]
  voiced = sum(flags)
  ratio = voiced / len(flags)
  window: deque[bool] = deque(maxlen=WINDOW_FRAMES)
  max_window_ratio = 0.0
  for flag in flags:
    window.append(flag)
    if len(window) == WINDOW_FRAMES:
      max_window_ratio = max(max_window_ratio, sum(window) / WINDOW_FRAMES)
  if len(flags) < WINDOW_FRAMES:
    max_window_ratio = ratio
  envelope = [audioop.rms(frame, SAMPLE_WIDTH_BYTES) for frame in frames]
  mean_rms = sum(envelope) / len(envelope)
  envelope_cv = (
    math.sqrt(sum((value - mean_rms) ** 2 for value in envelope) / len(envelope)) / mean_rms
    if mean_rms > 0 else 0.0
  )
  return (
    voiced * FRAME_MILLISECONDS / 1_000,
    ratio,
    max_window_ratio >= MIN_WINDOW_SPEECH_RATIO,
    envelope_cv,
    flags,
  )


def _has_dominant_stereo_speech(
  stereo_pcm: bytes, mid_flags: list[bool], mid_ratio: float, envelope_cv: float,
) -> bool:
  if envelope_cv < MIN_DYNAMIC_ENVELOPE_CV:
    return False
  side_pcm = audioop.tomono(stereo_pcm, SAMPLE_WIDTH_BYTES, 0.5, -0.5)
  _, side_ratio, _, _, _ = _measure_speech(side_pcm)
  if mid_ratio - side_ratio < MIN_MID_SIDE_VAD_DELTA:
    return False

  mid_energy = 0
  side_energy = 0
  stereo_frame_bytes = FRAME_BYTES * 2
  for index, is_speech in enumerate(mid_flags):
    if not is_speech:
      continue
    frame = stereo_pcm[index * stereo_frame_bytes:(index + 1) * stereo_frame_bytes]
    mid = audioop.tomono(frame, SAMPLE_WIDTH_BYTES, 0.5, 0.5)
    side = audioop.tomono(frame, SAMPLE_WIDTH_BYTES, 0.5, -0.5)
    mid_energy += audioop.rms(mid, SAMPLE_WIDTH_BYTES) ** 2
    side_energy += audioop.rms(side, SAMPLE_WIDTH_BYTES) ** 2
  if mid_energy <= 0:
    return False
  energy_ratio = math.sqrt(mid_energy / max(1, side_energy))
  return energy_ratio >= MIN_MID_SIDE_ENERGY_RATIO


def _wav_bytes(pcm: bytes) -> bytes:
  output = io.BytesIO()
  with wave.open(output, "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
    wav_file.setframerate(SAMPLE_RATE)
    wav_file.writeframes(pcm)
  return output.getvalue()


def _remaining(deadline: float) -> float:
  remaining = deadline - time.monotonic()
  if remaining < 1.0:
    raise AudioError("audio processing timed out")
  return remaining


def _run(
  command: list[str], timeout: float, *, text: bool,
) -> subprocess.CompletedProcess:
  try:
    completed = subprocess.run(
      command, capture_output=True, text=text, timeout=timeout, check=False,
    )
  except (OSError, subprocess.SubprocessError) as exc:
    raise AudioError(f"media command failed: {exc}") from exc
  if completed.returncode != 0:
    raw_error = completed.stderr or "media command failed"
    error = raw_error if isinstance(raw_error, str) else raw_error.decode("utf-8", errors="replace")
    raise AudioError(error.strip()[:240])
  return completed
