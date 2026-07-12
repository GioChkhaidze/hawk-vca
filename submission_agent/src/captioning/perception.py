import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from config import AppConfig
from proxy_client import (
  CaptionProxyError, ProxyMetadata, perceive_storyboard_via_proxy, perceive_video_via_proxy,
  proxy_configured, transcribe_audio_via_proxy,
)
from runtime_budget import RuntimeBudget
from video.analysis import AnalysisVideoError, prepare_analysis_video
from video.audio import AudioError, collect_speech_evidence
from video.download import DownloadError, download_video, video_data_url
from video.storyboard import StoryboardError, extract_storyboard


GENERIC_FACTS = {
  "factual_summary": "The specific subjects and actions are unclear.",
  "do_not_claim": [
    "Specific subjects, actions, settings, identities, emotions, locations, and visible text are unverified."
  ],
}
DOWNLOAD_TIMEOUT_SECONDS = 40
MAX_VIDEO_BYTES = 160 * 1024 * 1024
MAX_BASE64_VIDEO_BYTES = 48 * 1024 * 1024
# Keeps the complete request comfortably below the Worker's 72 MiB cap after base64 and JSON expansion.
MAX_NATIVE_ANALYSIS_VIDEO_BYTES = 24 * 1024 * 1024
AUDIO_PROCESSING_TIMEOUT_SECONDS = 25
OPTIONAL_AUDIO_RESERVE_SECONDS = 90
ANALYSIS_VIDEO_TIMEOUT_SECONDS = 35
FULL_ENSEMBLE_MIN_REMAINING_SECONDS = 150
POST_DOWNLOAD_MIN_REMAINING_SECONDS = 125
NATIVE_VIDEO_MIN_REMAINING_SECONDS = 90


def perceive_video(
  config: AppConfig, *, video_url: str | None, budget: RuntimeBudget | None = None, urlopen: Any = None,
  downloader: Any = download_video, storyboard_extractor: Any = extract_storyboard,
  native_video_preparer: Any = prepare_analysis_video,
  on_metadata: Callable[[ProxyMetadata], None] | None = None,
) -> dict[str, Any]:
  if not video_url:
    print("PERCEPTION_FALLBACK reason=missing_video_url", file=sys.stderr)
    return fallback_facts()
  if not proxy_configured(config):
    print("PERCEPTION_FALLBACK reason=missing_proxy_config", file=sys.stderr)
    return fallback_facts()

  if config.storyboard_perception_enabled:
    return _perceive_storyboard_first(
      config, video_url, budget, urlopen, downloader, storyboard_extractor, native_video_preparer,
      on_metadata,
    )

  direct_facts = _request_facts(
    config, video_url=video_url, budget=budget, urlopen=urlopen, source="url", on_metadata=on_metadata,
  )
  if direct_facts != fallback_facts():
    return direct_facts

  return _downloaded_video_fallback(
    config, video_url, budget, urlopen, downloader, on_metadata, direct_facts,
  )


def _perceive_storyboard_first(
  config: AppConfig, video_url: str, budget: RuntimeBudget | None, urlopen: Any, downloader: Any,
  storyboard_extractor: Any, native_video_preparer: Any,
  on_metadata: Callable[[ProxyMetadata], None] | None,
) -> dict[str, Any]:
  clip_label = _clip_label(video_url)
  if budget is not None and budget.remaining_seconds() < FULL_ENSEMBLE_MIN_REMAINING_SECONDS:
    print(
      f"PERCEPTION_FAST_PATH clip={clip_label!r} reason=remaining_budget_before_download "
      f"remaining_seconds={budget.remaining_seconds():.3f}",
      file=sys.stderr,
    )
    return _request_facts(
      config, video_url=video_url, budget=budget, urlopen=urlopen, source="deadline_url",
      on_metadata=on_metadata,
    )
  download_timeout = _bounded_timeout(budget, DOWNLOAD_TIMEOUT_SECONDS)
  if download_timeout is None:
    print("RUN_DEADLINE_REACHED stage=storyboard_download", file=sys.stderr)
    return fallback_facts()
  try:
    with TemporaryDirectory(prefix="submission_agent_storyboard_") as temp_dir:
      root = Path(temp_dir)
      download_started = time.perf_counter()
      path = downloader(video_url, root, download_timeout, MAX_VIDEO_BYTES)
      _log_media_stage(
        clip_label, "download", download_started, budget, bytes_written=path.stat().st_size,
      )
      if budget is not None and budget.remaining_seconds() < POST_DOWNLOAD_MIN_REMAINING_SECONDS:
        print(
          f"PERCEPTION_FAST_PATH clip={clip_label!r} reason=remaining_budget_after_download "
          f"remaining_seconds={budget.remaining_seconds():.3f}",
          file=sys.stderr,
        )
        return _request_facts(
          config, video_url=video_url, budget=budget, urlopen=urlopen, source="deadline_url",
          on_metadata=on_metadata,
        )
      facts = _request_storyboard(
        config, path, root / "frames", budget, urlopen, storyboard_extractor,
        native_video_preparer, on_metadata, clip_label,
      )
      if facts != fallback_facts():
        return facts

      direct_facts = _request_facts(
        config, video_url=video_url, budget=budget, urlopen=urlopen, source="url",
        on_metadata=on_metadata,
      )
      if direct_facts != fallback_facts() or path.stat().st_size > MAX_BASE64_VIDEO_BYTES:
        if direct_facts == fallback_facts():
          _log_generic_selection("storyboard_and_direct_url_failed_base64_oversized", budget)
        return direct_facts
      base64_facts = _request_facts(
        config, video_data=video_data_url(path), budget=budget, urlopen=urlopen, source="base64",
        on_metadata=on_metadata,
      )
      if base64_facts == fallback_facts():
        _log_generic_selection("storyboard_direct_url_and_base64_failed", budget)
      return base64_facts
  except (DownloadError, OSError, ValueError) as exc:
    print(f"STORYBOARD_DOWNLOAD_FAILED error={str(exc)[:240]!r}", file=sys.stderr)
    direct_facts = _request_facts(
      config, video_url=video_url, budget=budget, urlopen=urlopen, source="url",
      on_metadata=on_metadata,
    )
    if direct_facts == fallback_facts():
      _log_generic_selection("storyboard_download_and_direct_url_failed", budget)
    return direct_facts


def _log_generic_selection(reason: str, budget: RuntimeBudget | None) -> None:
  remaining = budget.remaining_seconds() if budget is not None else -1.0
  print(
    f"GENERIC_FACTS_SELECTED reason={reason} remaining_runtime_budget={remaining:.3f}",
    file=sys.stderr,
  )


def _request_storyboard(
  config: AppConfig, video_path: Path, frames_dir: Path, budget: RuntimeBudget | None, urlopen: Any,
  storyboard_extractor: Any, native_video_preparer: Any,
  on_metadata: Callable[[ProxyMetadata], None] | None, clip_label: str,
) -> dict[str, Any]:
  try:
    def prepare_storyboard() -> tuple[float, list[Any], float]:
      started = time.perf_counter()
      duration, frames = storyboard_extractor(video_path, frames_dir, config.storyboard_max_frames)
      return duration, frames, time.perf_counter() - started

    def prepare_speech() -> tuple[str | None, float]:
      started = time.perf_counter()
      transcript = _collect_transcript(config, video_path, budget, urlopen, on_metadata)
      return transcript, time.perf_counter() - started

    def prepare_native_video() -> tuple[Path | None, float]:
      started = time.perf_counter()
      path = _native_video_path(
        video_path, frames_dir.parent / "native", budget, native_video_preparer,
      )
      return path, time.perf_counter() - started

    # These operations read the same immutable download and produce independent
    # artifacts. Running them together removes avoidable serial media latency
    # while leaving clip-level and provider-level concurrency unchanged.
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="media_prepare") as pool:
      storyboard_future = pool.submit(prepare_storyboard)
      speech_future = pool.submit(prepare_speech)
      native_future = pool.submit(prepare_native_video)
      duration, frames, storyboard_seconds = storyboard_future.result()
      speech_transcript, audio_seconds = speech_future.result()
      native_video_path, native_seconds = native_future.result()

    _log_media_stage(
      clip_label, "storyboard", None, budget, duration_seconds=storyboard_seconds,
      frame_count=len(frames),
    )
    _log_media_stage(
      clip_label, "speech_gate", None, budget, duration_seconds=audio_seconds,
      transcript_used=str(bool(speech_transcript)).lower(),
    )
    _log_media_stage(
      clip_label, "native_video", None, budget, duration_seconds=native_seconds,
      native_bytes=native_video_path.stat().st_size if native_video_path else 0,
    )
    timeout = _bounded_timeout(budget, config.caption_proxy_timeout_seconds)
    if timeout is None:
      print("RUN_DEADLINE_REACHED stage=storyboard_perception", file=sys.stderr)
      return fallback_facts()
    encode_started = time.perf_counter()
    native_video_data = video_data_url(native_video_path) if native_video_path else None
    _log_media_stage(
      clip_label, "request_encoding", encode_started, budget,
      encoded_video_chars=len(native_video_data) if native_video_data else 0,
    )
    kwargs = {
      "config": config,
      "duration_seconds": duration,
      "frames": [{"id": frame.frame_id, "data_url": frame.data_url} for frame in frames],
      "speech_transcript": speech_transcript,
      "video_data_url": native_video_data,
      "timeout_seconds": timeout,
    }
    if urlopen is not None:
      kwargs["urlopen"] = urlopen
    proxy_started = time.perf_counter()
    result = perceive_storyboard_via_proxy(**kwargs)
    _log_media_stage(clip_label, "perception_proxy", proxy_started, budget)
    if on_metadata:
      on_metadata(result.metadata)
    print(
      f"PERCEPTION_PROXY_USED source=storyboard model={result.metadata.model} "
      f"passes={len(result.metadata.perception_passes)} "
      f"speech_used={str(result.metadata.speech_used).lower()} "
      f"speech_context_used={str(result.metadata.speech_context_used).lower()} "
      f"corroboration_used={str(result.metadata.corroboration_used).lower()} "
      f"reconciliation_used={str(result.metadata.reconciliation_used).lower()} "
      f"native_video_included={str(result.metadata.native_video_included).lower()} "
      f"ensemble_mode={result.metadata.ensemble_mode} "
      f"policy_version={result.metadata.policy_version}",
      file=sys.stderr,
    )
    return {**result.facts, "duration_seconds": float(duration)}
  except (CaptionProxyError, StoryboardError, OSError, ValueError) as exc:
    print(f"STORYBOARD_PERCEPTION_FAILED error={str(exc)[:240]!r}", file=sys.stderr)
    return fallback_facts()


def _native_video_path(
  video_path: Path, destination_dir: Path, budget: RuntimeBudget | None, native_video_preparer: Any,
) -> Path | None:
  if budget is not None and budget.remaining_seconds() < NATIVE_VIDEO_MIN_REMAINING_SECONDS:
    print(
      f"NATIVE_VIDEO_SOURCE kind=omitted reason=deadline "
      f"remaining_seconds={budget.remaining_seconds():.3f}",
      file=sys.stderr,
    )
    return None
  source_size = video_path.stat().st_size
  if source_size <= MAX_NATIVE_ANALYSIS_VIDEO_BYTES:
    print(f"NATIVE_VIDEO_SOURCE kind=original bytes={source_size}", file=sys.stderr)
    return video_path
  timeout = _bounded_timeout(budget, ANALYSIS_VIDEO_TIMEOUT_SECONDS, reserve_seconds=1.0)
  if timeout is None:
    print("NATIVE_VIDEO_SOURCE kind=omitted reason=deadline", file=sys.stderr)
    return None
  try:
    prepared = native_video_preparer(
      video_path, destination_dir, timeout, MAX_NATIVE_ANALYSIS_VIDEO_BYTES,
    )
    prepared_size = prepared.stat().st_size
    if prepared_size <= 0 or prepared_size > MAX_NATIVE_ANALYSIS_VIDEO_BYTES:
      raise AnalysisVideoError("analysis-video preparer returned an invalid file")
    print(
      f"NATIVE_VIDEO_SOURCE kind=transcoded source_bytes={source_size} bytes={prepared_size}",
      file=sys.stderr,
    )
    return prepared
  except (AnalysisVideoError, OSError, ValueError) as exc:
    print(f"NATIVE_VIDEO_SOURCE kind=omitted reason=transcode_failed error={str(exc)[:240]!r}", file=sys.stderr)
    return None


def _collect_transcript(
  config: AppConfig, video_path: Path, budget: RuntimeBudget | None, urlopen: Any,
  on_metadata: Callable[[ProxyMetadata], None] | None,
) -> str | None:
  if budget is not None and budget.remaining_seconds() < (
    OPTIONAL_AUDIO_RESERVE_SECONDS + AUDIO_PROCESSING_TIMEOUT_SECONDS
  ):
    print(
      f"SPEECH_GATE activated=false reason=deadline remaining_seconds="
      f"{budget.remaining_seconds():.3f}",
      file=sys.stderr,
    )
    return None
  processing_timeout = _bounded_timeout(
    budget, AUDIO_PROCESSING_TIMEOUT_SECONDS, reserve_seconds=OPTIONAL_AUDIO_RESERVE_SECONDS,
  )
  if processing_timeout is None:
    print("SPEECH_GATE activated=false reason=deadline", file=sys.stderr)
    return None
  try:
    evidence = collect_speech_evidence(video_path, processing_timeout)
  except (AudioError, OSError, ValueError) as exc:
    print(f"SPEECH_GATE activated=false reason=media_error error={str(exc)[:240]!r}", file=sys.stderr)
    return None
  if evidence is None:
    print("SPEECH_GATE activated=false reason=no_likely_speech", file=sys.stderr)
    return None

  timeout = _bounded_timeout(
    budget, config.caption_proxy_timeout_seconds, reserve_seconds=OPTIONAL_AUDIO_RESERVE_SECONDS,
  )
  if timeout is None:
    print("SPEECH_GATE activated=true transcribed=false reason=deadline", file=sys.stderr)
    return None
  try:
    kwargs = {
      "config": config,
      "audio_data_url": evidence.audio_data_url,
      "timeout_seconds": timeout,
    }
    if urlopen is not None:
      kwargs["urlopen"] = urlopen
    result = transcribe_audio_via_proxy(**kwargs)
    if on_metadata:
      on_metadata(result.metadata)
    print(
      f"SPEECH_GATE activated=true transcribed=true useful={str(bool(result.transcript)).lower()} "
      f"audio_seconds={evidence.duration_seconds:.3f} speech_seconds={evidence.speech_seconds:.3f} "
      f"speech_ratio={evidence.speech_ratio:.4f} near_mono_ratio={evidence.near_mono_ratio:.4f} "
      f"envelope_cv={evidence.envelope_cv:.4f} "
      f"model={result.metadata.model} "
      f"duration_ms={result.metadata.attempts[-1].duration_ms} "
      f"cost_usd={result.metadata.attempts[-1].cost_usd}",
      file=sys.stderr,
    )
    return result.transcript
  except (CaptionProxyError, OSError) as exc:
    print(f"SPEECH_TRANSCRIPTION_FAILED error={str(exc)[:240]!r}", file=sys.stderr)
    return None


def _clip_label(video_url: str) -> str:
  parsed = urllib.parse.urlparse(video_url)
  return Path(urllib.parse.unquote(parsed.path)).name[:96] or parsed.netloc[:96] or "video"


def _log_media_stage(
  clip_label: str, stage: str, started: float | None, budget: RuntimeBudget | None, **details: object,
) -> None:
  remaining = budget.remaining_seconds() if budget is not None else -1.0
  measured_seconds = details.pop("duration_seconds", None)
  duration_seconds = (
    float(measured_seconds) if measured_seconds is not None
    else time.perf_counter() - float(started)
  )
  suffix = " ".join(f"{key}={value}" for key, value in details.items())
  print(
    f"MEDIA_STAGE clip={clip_label!r} stage={stage} "
    f"duration_seconds={duration_seconds:.3f} "
    f"remaining_runtime_budget={remaining:.3f}" + (f" {suffix}" if suffix else ""),
    file=sys.stderr,
  )


def _downloaded_video_fallback(
  config: AppConfig, video_url: str, budget: RuntimeBudget | None, urlopen: Any, downloader: Any,
  on_metadata: Callable[[ProxyMetadata], None] | None, direct_facts: dict[str, Any],
) -> dict[str, Any]:
  download_timeout = _bounded_timeout(budget, DOWNLOAD_TIMEOUT_SECONDS)
  if download_timeout is None:
    print("RUN_DEADLINE_REACHED stage=video_download", file=sys.stderr)
    return direct_facts
  try:
    with TemporaryDirectory(prefix="submission_agent_video_") as temp_dir:
      path = downloader(video_url, Path(temp_dir), download_timeout, MAX_BASE64_VIDEO_BYTES)
      facts = _request_facts(
        config, video_data=video_data_url(path), budget=budget, urlopen=urlopen, source="base64",
        on_metadata=on_metadata,
      )
      return facts if facts != fallback_facts() else direct_facts
  except (DownloadError, OSError, ValueError) as exc:
    print(f"PERCEPTION_DOWNLOAD_FAILED error={str(exc)[:240]!r}", file=sys.stderr)
    return direct_facts


def _request_facts(
  config: AppConfig, *, budget: RuntimeBudget | None, urlopen: Any, source: str,
  video_url: str | None = None, video_data: str | None = None,
  on_metadata: Callable[[ProxyMetadata], None] | None = None,
) -> dict[str, Any]:
  timeout = _bounded_timeout(budget, config.caption_proxy_timeout_seconds)
  if timeout is None:
    print("RUN_DEADLINE_REACHED stage=perception", file=sys.stderr)
    return fallback_facts()

  try:
    kwargs = {
      "config": config,
      "video_url": video_url,
      "video_data_url": video_data,
      "timeout_seconds": timeout,
    }
    if urlopen is not None:
      kwargs["urlopen"] = urlopen
    result = perceive_video_via_proxy(**kwargs)
    if on_metadata:
      on_metadata(result.metadata)
    print(
      f"PERCEPTION_PROXY_USED source={source} model={result.metadata.model} "
      f"fallback_used={str(result.metadata.fallback_used).lower()} policy_version={result.metadata.policy_version}",
      file=sys.stderr,
    )
    return result.facts
  except (CaptionProxyError, OSError) as exc:
    print(f"PERCEPTION_PROXY_FAILED source={source} error={str(exc)[:240]!r}", file=sys.stderr)
    return fallback_facts()


def _bounded_timeout(
  budget: RuntimeBudget | None, configured_seconds: float, reserve_seconds: float = 1.0,
) -> float | None:
  return configured_seconds if budget is None else budget.request_timeout(
    configured_seconds, reserve_seconds=reserve_seconds,
  )


def fallback_facts() -> dict[str, Any]:
  return {"factual_summary": GENERIC_FACTS["factual_summary"], "do_not_claim": list(GENERIC_FACTS["do_not_claim"])}
