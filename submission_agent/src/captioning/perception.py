import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from config import AppConfig
from proxy_client import (
  CaptionProxyError, ProxyMetadata, perceive_storyboard_via_proxy, perceive_video_via_proxy,
  proxy_configured, transcribe_audio_via_proxy,
)
from runtime_budget import RuntimeBudget
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
AUDIO_PROCESSING_TIMEOUT_SECONDS = 25
OPTIONAL_AUDIO_RESERVE_SECONDS = 90


def perceive_video(
  config: AppConfig, *, video_url: str | None, budget: RuntimeBudget | None = None, urlopen: Any = None,
  downloader: Any = download_video, storyboard_extractor: Any = extract_storyboard,
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
      config, video_url, budget, urlopen, downloader, storyboard_extractor, on_metadata,
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
  storyboard_extractor: Any, on_metadata: Callable[[ProxyMetadata], None] | None,
) -> dict[str, Any]:
  download_timeout = _bounded_timeout(budget, DOWNLOAD_TIMEOUT_SECONDS)
  if download_timeout is None:
    print("RUN_DEADLINE_REACHED stage=storyboard_download", file=sys.stderr)
    return fallback_facts()
  try:
    with TemporaryDirectory(prefix="submission_agent_storyboard_") as temp_dir:
      root = Path(temp_dir)
      path = downloader(video_url, root, download_timeout, MAX_VIDEO_BYTES)
      facts = _request_storyboard(
        config, path, root / "frames", budget, urlopen, storyboard_extractor, on_metadata,
      )
      if facts != fallback_facts():
        return facts

      direct_facts = _request_facts(
        config, video_url=video_url, budget=budget, urlopen=urlopen, source="url",
        on_metadata=on_metadata,
      )
      if direct_facts != fallback_facts() or path.stat().st_size > MAX_BASE64_VIDEO_BYTES:
        return direct_facts
      return _request_facts(
        config, video_data=video_data_url(path), budget=budget, urlopen=urlopen, source="base64",
        on_metadata=on_metadata,
      )
  except (DownloadError, OSError, ValueError) as exc:
    print(f"STORYBOARD_DOWNLOAD_FAILED error={str(exc)[:240]!r}", file=sys.stderr)
    return _request_facts(
      config, video_url=video_url, budget=budget, urlopen=urlopen, source="url",
      on_metadata=on_metadata,
    )


def _request_storyboard(
  config: AppConfig, video_path: Path, frames_dir: Path, budget: RuntimeBudget | None, urlopen: Any,
  storyboard_extractor: Any, on_metadata: Callable[[ProxyMetadata], None] | None,
) -> dict[str, Any]:
  try:
    duration, frames = storyboard_extractor(video_path, frames_dir, config.storyboard_max_frames)
    speech_transcript = _collect_transcript(config, video_path, budget, urlopen, on_metadata)
    timeout = _bounded_timeout(budget, config.caption_proxy_timeout_seconds)
    if timeout is None:
      print("RUN_DEADLINE_REACHED stage=storyboard_perception", file=sys.stderr)
      return fallback_facts()
    kwargs = {
      "config": config,
      "duration_seconds": duration,
      "frames": [{"id": frame.frame_id, "data_url": frame.data_url} for frame in frames],
      "speech_transcript": speech_transcript,
      "video_data_url": (
        video_data_url(video_path) if video_path.stat().st_size <= MAX_BASE64_VIDEO_BYTES else None
      ),
      "timeout_seconds": timeout,
    }
    if urlopen is not None:
      kwargs["urlopen"] = urlopen
    result = perceive_storyboard_via_proxy(**kwargs)
    if on_metadata:
      on_metadata(result.metadata)
    print(
      f"PERCEPTION_PROXY_USED source=storyboard model={result.metadata.model} "
      f"passes={len(result.metadata.perception_passes)} "
      f"speech_used={str(result.metadata.speech_used).lower()} "
      f"speech_context_used={str(result.metadata.speech_context_used).lower()} "
      f"corroboration_used={str(result.metadata.corroboration_used).lower()} "
      f"reconciliation_used={str(result.metadata.reconciliation_used).lower()} "
      f"ensemble_mode={result.metadata.ensemble_mode} "
      f"policy_version={result.metadata.policy_version}",
      file=sys.stderr,
    )
    return {**result.facts, "duration_seconds": float(duration)}
  except (CaptionProxyError, StoryboardError, OSError, ValueError) as exc:
    print(f"STORYBOARD_PERCEPTION_FAILED error={str(exc)[:240]!r}", file=sys.stderr)
    return fallback_facts()


def _collect_transcript(
  config: AppConfig, video_path: Path, budget: RuntimeBudget | None, urlopen: Any,
  on_metadata: Callable[[ProxyMetadata], None] | None,
) -> str | None:
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
