import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import AppConfig


class CaptionProxyError(Exception):
  pass


EXPECTED_POLICY_VERSION = "style-spec-v9.3.1-relaxed-voice-20260711"
EXPECTED_PIPELINE_VERSION = "v9.3.1-qwen-gemini-direct-20260711"
MAX_TRANSCRIPT_CHARS = 6_000
FACT_FIELDS = (
  "factual_summary", "do_not_claim", "duration_seconds", "scene_complexity", "media_type", "events",
  "visible_text", "uncertain_details",
)


@dataclass(frozen=True)
class ProxyAttempt:
  model: str
  outcome: str
  error: str | None
  duration_ms: int
  validation_error: str | None = None
  finish_reason: str | None = None
  prompt_tokens: int | None = None
  completion_tokens: int | None = None
  reasoning_tokens: int | None = None
  audio_seconds: float | None = None
  cost_usd: float | None = None
  stage: str | None = None
  response_excerpt: str | None = None


@dataclass(frozen=True)
class ProxyMetadata:
  policy_version: str
  model: str
  fallback_used: bool
  attempts: tuple[ProxyAttempt, ...]
  source_kind: str | None = None
  perception_passes: tuple[ProxyAttempt, ...] = ()
  speech_used: bool = False
  speech_context_used: bool = False
  corroboration_used: bool = False
  reconciliation_used: bool = False
  native_video_included: bool = False
  ensemble_mode: str = "always"
  pipeline_version: str = EXPECTED_PIPELINE_VERSION


@dataclass(frozen=True)
class PerceptionProxyResult:
  facts: dict[str, Any]
  metadata: ProxyMetadata


@dataclass(frozen=True)
class StyleProxyResult:
  caption: str
  metadata: ProxyMetadata


@dataclass(frozen=True)
class TranscriptionProxyResult:
  transcript: str | None
  metadata: ProxyMetadata


def proxy_configured(config: AppConfig) -> bool:
  return bool(config.caption_proxy_url and config.caption_proxy_access_id)


def perceive_video_via_proxy(
  config: AppConfig, *, video_url: str | None = None, video_data_url: str | None = None,
  timeout_seconds: float | None = None,
  urlopen: Any = urllib.request.urlopen,
) -> PerceptionProxyResult:
  if bool(video_url) == bool(video_data_url):
    raise CaptionProxyError("exactly one video source is required")
  payload = {"video_url": video_url} if video_url else {"video_data_url": video_data_url}
  response = post_proxy(
    config, "/perceive", payload, timeout_seconds=timeout_seconds, urlopen=urlopen
  )
  normalized_facts = _normalize_facts(response.get("facts"))
  expected_source = "url" if video_url else "base64"
  return PerceptionProxyResult(normalized_facts, _proxy_metadata(response, expected_source))


def perceive_storyboard_via_proxy(
  config: AppConfig, *, duration_seconds: float, frames: list[dict[str, str]],
  speech_transcript: str | None = None, video_data_url: str | None = None,
  timeout_seconds: float | None = None, urlopen: Any = urllib.request.urlopen,
) -> PerceptionProxyResult:
  payload: dict[str, Any] = {"duration_seconds": duration_seconds, "frames": frames}
  if video_data_url:
    if not video_data_url.startswith("data:video/") or ";base64," not in video_data_url[:100]:
      raise CaptionProxyError("native video must be a video data URL")
    payload["video_data_url"] = video_data_url
  if speech_transcript:
    normalized_transcript = " ".join(speech_transcript.split())
    if not normalized_transcript or len(normalized_transcript) > MAX_TRANSCRIPT_CHARS:
      raise CaptionProxyError("speech transcript is invalid")
    payload["speech_evidence"] = {"transcript": normalized_transcript}
  response = post_proxy(
    config, "/perceive-storyboard", payload,
    timeout_seconds=timeout_seconds, urlopen=urlopen,
  )
  normalized_facts = _normalize_facts(response.get("facts"))
  return PerceptionProxyResult(normalized_facts, _proxy_metadata(response, "storyboard"))


def transcribe_audio_via_proxy(
  config: AppConfig, *, audio_data_url: str, timeout_seconds: float | None = None,
  urlopen: Any = urllib.request.urlopen,
) -> TranscriptionProxyResult:
  if not audio_data_url.startswith("data:audio/wav;base64,"):
    raise CaptionProxyError("speech audio must be a WAV data URL")
  response = post_proxy(
    config, "/transcribe", {"audio_data_url": audio_data_url},
    timeout_seconds=timeout_seconds, urlopen=urlopen,
  )
  transcript = response.get("transcript")
  if not isinstance(transcript, str):
    raise CaptionProxyError("proxy response missing transcript")
  normalized = " ".join(transcript.split())
  if len(normalized) > MAX_TRANSCRIPT_CHARS:
    raise CaptionProxyError("proxy response transcript is oversized")
  return TranscriptionProxyResult(normalized or None, _proxy_metadata(response, "audio"))


def style_caption_via_proxy(
  config: AppConfig, *, style: str, facts: dict[str, Any], timeout_seconds: float | None = None,
  avoid_captions: list[str] | None = None,
  urlopen: Any = urllib.request.urlopen,
) -> StyleProxyResult:
  payload: dict[str, Any] = {"style": style, "facts": facts}
  if avoid_captions:
    payload["avoid_captions"] = avoid_captions
  response = post_proxy(
    config, "/style", payload, timeout_seconds=timeout_seconds, urlopen=urlopen
  )
  caption = response.get("caption")
  if not isinstance(caption, str) or not caption.strip():
    raise CaptionProxyError("proxy response missing caption")
  return StyleProxyResult(caption.strip(), _proxy_metadata(response))


def post_proxy(
  config: AppConfig, path: str, payload: dict[str, Any], *, timeout_seconds: float | None = None,
  urlopen: Any = urllib.request.urlopen,
) -> dict[str, Any]:
  if not proxy_configured(config):
    raise CaptionProxyError("caption proxy is not configured")

  request = urllib.request.Request(
    proxy_url(str(config.caption_proxy_url), path),
    data=json.dumps(payload).encode("utf-8"),
    headers={
      "Authorization": f"Bearer {config.caption_proxy_access_id}",
      "Accept": "application/json",
      "Content-Type": "application/json",
      "User-Agent": "VideoCaptionAgent/1.0",
    },
    method="POST",
  )
  timeout = timeout_seconds if timeout_seconds is not None else config.caption_proxy_timeout_seconds
  if timeout <= 0:
    raise CaptionProxyError("proxy request timeout must be positive")

  try:
    with urlopen(request, timeout=timeout) as response:
      status = getattr(response, "status", 200)
      body = response.read()
  except urllib.error.HTTPError as exc:
    body = exc.read()
    raise CaptionProxyError(
      f"proxy request failed with HTTP {exc.code}: {_safe_excerpt(body, config.caption_proxy_access_id)}"
    ) from exc
  except (OSError, urllib.error.URLError) as exc:
    raise CaptionProxyError(f"proxy request failed: {exc}") from exc

  if status >= 400:
    raise CaptionProxyError(
      f"proxy request failed with HTTP {status}: {_safe_excerpt(body, config.caption_proxy_access_id)}"
    )

  try:
    decoded = json.loads(body.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise CaptionProxyError("proxy response was not valid JSON") from exc
  if not isinstance(decoded, dict):
    raise CaptionProxyError("proxy response JSON must be an object")
  return decoded


def proxy_url(base_url: str, path: str) -> str:
  return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _proxy_metadata(response: dict[str, Any], expected_source: str | None = None) -> ProxyMetadata:
  policy_version = response.get("policy_version")
  if policy_version != EXPECTED_POLICY_VERSION:
    raise CaptionProxyError(
      f"proxy policy mismatch: expected {EXPECTED_POLICY_VERSION}, received {policy_version or 'missing'}"
    )
  pipeline_version = response.get("pipeline_version")
  if pipeline_version != EXPECTED_PIPELINE_VERSION:
    raise CaptionProxyError(
      f"proxy pipeline mismatch: expected {EXPECTED_PIPELINE_VERSION}, "
      f"received {pipeline_version or 'missing'}"
    )

  model = response.get("model")
  fallback_used = response.get("fallback_used")
  raw_attempts = response.get("attempts")
  if not isinstance(model, str) or not model.strip():
    raise CaptionProxyError("proxy response missing model metadata")
  if not isinstance(fallback_used, bool):
    raise CaptionProxyError("proxy response missing fallback metadata")
  if not isinstance(raw_attempts, list) or not raw_attempts:
    raise CaptionProxyError("proxy response missing attempt metadata")

  attempts = tuple(_proxy_attempt(item) for item in raw_attempts)
  used_indexes = [index for index, attempt in enumerate(attempts) if attempt.outcome == "used"]
  if len(used_indexes) != 1 or used_indexes[0] != len(attempts) - 1:
    raise CaptionProxyError("proxy response contains inconsistent attempt metadata")
  used_fallback_model = attempts[-1].model != attempts[0].model
  if attempts[-1].model != model.strip() or fallback_used != used_fallback_model:
    raise CaptionProxyError("proxy response contains inconsistent model metadata")

  source_kind = response.get("source_kind")
  if expected_source is not None and source_kind != expected_source:
    raise CaptionProxyError("proxy response contains inconsistent source metadata")
  if source_kind is not None and source_kind not in {"url", "base64", "storyboard", "audio"}:
    raise CaptionProxyError("proxy response contains invalid source metadata")

  raw_passes = response.get("perception_passes", [])
  if not isinstance(raw_passes, list):
    raise CaptionProxyError("proxy response contains invalid perception pass metadata")
  perception_passes = tuple(_proxy_attempt(item) for item in raw_passes)
  if expected_source == "storyboard":
    if not perception_passes or not any(item.outcome == "used" for item in perception_passes):
      raise CaptionProxyError("proxy response missing storyboard pass metadata")
    if len(perception_passes) > 6:
      raise CaptionProxyError("proxy response contains too many storyboard passes")

  speech_used = response.get("speech_used", False)
  speech_context_used = response.get("speech_context_used", False)
  corroboration_used = response.get("corroboration_used", False)
  reconciliation_used = response.get("reconciliation_used", False)
  native_video_included = response.get("native_video_included", False)
  ensemble_mode = response.get("ensemble_mode", "disabled")
  if expected_source == "storyboard" and any(
    field not in response for field in (
      "speech_used", "speech_context_used", "corroboration_used", "native_video_included",
    )
  ):
    raise CaptionProxyError("proxy response missing evidence metadata")
  if (not isinstance(speech_used, bool) or not isinstance(speech_context_used, bool)
      or not isinstance(corroboration_used, bool) or not isinstance(reconciliation_used, bool)
      or not isinstance(native_video_included, bool)):
    raise CaptionProxyError("proxy response contains invalid evidence metadata")
  if ensemble_mode not in {"disabled", "conditional", "always"}:
    raise CaptionProxyError("proxy response contains invalid ensemble metadata")
  if expected_source != "storyboard" and (
    speech_used or speech_context_used or corroboration_used or reconciliation_used or native_video_included
  ):
    raise CaptionProxyError("proxy response contains inconsistent evidence metadata")
  if speech_context_used and not speech_used:
    raise CaptionProxyError("proxy response contains inconsistent evidence metadata")
  if reconciliation_used and not corroboration_used:
    raise CaptionProxyError("proxy response contains inconsistent ensemble metadata")

  return ProxyMetadata(
    policy_version=policy_version,
    model=model.strip(),
    fallback_used=fallback_used,
    attempts=attempts,
    source_kind=source_kind,
    perception_passes=perception_passes,
    speech_used=speech_used,
    speech_context_used=speech_context_used,
    corroboration_used=corroboration_used,
    reconciliation_used=reconciliation_used,
    native_video_included=native_video_included,
    ensemble_mode=ensemble_mode,
    pipeline_version=pipeline_version,
  )


def _normalize_facts(value: object) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise CaptionProxyError("proxy response missing facts")
  if any(key not in FACT_FIELDS for key in value):
    raise CaptionProxyError("proxy response contains unknown fact fields")
  summary = value.get("factual_summary")
  exclusions = value.get("do_not_claim")
  if not isinstance(summary, str) or not summary.strip() or not isinstance(exclusions, list):
    raise CaptionProxyError("proxy response contains invalid facts")
  if not all(isinstance(item, str) and item.strip() for item in exclusions):
    raise CaptionProxyError("proxy response contains invalid do_not_claim values")
  normalized: dict[str, Any] = {
    "factual_summary": summary.strip(),
    "do_not_claim": [item.strip() for item in exclusions],
  }
  duration_seconds = value.get("duration_seconds")
  if duration_seconds is not None:
    if (isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float))
        or duration_seconds <= 0 or duration_seconds > 600):
      raise CaptionProxyError("proxy response contains invalid duration_seconds")
    normalized["duration_seconds"] = float(duration_seconds)
  scene_complexity = value.get("scene_complexity")
  if scene_complexity is not None:
    if scene_complexity not in {"sustained", "developing", "montage"}:
      raise CaptionProxyError("proxy response contains invalid scene_complexity")
    normalized["scene_complexity"] = scene_complexity
  structured_fields = FACT_FIELDS[4:]
  present = [field for field in structured_fields if field in value]
  if present and len(present) != len(structured_fields):
    raise CaptionProxyError("proxy response contains incomplete structured facts")
  normalized.update({field: value[field] for field in present})
  return normalized


def _proxy_attempt(value: object) -> ProxyAttempt:
  if not isinstance(value, dict):
    raise CaptionProxyError("proxy response contains invalid attempt metadata")
  model = value.get("model")
  outcome = value.get("outcome")
  error = value.get("error")
  validation_error = value.get("validation_error")
  finish_reason = value.get("finish_reason")
  prompt_tokens = value.get("prompt_tokens")
  completion_tokens = value.get("completion_tokens")
  reasoning_tokens = value.get("reasoning_tokens")
  audio_seconds = value.get("audio_seconds")
  cost_usd = value.get("cost_usd")
  stage = value.get("stage")
  response_excerpt = value.get("response_excerpt")
  duration_ms = value.get("duration_ms")
  if not isinstance(model, str) or not model.strip() or outcome not in {"used", "failed"}:
    raise CaptionProxyError("proxy response contains invalid attempt metadata")
  if error is not None and (not isinstance(error, str) or not error.startswith("upstream_")):
    raise CaptionProxyError("proxy response contains invalid attempt error metadata")
  if outcome == "used" and error is not None:
    raise CaptionProxyError("proxy response contains inconsistent attempt metadata")
  if outcome == "failed" and error is None:
    raise CaptionProxyError("proxy response contains inconsistent attempt metadata")
  if validation_error is not None and (
    not isinstance(validation_error, str) or not validation_error
    or not validation_error.replace("_", "").isalnum()
  ):
    raise CaptionProxyError("proxy response contains invalid validation metadata")
  if validation_error is not None and error != "upstream_invalid_caption":
    raise CaptionProxyError("proxy response contains inconsistent validation metadata")
  if finish_reason is not None and (not isinstance(finish_reason, str) or not finish_reason.strip()):
    raise CaptionProxyError("proxy response contains invalid finish metadata")
  for token_count in (prompt_tokens, completion_tokens, reasoning_tokens):
    if token_count is not None and (
      isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0
    ):
      raise CaptionProxyError("proxy response contains invalid token metadata")
  if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 0:
    raise CaptionProxyError("proxy response contains invalid attempt duration metadata")
  for measured_value in (audio_seconds, cost_usd):
    if measured_value is not None and (
      isinstance(measured_value, bool) or not isinstance(measured_value, (int, float))
      or measured_value < 0
    ):
      raise CaptionProxyError("proxy response contains invalid cost metadata")
  if stage is not None and stage not in {
    "qwen_storyboard", "minimax_storyboard_fallback", "native_video", "reconciliation",
  }:
    raise CaptionProxyError("proxy response contains invalid attempt stage metadata")
  if response_excerpt is not None and (
    not isinstance(response_excerpt, str) or not response_excerpt.strip()
    or len(response_excerpt) > 600
  ):
    raise CaptionProxyError("proxy response contains invalid attempt excerpt metadata")
  return ProxyAttempt(
    model.strip(), outcome, error, duration_ms,
    validation_error=validation_error,
    finish_reason=finish_reason.strip() if finish_reason else None,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
    reasoning_tokens=reasoning_tokens,
    audio_seconds=float(audio_seconds) if audio_seconds is not None else None,
    cost_usd=float(cost_usd) if cost_usd is not None else None,
    stage=stage,
    response_excerpt=response_excerpt.strip() if response_excerpt else None,
  )


def _safe_excerpt(body: bytes, access_id: str | None) -> str:
  text = body.decode("utf-8", errors="replace").strip()
  if access_id:
    text = text.replace(access_id, "[redacted]")
  return text[:500] or "empty response body"
