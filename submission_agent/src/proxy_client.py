import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import AppConfig


class CaptionProxyError(Exception):
  pass


EXPECTED_POLICY_VERSION = "style-spec-v4-20260710"


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


@dataclass(frozen=True)
class ProxyMetadata:
  policy_version: str
  model: str
  fallback_used: bool
  attempts: tuple[ProxyAttempt, ...]
  source_kind: str | None = None


@dataclass(frozen=True)
class PerceptionProxyResult:
  facts: dict[str, Any]
  metadata: ProxyMetadata


@dataclass(frozen=True)
class StyleProxyResult:
  caption: str
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
  facts = response.get("facts")
  if not isinstance(facts, dict):
    raise CaptionProxyError("proxy response missing facts")

  summary = facts.get("factual_summary")
  do_not_claim = facts.get("do_not_claim")
  if not isinstance(summary, str) or not summary.strip() or not isinstance(do_not_claim, list):
    raise CaptionProxyError("proxy response contains invalid facts")
  if not all(isinstance(item, str) and item.strip() for item in do_not_claim):
    raise CaptionProxyError("proxy response contains invalid do_not_claim values")
  normalized_facts = {
    "factual_summary": summary.strip(),
    "do_not_claim": [item.strip() for item in do_not_claim],
  }
  expected_source = "url" if video_url else "base64"
  return PerceptionProxyResult(normalized_facts, _proxy_metadata(response, expected_source))


def style_caption_via_proxy(
  config: AppConfig, *, style: str, facts: dict[str, Any], timeout_seconds: float | None = None,
  urlopen: Any = urllib.request.urlopen,
) -> StyleProxyResult:
  response = post_proxy(
    config, "/style", {"style": style, "facts": facts}, timeout_seconds=timeout_seconds, urlopen=urlopen
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
  if source_kind is not None and source_kind not in {"url", "base64"}:
    raise CaptionProxyError("proxy response contains invalid source metadata")

  return ProxyMetadata(
    policy_version=policy_version,
    model=model.strip(),
    fallback_used=fallback_used,
    attempts=attempts,
    source_kind=source_kind,
  )


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
  return ProxyAttempt(
    model.strip(), outcome, error, duration_ms,
    validation_error=validation_error,
    finish_reason=finish_reason.strip() if finish_reason else None,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
    reasoning_tokens=reasoning_tokens,
  )


def _safe_excerpt(body: bytes, access_id: str | None) -> str:
  text = body.decode("utf-8", errors="replace").strip()
  if access_id:
    text = text.replace(access_id, "[redacted]")
  return text[:500] or "empty response body"
