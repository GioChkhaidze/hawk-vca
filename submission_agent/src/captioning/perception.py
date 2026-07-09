import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from config import AppConfig
from proxy_client import CaptionProxyError, ProxyMetadata, perceive_video_via_proxy, proxy_configured
from runtime_budget import RuntimeBudget
from video.download import DownloadError, download_video, video_data_url


GENERIC_FACTS = {
  "factual_summary": "The specific subjects and actions are unclear.",
  "do_not_claim": [
    "Specific subjects, actions, settings, identities, emotions, locations, and visible text are unverified."
  ],
}
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_VIDEO_BYTES = 48 * 1024 * 1024


def perceive_video(
  config: AppConfig, *, video_url: str | None, budget: RuntimeBudget | None = None, urlopen: Any = None,
  downloader: Any = download_video, on_metadata: Callable[[ProxyMetadata], None] | None = None,
) -> dict[str, Any]:
  if not video_url:
    print("PERCEPTION_FALLBACK reason=missing_video_url", file=sys.stderr)
    return fallback_facts()
  if not proxy_configured(config):
    print("PERCEPTION_FALLBACK reason=missing_proxy_config", file=sys.stderr)
    return fallback_facts()

  direct_facts = _request_facts(
    config, video_url=video_url, budget=budget, urlopen=urlopen, source="url", on_metadata=on_metadata,
  )
  if direct_facts != fallback_facts():
    return direct_facts

  download_timeout = _bounded_timeout(budget, DOWNLOAD_TIMEOUT_SECONDS)
  if download_timeout is None:
    print("RUN_DEADLINE_REACHED stage=video_download", file=sys.stderr)
    return fallback_facts()

  try:
    with TemporaryDirectory(prefix="submission_agent_video_") as temp_dir:
      path = downloader(video_url, Path(temp_dir), download_timeout, MAX_VIDEO_BYTES)
      encoded_video = video_data_url(path)
      facts = _request_facts(
        config, video_data=encoded_video, budget=budget, urlopen=urlopen, source="base64", on_metadata=on_metadata,
      )
      if facts != fallback_facts():
        return facts
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


def _bounded_timeout(budget: RuntimeBudget | None, configured_seconds: float) -> float | None:
  return configured_seconds if budget is None else budget.request_timeout(configured_seconds)


def fallback_facts() -> dict[str, Any]:
  return {"factual_summary": GENERIC_FACTS["factual_summary"], "do_not_claim": list(GENERIC_FACTS["do_not_claim"])}
