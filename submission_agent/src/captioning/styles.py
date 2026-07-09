from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from typing import Any, Callable

from config import AppConfig
from fallbacks import fallback_caption
from proxy_client import CaptionProxyError, ProxyMetadata, proxy_configured, style_caption_via_proxy
from runtime_budget import RuntimeBudget
from validators import validate_caption


MAX_STYLE_WORKERS = 4


def render_style_captions(
  facts: dict[str, Any], requested_styles: list[str], config: AppConfig, urlopen: Any = None,
  on_caption: Callable[[str, str], None] | None = None, budget: RuntimeBudget | None = None,
  on_metadata: Callable[[str, ProxyMetadata], None] | None = None,
) -> dict[str, str]:
  if not proxy_configured(config):
    print("STYLE_FALLBACK reason=missing_proxy_config", file=sys.stderr)
    return fallback_captions(facts, requested_styles, on_caption)
  if budget is not None and budget.exhausted():
    print("RUN_DEADLINE_REACHED stage=styles", file=sys.stderr)
    return fallback_captions(facts, requested_styles, on_caption)

  def render(style: str) -> str:
    return render_style_caption(
      facts, style, config, urlopen=urlopen, budget=budget, on_metadata=on_metadata,
    )

  with ThreadPoolExecutor(max_workers=min(MAX_STYLE_WORKERS, max(1, len(requested_styles)))) as pool:
    futures = {pool.submit(render, style): style for style in requested_styles}
    captions = {}
    for future in as_completed(futures):
      style = futures[future]
      captions[style] = future.result()
      if on_caption:
        on_caption(style, captions[style])
    return {style: captions[style] for style in requested_styles}


def render_style_caption(
  facts: dict[str, Any], style: str, config: AppConfig, urlopen: Any = None,
  budget: RuntimeBudget | None = None, on_metadata: Callable[[str, ProxyMetadata], None] | None = None,
) -> str:
  summary = facts.get("factual_summary")
  timeout = config.caption_proxy_timeout_seconds
  if budget is not None:
    bounded_timeout = budget.request_timeout(timeout)
    if bounded_timeout is None:
      print(f"RUN_DEADLINE_REACHED stage=style style={style}", file=sys.stderr)
      return fallback_caption(style, summary)
    timeout = bounded_timeout

  try:
    kwargs = {
      "config": config,
      "style": style,
      "facts": facts,
      "timeout_seconds": timeout,
    }
    if urlopen is not None:
      kwargs["urlopen"] = urlopen
    result = style_caption_via_proxy(**kwargs)
    if on_metadata:
      on_metadata(style, result.metadata)
    caption = result.caption
    reasons = validate_caption(caption, style)
    if not reasons:
      print(
        f"STYLE_PROXY_USED style={style} model={result.metadata.model} "
        f"fallback_used={str(result.metadata.fallback_used).lower()} "
        f"policy_version={result.metadata.policy_version}",
        file=sys.stderr,
      )
      return caption
    print(f"STYLE_PROXY_INVALID style={style} reasons={reasons} excerpt={caption[:160]!r}", file=sys.stderr)
  except (CaptionProxyError, OSError) as exc:
    print(f"STYLE_PROXY_FAILED style={style} error={str(exc)[:240]!r}", file=sys.stderr)

  print(f"STYLE_FALLBACK style={style} reason=proxy_failed_or_invalid", file=sys.stderr)
  return fallback_caption(style, summary)


def fallback_captions(
  facts: dict[str, Any], requested_styles: list[str], on_caption: Callable[[str, str], None] | None = None
) -> dict[str, str]:
  summary = facts.get("factual_summary")
  captions = {style: fallback_caption(style, summary) for style in requested_styles}
  if on_caption:
    for style, caption in captions.items():
      on_caption(style, caption)
  return captions
