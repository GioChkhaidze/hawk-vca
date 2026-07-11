from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import sys
from typing import Any, Callable

from config import AppConfig
from fallbacks import fallback_caption
from proxy_client import CaptionProxyError, ProxyMetadata, proxy_configured, style_caption_via_proxy
from runtime_budget import RuntimeBudget
from validators import validate_caption


MAX_STYLE_WORKERS = 4
MAX_AVOID_CAPTION_CHARS = 600


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
    ordered = {style: captions[style] for style in requested_styles}

  repair_targets = diversity_repair_targets(ordered)
  for style in repair_targets:
    if budget is not None and budget.exhausted():
      break
    avoided = [
      avoid_caption_excerpt(caption)
      for other, caption in ordered.items()
      if other != style
    ][:3]
    candidate = render_style_caption(
      facts, style, config, urlopen=urlopen, budget=budget, on_metadata=on_metadata,
      avoid_captions=avoided,
    )
    previous_similarity = max(
      caption_similarity(ordered[style], caption) for other, caption in ordered.items() if other != style
    )
    candidate_similarity = max(
      caption_similarity(candidate, caption) for other, caption in ordered.items() if other != style
    )
    if not validate_caption(candidate, style) and candidate_similarity + 0.03 < previous_similarity:
      ordered[style] = candidate
      print(
        f"STYLE_DIVERSITY_REPAIRED style={style} similarity={candidate_similarity:.3f}", file=sys.stderr,
      )
      if on_caption:
        on_caption(style, candidate)
  return ordered


def render_style_caption(
  facts: dict[str, Any], style: str, config: AppConfig, urlopen: Any = None,
  budget: RuntimeBudget | None = None, on_metadata: Callable[[str, ProxyMetadata], None] | None = None,
  avoid_captions: list[str] | None = None,
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
      "avoid_captions": avoid_captions,
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


def diversity_repair_targets(captions: dict[str, str], threshold: float = 0.72) -> list[str]:
  styles = list(captions)
  targets = []
  for index, left in enumerate(styles):
    for right in styles[index + 1:]:
      if caption_similarity(captions[left], captions[right]) < threshold:
        continue
      target = left if right == "formal" else right
      if target not in targets:
        targets.append(target)
  return targets


def avoid_caption_excerpt(caption: str) -> str:
  normalized = " ".join(caption.split())
  if len(normalized) <= MAX_AVOID_CAPTION_CHARS:
    return normalized
  clipped = normalized[:MAX_AVOID_CAPTION_CHARS]
  word_boundary = clipped.rsplit(" ", 1)[0].rstrip()
  return word_boundary or clipped


def caption_similarity(left: str, right: str) -> float:
  normalized_left = " ".join(left.lower().split())
  normalized_right = " ".join(right.lower().split())
  left_tokens = normalized_left.split()
  right_tokens = normalized_right.split()
  shared_prefix = 0
  for left_token, right_token in zip(left_tokens[:8], right_tokens[:8]):
    if left_token != right_token:
      break
    shared_prefix += 1
  prefix_denominator = max(1, min(8, len(left_tokens), len(right_tokens)))
  return max(
    SequenceMatcher(None, normalized_left, normalized_right).ratio(),
    SequenceMatcher(None, left_tokens, right_tokens).ratio(),
    shared_prefix / prefix_denominator,
  )


def fallback_captions(
  facts: dict[str, Any], requested_styles: list[str], on_caption: Callable[[str, str], None] | None = None
) -> dict[str, str]:
  summary = facts.get("factual_summary")
  captions = {style: fallback_caption(style, summary) for style in requested_styles}
  if on_caption:
    for style, caption in captions.items():
      on_caption(style, caption)
  return captions
