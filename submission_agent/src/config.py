import os
from dataclasses import dataclass
from typing import Mapping

from contracts import ContractError


DEFAULT_CAPTION_PROXY_TIMEOUT_SECONDS = 90
DEFAULT_RUN_DEADLINE_SECONDS = 570
DEFAULT_MAX_CLIP_CONCURRENCY = 3
DEFAULT_STORYBOARD_PERCEPTION_ENABLED = False
DEFAULT_STORYBOARD_MAX_FRAMES = 16


@dataclass(frozen=True)
class AppConfig:
  caption_proxy_url: str | None = None
  caption_proxy_access_id: str | None = None
  caption_proxy_timeout_seconds: int = DEFAULT_CAPTION_PROXY_TIMEOUT_SECONDS
  run_deadline_seconds: int = DEFAULT_RUN_DEADLINE_SECONDS
  max_clip_concurrency: int = DEFAULT_MAX_CLIP_CONCURRENCY
  storyboard_perception_enabled: bool = DEFAULT_STORYBOARD_PERCEPTION_ENABLED
  storyboard_max_frames: int = DEFAULT_STORYBOARD_MAX_FRAMES


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
  env = os.environ if env is None else env

  def env_str(name: str) -> str | None:
    return env.get(name, "").strip() or None

  def env_int(name: str, default: int) -> int:
    value = env_str(name)
    if value is None:
      return default
    try:
      parsed = int(value)
    except ValueError as exc:
      raise ContractError(f"{name} must be a positive integer") from exc
    if parsed < 1:
      raise ContractError(f"{name} must be a positive integer")
    return parsed

  def env_bool(name: str, default: bool) -> bool:
    value = env_str(name)
    if value is None:
      return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
      return True
    if normalized in {"0", "false", "no", "off"}:
      return False
    raise ContractError(f"{name} must be a boolean")

  return AppConfig(
    caption_proxy_url=env_str("CAPTION_PROXY_URL"),
    caption_proxy_access_id=env_str("CAPTION_PROXY_ACCESS_ID") or env_str("CAPTION_PROXY_TOKEN"),
    caption_proxy_timeout_seconds=env_int(
      "CAPTION_PROXY_TIMEOUT_SECONDS", DEFAULT_CAPTION_PROXY_TIMEOUT_SECONDS
    ),
    run_deadline_seconds=env_int("RUN_DEADLINE_SECONDS", DEFAULT_RUN_DEADLINE_SECONDS),
    max_clip_concurrency=env_int("MAX_CLIP_CONCURRENCY", DEFAULT_MAX_CLIP_CONCURRENCY),
    storyboard_perception_enabled=env_bool(
      "STORYBOARD_PERCEPTION_ENABLED", DEFAULT_STORYBOARD_PERCEPTION_ENABLED
    ),
    storyboard_max_frames=min(16, env_int("STORYBOARD_MAX_FRAMES", DEFAULT_STORYBOARD_MAX_FRAMES)),
  )
