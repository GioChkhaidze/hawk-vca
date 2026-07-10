import os
from dataclasses import dataclass
from typing import Mapping

from contracts import ContractError


DEFAULT_CAPTION_PROXY_TIMEOUT_SECONDS = 90
DEFAULT_RUN_DEADLINE_SECONDS = 570
DEFAULT_MAX_CLIP_CONCURRENCY = 3


@dataclass(frozen=True)
class AppConfig:
  caption_proxy_url: str | None = None
  caption_proxy_access_id: str | None = None
  caption_proxy_timeout_seconds: int = DEFAULT_CAPTION_PROXY_TIMEOUT_SECONDS
  run_deadline_seconds: int = DEFAULT_RUN_DEADLINE_SECONDS
  max_clip_concurrency: int = DEFAULT_MAX_CLIP_CONCURRENCY


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
  if env is None:
    load_local_env()
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

  return AppConfig(
    caption_proxy_url=env_str("CAPTION_PROXY_URL"),
    caption_proxy_access_id=env_str("CAPTION_PROXY_ACCESS_ID") or env_str("CAPTION_PROXY_TOKEN"),
    caption_proxy_timeout_seconds=env_int(
      "CAPTION_PROXY_TIMEOUT_SECONDS", DEFAULT_CAPTION_PROXY_TIMEOUT_SECONDS
    ),
    run_deadline_seconds=env_int("RUN_DEADLINE_SECONDS", DEFAULT_RUN_DEADLINE_SECONDS),
    max_clip_concurrency=env_int("MAX_CLIP_CONCURRENCY", DEFAULT_MAX_CLIP_CONCURRENCY),
  )


def load_local_env(path: str = ".env") -> None:
  if not os.path.exists(path):
    return

  with open(path, "r", encoding="utf-8") as env_file:
    lines = env_file.readlines()

  for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
      continue
    name, value = stripped.split("=", 1)
    os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
