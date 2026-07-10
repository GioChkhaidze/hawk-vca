from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import sys
import time
from threading import Lock
from typing import Any, Callable

from captioning.perception import fallback_facts, perceive_video
from captioning.styles import render_style_captions
from config import AppConfig
from fallbacks import fallback_caption
from proxy_client import ProxyMetadata
from runtime_budget import RuntimeBudget
from validators import validate_caption


CaptionCallback = Callable[[int, str, str], None]
ResultCallback = Callable[[int, dict[str, Any]], None]
Clock = Callable[[], float]


@dataclass(frozen=True)
class TaskTrace:
  task_id: str
  facts: dict[str, Any]
  perception_seconds: float
  style_seconds: float
  total_seconds: float
  perception_metadata: ProxyMetadata | None = None
  style_metadata: dict[str, ProxyMetadata] = field(default_factory=dict)
  perception_calls: tuple[ProxyMetadata, ...] = ()


TraceCallback = Callable[[int, TaskTrace], None]


def process_tasks(
  tasks: list[dict[str, Any]], config: AppConfig, on_caption: CaptionCallback | None = None,
  on_result: ResultCallback | None = None, on_trace: TraceCallback | None = None,
  budget: RuntimeBudget | None = None, clock: Clock = time.perf_counter,
) -> list[dict[str, Any]]:
  results: list[dict[str, Any] | None] = [None] * len(tasks)
  workers = min(max(1, config.max_clip_concurrency), max(1, len(tasks)))

  def run_one(index: int) -> dict[str, Any]:
    def report_caption(style: str, caption: str) -> None:
      if on_caption:
        on_caption(index, style, caption)

    def report_trace(trace: TaskTrace) -> None:
      if on_trace:
        on_trace(index, trace)

    result = process_task(
      tasks[index], config, on_caption=report_caption if on_caption else None,
      on_trace=report_trace if on_trace else None, budget=budget, clock=clock,
    )
    if on_result:
      on_result(index, result)
    return result

  if workers == 1:
    for index in range(len(tasks)):
      results[index] = run_one(index)
  else:
    with ThreadPoolExecutor(max_workers=workers) as pool:
      futures = {pool.submit(run_one, index): index for index in range(len(tasks))}
      for future in as_completed(futures):
        results[futures[future]] = future.result()

  return [result for result in results if result is not None]


def process_task(
  task: dict[str, Any], config: AppConfig, on_caption: Callable[[str, str], None] | None = None,
  on_trace: Callable[[TaskTrace], None] | None = None, budget: RuntimeBudget | None = None,
  clock: Clock = time.perf_counter,
) -> dict[str, Any]:
  started = clock()
  metadata_lock = Lock()
  perception_metadata: ProxyMetadata | None = None
  perception_calls: list[ProxyMetadata] = []
  style_metadata: dict[str, ProxyMetadata] = {}

  def record_perception_metadata(metadata: ProxyMetadata) -> None:
    nonlocal perception_metadata
    with metadata_lock:
      perception_calls.append(metadata)
      perception_metadata = metadata

  def record_style_metadata(style: str, metadata: ProxyMetadata) -> None:
    with metadata_lock:
      style_metadata[style] = metadata

  perception_started = clock()
  if budget is not None and budget.exhausted():
    print("RUN_DEADLINE_REACHED stage=task_start", file=sys.stderr)
    facts = fallback_facts()
  else:
    try:
      facts = perceive_video(
        config, video_url=task["video_url"], budget=budget, on_metadata=record_perception_metadata,
      )
    except Exception as exc:
      print(f"PERCEPTION_EXCEPTION source=url error={str(exc)[:240]!r}", file=sys.stderr)
      facts = fallback_facts()
  perception_seconds = clock() - perception_started

  if on_caption:
    summary = facts.get("factual_summary")
    for style in task["styles"]:
      on_caption(style, fallback_caption(style, summary))

  style_started = clock()
  captions = render_task_captions(
    facts, task["styles"], config, on_caption, budget, on_metadata=record_style_metadata,
  )
  style_seconds = clock() - style_started
  result = {"task_id": task["task_id"], "captions": captions}

  if on_trace:
    on_trace(TaskTrace(
      task_id=task["task_id"],
      facts=facts,
      perception_seconds=perception_seconds,
      style_seconds=style_seconds,
      total_seconds=clock() - started,
      perception_metadata=perception_metadata,
      style_metadata=dict(style_metadata),
      perception_calls=tuple(perception_calls),
    ))
  return result


def render_task_captions(
  facts: dict[str, Any], requested_styles: list[str], config: AppConfig,
  on_caption: Callable[[str, str], None] | None = None, budget: RuntimeBudget | None = None,
  on_metadata: Callable[[str, ProxyMetadata], None] | None = None,
) -> dict[str, str]:
  reported = {}

  def report_caption(style: str, caption: str) -> None:
    reported[style] = caption
    if on_caption:
      on_caption(style, caption)

  try:
    captions = render_style_captions(
      facts, requested_styles, config, on_caption=report_caption if on_caption else None, budget=budget,
      on_metadata=on_metadata,
    )
  except Exception as exc:
    print(f"STYLE_RENDERER_EXCEPTION error={str(exc)[:240]!r}", file=sys.stderr)
    captions = {}

  completed = complete_captions(facts, requested_styles, captions)
  if on_caption:
    for style, caption in completed.items():
      if reported.get(style) != caption:
        report_caption(style, caption)
  return completed


def complete_captions(
  facts: dict[str, Any], requested_styles: list[str], captions: dict[str, object] | object
) -> dict[str, str]:
  raw_captions = captions if isinstance(captions, dict) else {}
  summary = facts.get("factual_summary")
  completed = {}
  for style in requested_styles:
    caption = raw_captions.get(style)
    reasons = validate_caption(caption, style)
    if reasons:
      print(f"CAPTION_FALLBACK style={style} reasons={reasons}", file=sys.stderr)
      caption = fallback_caption(style, summary)
    completed[style] = str(caption).strip()
  return completed
