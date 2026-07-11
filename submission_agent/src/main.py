import json
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from config import load_config
from contracts import validate_tasks
from fallbacks import fallback_caption
from io_paths import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, ProgressiveResultsWriter, write_results_atomic
from orchestrator import TraceCallback, process_tasks
from runtime_budget import RuntimeBudget


def load_tasks(input_path: Path) -> list[dict[str, Any]]:
  with input_path.open("r", encoding="utf-8") as task_file:
    return validate_tasks(json.load(task_file))


def run(
  input_path: Path = DEFAULT_INPUT_PATH, output_path: Path = DEFAULT_OUTPUT_PATH,
  on_trace: TraceCallback | None = None,
) -> int:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  stderr_path = output_path.with_name("stderr.log")
  with stderr_path.open("w", encoding="utf-8") as stderr_file:
    previous_stderr = sys.stderr
    sys.stderr = _Tee(previous_stderr, stderr_file)
    try:
      return _run(input_path, output_path, on_trace)
    finally:
      sys.stderr = previous_stderr


def _run(
  input_path: Path, output_path: Path, on_trace: TraceCallback | None,
) -> int:
  config = load_config()
  budget = RuntimeBudget.for_seconds(config.run_deadline_seconds)
  tasks = load_tasks(input_path)
  writer = ProgressiveResultsWriter(output_path, tasks, initial_results(tasks))
  writer.write_current()
  diagnostics_path = output_path.with_name("diagnostics.jsonl")
  diagnostics_lock = Lock()
  diagnostics_path.write_text("", encoding="utf-8")

  def save_caption(index: int, style: str, caption: str) -> None:
    writer.update_caption(index, style, caption)
    print(f"RESULT_CAPTION_WRITTEN task_id={tasks[index]['task_id']} style={style}", file=sys.stderr)

  def save_result(index: int, result: dict[str, Any]) -> None:
    writer.update_result(index, result)
    print(f"RESULT_TASK_WRITTEN task_id={tasks[index]['task_id']}", file=sys.stderr)

  def save_trace(index: int, trace: Any) -> None:
    generic = trace.facts.get("factual_summary") == "The specific subjects and actions are unclear."
    payload = {
      "task_id": trace.task_id,
      "facts_selection": {
        "generic": generic,
        "reason": (
          "all perception paths returned generic facts; inspect stderr.log for the exact failed stage"
          if generic else "provider factual packet selected"
        ),
      },
      "facts": trace.facts,
      "timing_seconds": {
        "perception": trace.perception_seconds,
        "style": trace.style_seconds,
        "total": trace.total_seconds,
        "remaining_runtime_budget": budget.remaining_seconds(),
      },
      "perception_metadata": asdict(trace.perception_metadata) if trace.perception_metadata else None,
      "perception_calls": [asdict(metadata) for metadata in trace.perception_calls],
      "style_metadata": {style: asdict(metadata) for style, metadata in trace.style_metadata.items()},
    }
    with diagnostics_lock:
      with diagnostics_path.open("a", encoding="utf-8") as diagnostics_file:
        diagnostics_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(
      f"TASK_TRACE task_id={trace.task_id} generic_facts={str(generic).lower()} "
      f"perception_seconds={trace.perception_seconds:.3f} style_seconds={trace.style_seconds:.3f} "
      f"remaining_runtime_budget={budget.remaining_seconds():.3f}",
      file=sys.stderr,
    )
    if on_trace:
      on_trace(index, trace)

  results = process_tasks(
    tasks, config, on_caption=save_caption, on_result=save_result, on_trace=save_trace, budget=budget
  )
  write_results_atomic(output_path, results, tasks)
  print(f"RESULTS_WRITTEN path={output_path}", file=sys.stderr)
  return 0


class _Tee:
  def __init__(self, *streams: Any) -> None:
    self.streams = streams

  def write(self, value: str) -> int:
    for stream in self.streams:
      stream.write(value)
    return len(value)

  def flush(self) -> None:
    for stream in self.streams:
      stream.flush()


def initial_results(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [
    {"task_id": task["task_id"], "captions": {style: fallback_caption(style) for style in task["styles"]}}
    for task in tasks
  ]


def main() -> int:
  try:
    return run()
  except Exception as exc:
    print(f"submission contract failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
