import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator

from caption_output import fallback_caption
from io_paths import ProgressiveResultsWriter, write_results_atomic
from runtime_budget import RuntimeBudget


class RunJournal:
  """Owns every durable artifact produced during one container run."""

  def __init__(
    self, output_path: Path, tasks: list[dict[str, Any]], budget: RuntimeBudget,
    on_trace: Callable[[int, Any], None] | None = None,
  ) -> None:
    self.output_path = output_path
    self.tasks = tasks
    self.budget = budget
    self.on_trace = on_trace
    self.writer = ProgressiveResultsWriter(output_path, tasks, _initial_results(tasks))
    self.diagnostics_path = output_path.with_name("diagnostics.jsonl")
    self._diagnostics_lock = Lock()
    self.writer.write_current()
    self.diagnostics_path.write_text("", encoding="utf-8")

  def record_caption(self, index: int, style: str, caption: str) -> None:
    self.writer.update_caption(index, style, caption)
    print(f"RESULT_CAPTION_WRITTEN task_id={self.tasks[index]['task_id']} style={style}", file=sys.stderr)

  def record_result(self, index: int, result: dict[str, Any]) -> None:
    self.writer.update_result(index, result)
    print(f"RESULT_TASK_WRITTEN task_id={self.tasks[index]['task_id']}", file=sys.stderr)

  def record_trace(self, index: int, trace: Any) -> None:
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
        "remaining_runtime_budget": self.budget.remaining_seconds(),
      },
      "perception_metadata": asdict(trace.perception_metadata) if trace.perception_metadata else None,
      "perception_calls": [asdict(metadata) for metadata in trace.perception_calls],
      "style_metadata": {style: asdict(metadata) for style, metadata in trace.style_metadata.items()},
    }
    with self._diagnostics_lock:
      with self.diagnostics_path.open("a", encoding="utf-8") as diagnostics_file:
        diagnostics_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(
      f"TASK_TRACE task_id={trace.task_id} generic_facts={str(generic).lower()} "
      f"perception_seconds={trace.perception_seconds:.3f} style_seconds={trace.style_seconds:.3f} "
      f"remaining_runtime_budget={self.budget.remaining_seconds():.3f}",
      file=sys.stderr,
    )
    if self.on_trace:
      self.on_trace(index, trace)

  def finalize(self, results: list[dict[str, Any]]) -> None:
    write_results_atomic(self.output_path, results, self.tasks)
    print(f"RESULTS_WRITTEN path={self.output_path}", file=sys.stderr)


@contextmanager
def capture_stderr(output_path: Path) -> Iterator[None]:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  stderr_path = output_path.with_name("stderr.log")
  with stderr_path.open("w", encoding="utf-8") as stderr_file:
    previous_stderr = sys.stderr
    sys.stderr = _Tee(previous_stderr, stderr_file)
    try:
      yield
    finally:
      sys.stderr = previous_stderr


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


def _initial_results(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [
    {"task_id": task["task_id"], "captions": {style: fallback_caption(style) for style in task["styles"]}}
    for task in tasks
  ]
