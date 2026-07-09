import json
import sys
from pathlib import Path
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
  config = load_config()
  budget = RuntimeBudget.for_seconds(config.run_deadline_seconds)
  tasks = load_tasks(input_path)
  writer = ProgressiveResultsWriter(output_path, tasks, initial_results(tasks))
  writer.write_current()

  def save_caption(index: int, style: str, caption: str) -> None:
    writer.update_caption(index, style, caption)
    print(f"RESULT_CAPTION_WRITTEN task_id={tasks[index]['task_id']} style={style}", file=sys.stderr)

  def save_result(index: int, result: dict[str, Any]) -> None:
    writer.update_result(index, result)
    print(f"RESULT_TASK_WRITTEN task_id={tasks[index]['task_id']}", file=sys.stderr)

  results = process_tasks(
    tasks, config, on_caption=save_caption, on_result=save_result, on_trace=on_trace, budget=budget
  )
  write_results_atomic(output_path, results, tasks)
  print(f"RESULTS_WRITTEN path={output_path}", file=sys.stderr)
  return 0


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
