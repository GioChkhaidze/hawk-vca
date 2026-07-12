import json
import sys
from pathlib import Path
from typing import Any

from config import load_config
from contracts import validate_tasks
from io_paths import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH
from orchestrator import TraceCallback, process_tasks
from run_journal import RunJournal, capture_stderr
from runtime_budget import RuntimeBudget


def load_tasks(input_path: Path) -> list[dict[str, Any]]:
  with input_path.open("r", encoding="utf-8") as task_file:
    return validate_tasks(json.load(task_file))


def run(
  input_path: Path = DEFAULT_INPUT_PATH, output_path: Path = DEFAULT_OUTPUT_PATH,
  on_trace: TraceCallback | None = None,
) -> int:
  with capture_stderr(output_path):
    return _run(input_path, output_path, on_trace)


def _run(
  input_path: Path, output_path: Path, on_trace: TraceCallback | None,
) -> int:
  config = load_config()
  budget = RuntimeBudget.for_seconds(config.run_deadline_seconds)
  tasks = load_tasks(input_path)
  journal = RunJournal(output_path, tasks, budget, on_trace)

  results = process_tasks(
    tasks, config, on_caption=journal.record_caption, on_result=journal.record_result,
    on_trace=journal.record_trace, budget=budget,
  )
  journal.finalize(results)
  return 0


def main() -> int:
  try:
    return run()
  except Exception as exc:
    print(f"submission contract failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
