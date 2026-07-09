import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

from contracts import validate_results


DEFAULT_INPUT_PATH = Path("/input/tasks.json")
DEFAULT_OUTPUT_PATH = Path("/output/results.json")


class ProgressiveResultsWriter:
  def __init__(self, output_path: Path, tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    self.output_path = output_path
    self.tasks = tasks
    self._results = deepcopy(results)
    self._lock = Lock()

  def write_current(self) -> None:
    with self._lock:
      self._write_locked()

  def update_result(self, index: int, result: dict[str, Any]) -> None:
    with self._lock:
      self._results[index] = deepcopy(result)
      self._write_locked()

  def update_caption(self, index: int, style: str, caption: str) -> None:
    with self._lock:
      self._results[index]["captions"][style] = caption
      self._write_locked()

  def _write_locked(self) -> None:
    write_results_atomic(self.output_path, self._results, self.tasks)


def write_results_atomic(output_path: Path, results: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temp_path: Path | None = None

  try:
    with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
    ) as temp_file:
      temp_path = Path(temp_file.name)
      json.dump(results, temp_file, indent=2)
      temp_file.write("\n")

    with temp_path.open("r", encoding="utf-8") as temp_file:
      loaded_results = json.load(temp_file)

    validate_results(loaded_results, tasks)
    os.replace(temp_path, output_path)
    temp_path = None
  finally:
    if temp_path is not None and temp_path.exists():
      temp_path.unlink()
