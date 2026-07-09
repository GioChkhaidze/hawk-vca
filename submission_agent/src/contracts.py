from typing import Any


class ContractError(ValueError):
  pass


def validate_tasks(payload: Any) -> list[dict[str, Any]]:
  if not isinstance(payload, list):
    raise ContractError("tasks.json must contain a JSON array")

  for index, task in enumerate(payload):
    if not isinstance(task, dict):
      raise ContractError(f"task at index {index} must be an object")

    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
      raise ContractError(f"task at index {index} must include task_id")

    video_url = task.get("video_url")
    if not isinstance(video_url, str) or not video_url.strip():
      raise ContractError(f"task {task_id} must include video_url")

    styles = task.get("styles")
    if not isinstance(styles, list) or not styles:
      raise ContractError(f"task {task_id} must include styles")
    if not all(isinstance(style, str) and style.strip() for style in styles):
      raise ContractError(f"task {task_id} styles must be strings")

  return payload


def validate_results(payload: Any, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
  validate_tasks(tasks)

  if not isinstance(payload, list):
    raise ContractError("results.json must contain a JSON array")
  if len(payload) != len(tasks):
    raise ContractError("result count must match task count")

  for index, result in enumerate(payload):
    if not isinstance(result, dict):
      raise ContractError(f"result at index {index} must be an object")

    task_id = result.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
      raise ContractError(f"result at index {index} must include task_id")
    if task_id != tasks[index]["task_id"]:
      raise ContractError(f"result at index {index} has unexpected task_id")

    captions = result.get("captions")
    if not isinstance(captions, dict):
      raise ContractError(f"result {task_id} must include captions")

    for style in tasks[index]["styles"]:
      caption = captions.get(style)
      if not isinstance(caption, str) or not caption.strip():
        raise ContractError(f"result {task_id} missing caption for {style}")

  return payload
