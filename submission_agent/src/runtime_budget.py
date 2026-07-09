import time
from dataclasses import dataclass, field
from typing import Callable


Clock = Callable[[], float]
MIN_REQUEST_SECONDS = 1.0


@dataclass(frozen=True)
class RuntimeBudget:
  deadline: float
  clock: Clock = field(repr=False, compare=False)

  @classmethod
  def for_seconds(cls, seconds: float, clock: Clock = time.monotonic) -> "RuntimeBudget":
    if seconds <= 0:
      raise ValueError("runtime budget must be positive")
    return cls(deadline=clock() + seconds, clock=clock)

  def remaining_seconds(self) -> float:
    return max(0.0, self.deadline - self.clock())

  def request_timeout(self, configured_seconds: float, reserve_seconds: float = 1.0) -> float | None:
    available = self.remaining_seconds() - reserve_seconds
    if configured_seconds <= 0 or available < MIN_REQUEST_SECONDS:
      return None
    return min(float(configured_seconds), available)

  def exhausted(self, reserve_seconds: float = 1.0) -> bool:
    return self.remaining_seconds() <= reserve_seconds
