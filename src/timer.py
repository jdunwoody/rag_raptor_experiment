import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional


class TimerError(Exception): ...


@dataclass
class Timer(ContextDecorator):
    timers: ClassVar[Dict[str, list]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name and self.name not in self.timers:
            self.timers[self.name] = []

    def start(self) -> None:
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name].append(elapsed_time)

        return elapsed_time

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()
