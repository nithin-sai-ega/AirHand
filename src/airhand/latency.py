"""Per-stage latency measurements.

Each pipeline stage records how long it took (or, for the network, how old the data was)
into a rolling window; `summary()` gives p50/p95/p99 in milliseconds for the dashboard and
the results tables. Used from several threads, so access is locked.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

NS_PER_MS = 1_000_000


class StageTimer:
    def __init__(self, window: int = 2000) -> None:
        self.window = window
        self._samples: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def record(self, stage: str, milliseconds: float) -> None:
        with self._lock:
            if stage not in self._samples:
                self._samples[stage] = deque(maxlen=self.window)
            self._samples[stage].append(milliseconds)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time the body of a `with` block with the high-resolution performance counter."""
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter_ns() - start) / NS_PER_MS)

    def summary(self) -> dict[str, dict[str, float]]:
        """{stage: {"p50": ms, "p95": ms, "p99": ms, "n": count}} over the rolling window."""
        with self._lock:
            copies = {stage: np.array(values) for stage, values in self._samples.items()}
        result = {}
        for stage, values in copies.items():
            if len(values) == 0:
                continue
            p50, p95, p99 = np.percentile(values, [50, 95, 99])
            result[stage] = {"p50": p50, "p95": p95, "p99": p99, "n": float(len(values))}
        return result
