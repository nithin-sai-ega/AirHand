"""Fixed-size ring buffer of resampled frames.

Downstream stages only ever need the most recent few seconds (the pointer needs the
latest frame, the gesture model needs a sliding window). A ring buffer keeps exactly that
in pre-allocated numpy arrays: new frames overwrite the oldest ones, memory never grows,
and nothing is allocated per frame.

Layout: a write index walks around the arrays. After the buffer has filled once, the
oldest frame sits at the write index and the newest just before it, so `latest(n)` reads
the n frames ending at the write index and wraps around the end if needed.

Not thread-safe: in Phase 1 everything runs on one asyncio thread. A lock is added when a
second thread (the dashboard) starts reading it.
"""

from __future__ import annotations

import numpy as np

from airhand.ingest.chunk import Frames


class RingBuffer:
    """Holds the most recent `capacity` frames of a fixed set of channels."""

    def __init__(self, capacity: int, channels: tuple[str, ...]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.channels = channels
        self._time_ns = np.zeros(capacity, dtype=np.int64)
        self._values = np.full((capacity, len(channels)), np.nan, dtype=np.float64)
        self._valid = np.zeros(capacity, dtype=np.bool_)
        self._write = 0  # index the next frame goes to
        self._size = 0  # number of frames currently stored (<= capacity)
        self.total_written = 0  # frames ever written, including overwritten ones

    def __len__(self) -> int:
        return self._size

    def extend(self, frames: Frames) -> None:
        """Append frames in time order, overwriting the oldest when full."""
        if frames.channels != self.channels:
            raise ValueError("Frames channels do not match the ring buffer")
        n = len(frames)
        self.total_written += n
        if n == 0:
            return
        if n >= self.capacity:
            # Only the newest `capacity` frames can survive; store them from index 0.
            self._time_ns[:] = frames.time_ns[-self.capacity :]
            self._values[:] = frames.values[-self.capacity :]
            self._valid[:] = frames.valid[-self.capacity :]
            self._write = 0
            self._size = self.capacity
            return

        # Indices where the new frames go, wrapping past the end back to 0.
        positions = (self._write + np.arange(n)) % self.capacity
        self._time_ns[positions] = frames.time_ns
        self._values[positions] = frames.values
        self._valid[positions] = frames.valid
        self._write = (self._write + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def latest(self, n: int) -> Frames:
        """Copy of the newest n frames (or fewer if not that many yet), oldest first."""
        n = min(n, self._size)
        positions = (self._write - n + np.arange(n)) % self.capacity
        return Frames(
            time_ns=self._time_ns[positions].copy(),
            values=self._values[positions].copy(),
            valid=self._valid[positions].copy(),
            channels=self.channels,
        )
