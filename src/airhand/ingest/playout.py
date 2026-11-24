"""Playout buffer: turns bursty arrivals into a steady stream of frames.

Sensor Logger delivers about one packet per second (free tier), each holding ~1 s of 100 Hz
samples. Applying a whole packet at once would make the cursor jump once per second. We do
have every sample, just late, so the fix is the one video calls use for network jitter:
hold the frames back by a fixed delay and release each one at its own time.

Timing model. When playback starts, the first buffered frame (sensor time s0) is scheduled
at laptop monotonic time m0 = now + delay. Every later frame with sensor time t is due at

    m0 + (t - s0)

so frames come out exactly as far apart as they were recorded (10 ms on the 100 Hz grid).
The delay must be longer than the worst gap between packets, otherwise the buffer runs dry.

Two things can go wrong, and both are handled by re-anchoring (choosing a new s0, m0):
- underrun: a frame is due but has not arrived (the delay was too short for this burst).
  Playback pauses and restarts `delay` after the next packet arrives.
- overflow: far more than `delay` of data is buffered (e.g. a backlog after a stall, or the
  watch clock running slightly fast). The oldest frames are skipped so the latency does not
  keep growing.

The buffer is used by two threads (network receiver pushes, pointer loop pops), so every
public method takes a lock. Times are passed in explicitly, which makes it testable with a
fake clock.
"""

from __future__ import annotations

import threading

from airhand.ingest.chunk import Frames

NS_PER_S = 1_000_000_000


class PlayoutBuffer:
    def __init__(
        self,
        channels: tuple[str, ...],
        frame_period_ns: int,
        delay_s: float,
        max_excess_s: float,
    ) -> None:
        if delay_s <= 0 or max_excess_s <= 0:
            raise ValueError("delay_s and max_excess_s must be positive")
        self.channels = channels
        self.frame_period_ns = frame_period_ns
        self.delay_ns = round(delay_s * NS_PER_S)
        self.max_excess_ns = round(max_excess_s * NS_PER_S)
        self.n_underruns = 0
        self.n_skips = 0
        self._frames = Frames.empty(channels)
        self._anchor: tuple[int, int] | None = None  # (sensor time s0, monotonic time m0)
        self._starved = False
        self._last_released_ns: int | None = None
        self._lock = threading.Lock()

    def push(self, frames: Frames, now_ns: int) -> None:
        """Add newly resampled frames (in time order) that arrived at `now_ns`."""
        if len(frames) == 0:
            return
        with self._lock:
            self._frames = Frames.concatenate([self._frames, frames], self.channels)
            if self._anchor is None or self._starved:
                self._anchor = (int(self._frames.time_ns[0]), now_ns + self.delay_ns)
                self._starved = False
            self._skip_if_too_far_behind(now_ns)

    def _skip_if_too_far_behind(self, now_ns: int) -> None:
        # Lag = how far the newest frame is ahead of the one being played right now. Right
        # after a packet arrives it is normally about `delay`; much more means a backlog.
        assert self._anchor is not None
        sensor_s0, mono_m0 = self._anchor
        due_now = sensor_s0 + (now_ns - mono_m0)
        newest = int(self._frames.time_ns[-1])
        if newest - due_now <= self.delay_ns + self.max_excess_ns:
            return
        # Re-anchor so the newest frame plays `delay` from now, and drop the frames that
        # would already be overdue under the new timing instead of playing them in a rush.
        self._anchor = (newest, now_ns + self.delay_ns)
        first_kept = int((self._frames.time_ns < newest - self.delay_ns).sum())
        self._frames = self._frames.slice(first_kept, len(self._frames))
        self.n_skips += 1

    def pop_due(self, now_ns: int) -> Frames:
        """Remove and return every frame whose play time is at or before `now_ns`."""
        with self._lock:
            if self._anchor is None or self._starved:
                return Frames.empty(self.channels)
            sensor_s0, mono_m0 = self._anchor
            due_until = sensor_s0 + (now_ns - mono_m0)
            n_due = int((self._frames.time_ns <= due_until).sum())
            out = self._frames.slice(0, n_due)
            self._frames = self._frames.slice(n_due, len(self._frames))
            if n_due > 0:
                self._last_released_ns = int(out.time_ns[-1])

            next_expected = (
                self._last_released_ns + self.frame_period_ns
                if self._last_released_ns is not None
                else sensor_s0
            )
            if len(self._frames) == 0 and due_until >= next_expected:
                self._starved = True
                self.n_underruns += 1
            return out

    def _buffered_ns(self) -> int:
        if len(self._frames) == 0:
            return 0
        return int(self._frames.time_ns[-1] - self._frames.time_ns[0]) + self.frame_period_ns

    @property
    def buffered_s(self) -> float:
        """Seconds of data waiting to be played."""
        with self._lock:
            return self._buffered_ns() / NS_PER_S

    @property
    def starved(self) -> bool:
        with self._lock:
            return self._starved
