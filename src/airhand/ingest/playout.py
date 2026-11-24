"""Playout buffer: turns bursty arrivals into a steady stream of frames.

Sensor Logger delivers about one packet per second (free tier), each holding ~1 s of 100 Hz
samples. Applying a whole packet at once would make the cursor jump once per second. We do
have every sample, just late, so the fix is the one video calls use for network jitter:
hold the frames back by a fixed delay and release each one at its own time.

Timing model. When playback starts, the first buffered frame (sensor time s0) is scheduled
at laptop monotonic time m0 = now + delay. Every later frame with sensor time t is due at

    m0 + (t - s0)

so frames come out exactly as far apart as they were recorded (10 ms on the 100 Hz grid).
The delay must cover the worst lateness of a packet relative to the first one, otherwise the
buffer runs dry.

Two things can go wrong, and both are handled by re-anchoring (choosing a new s0, m0):
- underrun: a frame is due but has not arrived (the delay was too short for this packet).
  Playback pauses and restarts `delay` after the next packet arrives.
- excess latency: a new packet arrives on time but its first frame is scheduled more than
  `delay + max_excess` from now. That happens after a stall: the backlog that finally
  arrives re-anchors playback, and from then on every frame would be played that much later
  than necessary. The frames still waiting are skipped and the new packet plays `delay` from
  now, so latency cannot keep growing. (A large packet on its own is not a problem: its
  frames are simply played over the time they cover.)

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
            first_new = int(frames.time_ns[0])
            if self._anchor is None or self._starved:
                self._frames = Frames.concatenate([self._frames, frames], self.channels)
                self._anchor = (int(self._frames.time_ns[0]), now_ns + self.delay_ns)
                self._starved = False
                return
            sensor_s0, mono_m0 = self._anchor
            first_new_due = mono_m0 + (first_new - sensor_s0)
            if first_new_due - (now_ns + self.delay_ns) > self.max_excess_ns:
                # Skip the frames still waiting and play the new packet `delay` from now.
                self._frames = frames
                self._anchor = (first_new, now_ns + self.delay_ns)
                self.n_skips += 1
                return
            self._frames = Frames.concatenate([self._frames, frames], self.channels)

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
