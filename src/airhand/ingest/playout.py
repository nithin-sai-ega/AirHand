"""Playout (jitter) buffer: turns bursty arrivals into a steady stream of frames.

Sensor Logger delivers about one packet per second (free tier), each holding 1.0-1.5 s of
100 Hz samples. Applying a whole packet at once would make the cursor jump once per second.
Every sample does arrive, just late, so the fix is the one voice and video calls use for
network jitter: hold each frame back and release it at a fixed time after it was recorded.

Timing model. For every packet we note its *offset*: laptop arrival time minus the sensor
time of its newest frame (this mixes two clocks, which is fine: only differences matter).
The smallest offset seen recently, the *floor*, is the fastest the network has delivered.
A frame recorded at sensor time t is played at

    t + floor + delay

so frames come out exactly as far apart as they were recorded (10 ms on the 100 Hz grid),
and every frame has the same latency. A frame arrives in time if its packet was not later
than the floor by more than `delay` minus the packet's length, so `delay` must exceed the
longest packet (1.53 s measured) plus the arrival jitter.

Why a sliding floor instead of fixing the schedule once: the watch's timestamps and the
laptop clock do not run at exactly the same rate (Phase 0 measured arrival times slipping by
about 1 ms per second). A schedule fixed at the first packet would slowly run dry and stall
every few minutes. The floor is the minimum over the last `offset_window_s`, so it follows
slow drift. It also forgets the effect of an unusually early packet after the window.

When a packet is later than the margin allows, its overdue frames are released at the next
tick (the cursor catches up quickly instead of falling further behind) and the event is
counted as an underrun.

The buffer is used by two threads (network receiver pushes, pointer loop pops), so every
public method takes a lock. Times are passed in explicitly, which makes it testable with a
fake clock.
"""

from __future__ import annotations

import threading
from collections import deque

from airhand.ingest.chunk import Frames

NS_PER_S = 1_000_000_000


class PlayoutBuffer:
    def __init__(
        self,
        channels: tuple[str, ...],
        frame_period_ns: int,
        delay_s: float,
        offset_window_s: float,
    ) -> None:
        if delay_s <= 0 or offset_window_s <= 0:
            raise ValueError("delay_s and offset_window_s must be positive")
        self.channels = channels
        self.frame_period_ns = frame_period_ns
        self.delay_ns = round(delay_s * NS_PER_S)
        self.offset_window_ns = round(offset_window_s * NS_PER_S)
        self.n_underruns = 0
        self._frames = Frames.empty(channels)
        self._offsets: deque[tuple[int, int]] = deque()  # (arrival time, offset), recent
        self._last_released_ns: int | None = None
        self._starved = False
        self._lock = threading.Lock()

    def push(self, frames: Frames, now_ns: int) -> None:
        """Add newly resampled frames that arrived at `now_ns`.

        Frames must continue in time order; the resampler guarantees this (it drops samples
        that are not newer than the last one).
        """
        if len(frames) == 0:
            return
        with self._lock:
            self._frames = Frames.concatenate([self._frames, frames], self.channels)
            self._offsets.append((now_ns, now_ns - int(frames.time_ns[-1])))
            while now_ns - self._offsets[0][0] > self.offset_window_ns:
                self._offsets.popleft()

    def _floor_ns(self) -> int:
        return min(offset for _, offset in self._offsets)

    def pop_due(self, now_ns: int) -> Frames:
        """Remove and return every frame whose play time is at or before `now_ns`."""
        with self._lock:
            if not self._offsets:
                return Frames.empty(self.channels)
            # Invert "play time = t + floor + delay": frames with t <= due_until are due.
            due_until = now_ns - self._floor_ns() - self.delay_ns
            n_due = int((self._frames.time_ns <= due_until).sum())
            out = self._frames.slice(0, n_due)
            self._frames = self._frames.slice(n_due, len(self._frames))
            if n_due > 0:
                self._last_released_ns = int(out.time_ns[-1])

            # Underrun: the next frame should already be playing but has not arrived.
            waiting = (
                len(self._frames) == 0
                and self._last_released_ns is not None
                and due_until >= self._last_released_ns + self.frame_period_ns
            )
            if waiting and not self._starved:
                self.n_underruns += 1
            self._starved = waiting
            return out

    @property
    def buffered_s(self) -> float:
        """Seconds of data waiting to be played."""
        with self._lock:
            if len(self._frames) == 0:
                return 0.0
            span = int(self._frames.time_ns[-1] - self._frames.time_ns[0]) + self.frame_period_ns
            return span / NS_PER_S

    @property
    def starved(self) -> bool:
        """True while frames are due but have not arrived."""
        with self._lock:
            return self._starved
