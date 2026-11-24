"""Clutch: whether wrist motion currently moves the cursor.

Like lifting a mouse off the desk, the clutch lets the user move the arm without moving the
cursor. It starts disengaged (so starting the program never sends the cursor flying) and is
toggled by a double knock (knuckles on the desk twice) or by a hotkey.

Double-knock detection is plain signal processing, no learning:
1. High-pass the user acceleration, so slow arm motion is removed and only sharp events
   remain; the knock signal is the magnitude of that vector, in g.
2. A knock starts when the signal crosses `threshold`. Above-threshold frames closer than
   `min_spacing` to the previous ones belong to the same knock (a knock rings for a few
   frames).
3. Two knocks `pair_min`..`pair_max` apart form a double knock, but only if there was no knock
   in the `quiet_before` window before the first one and none arrives within `quiet_after`
   of the second. This rejects shaking, typing and triple knocks.
4. After a toggle, knocks are ignored for `cooldown`.

The detector waits `quiet_after` before confirming, so a toggle is reported that long after
the second knock. All timing uses frame timestamps, so replays are deterministic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class KnockTiming:
    threshold_g: float
    min_spacing_s: float
    pair_min_s: float
    pair_max_s: float
    quiet_before_s: float
    quiet_after_s: float
    cooldown_s: float


class DoubleKnockDetector:
    def __init__(self, timing: KnockTiming) -> None:
        self.t = timing
        self._knocks: deque[int] = deque()  # start times (ns) of recent knocks
        self._last_above_ns: int | None = None
        self._pending_ns: int | None = None  # time of a second knock awaiting confirmation
        self._cooldown_until_ns = -(2**62)

    def reset(self) -> None:
        """Forget partial patterns (e.g. after a gap in the data)."""
        self._knocks.clear()
        self._last_above_ns = None
        self._pending_ns = None

    def update(self, time_ns: int, signal_g: float) -> bool:
        """Feed one frame; return True on the frame where a double knock is confirmed."""
        new_knock = False
        if signal_g >= self.t.threshold_g:
            spacing_ns = self.t.min_spacing_s * NS_PER_S
            if self._last_above_ns is None or time_ns - self._last_above_ns > spacing_ns:
                new_knock = True
            self._last_above_ns = time_ns

        if time_ns < self._cooldown_until_ns:
            return False  # knocks during the cooldown are ignored entirely

        if new_knock:
            if self._pending_ns is not None:
                self._pending_ns = None  # a third knock: not a clean double knock
            self._knocks.append(time_ns)
            self._drop_old_knocks(time_ns)
            if self._is_clean_pair():
                self._pending_ns = time_ns

        if (
            self._pending_ns is not None
            and time_ns - self._pending_ns >= self.t.quiet_after_s * NS_PER_S
        ):
            self._pending_ns = None
            self._knocks.clear()
            self._cooldown_until_ns = time_ns + round(self.t.cooldown_s * NS_PER_S)
            return True
        return False

    def _drop_old_knocks(self, now_ns: int) -> None:
        horizon_ns = (self.t.pair_max_s + self.t.quiet_before_s) * NS_PER_S
        while self._knocks and now_ns - self._knocks[0] > horizon_ns:
            self._knocks.popleft()

    def _is_clean_pair(self) -> bool:
        if len(self._knocks) < 2:
            return False
        second, first = self._knocks[-1], self._knocks[-2]
        gap_s = (second - first) / NS_PER_S
        if not self.t.pair_min_s <= gap_s <= self.t.pair_max_s:
            return False
        if len(self._knocks) >= 3:
            before_s = (first - self._knocks[-3]) / NS_PER_S
            if before_s < self.t.quiet_before_s:
                return False
        return True


class Clutch:
    def __init__(self, start_engaged: bool) -> None:
        self.engaged = start_engaged
        self.n_toggles = 0

    def toggle(self) -> None:
        self.engaged = not self.engaged
        self.n_toggles += 1
