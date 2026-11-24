"""Causal IIR filters that keep their state between chunks.

Live data arrives a packet at a time, so a filter must give exactly the same output whether
it sees one long array or the same samples split into chunks. scipy's `sosfilt` supports
this through its `zi` (internal state) argument; this class just carries that state.

Butterworth filters are used because their pass band is maximally flat (no ripple), which
keeps slow hand motion undistorted. They are applied as second-order sections ("sos"),
which is numerically safer than one high-order polynomial.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal

FloatArray = npt.NDArray[np.float64]


class ButterworthFilter:
    """Streaming low-pass or high-pass Butterworth filter over several channels.

    Args:
        kind: "lowpass" or "highpass".
        cutoff_hz: -3 dB frequency.
        sample_hz: Sample rate of the input.
        order: Filter order (steepness of the roll-off).
        n_channels: Number of columns filtered independently.
    """

    def __init__(
        self,
        kind: Literal["lowpass", "highpass"],
        cutoff_hz: float,
        sample_hz: float,
        order: int,
        n_channels: int,
    ) -> None:
        if not 0 < cutoff_hz < sample_hz / 2:
            raise ValueError("cutoff_hz must be between 0 and the Nyquist frequency")
        self.sos = signal.butter(order, cutoff_hz, btype=kind, fs=sample_hz, output="sos")
        self.n_channels = n_channels
        self._state: FloatArray | None = None

    def reset(self) -> None:
        """Forget the past (e.g. after a gap in the data)."""
        self._state = None

    def process(self, x: FloatArray) -> FloatArray:
        """Filter rows of x (shape (N, n_channels)) and return the same shape."""
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            raise ValueError(f"Expected shape (N, {self.n_channels}), got {x.shape}")
        if len(x) == 0:
            return x.copy()
        if self._state is None:
            # Start as if the first sample had been constant forever, so a low-pass does not
            # begin with a jump from zero. (For a high-pass this correctly starts at zero.)
            zi = signal.sosfilt_zi(self.sos)  # shape (n_sections, 2), for a unit step
            self._state = zi[:, :, np.newaxis] * x[0][np.newaxis, np.newaxis, :]
        y, self._state = signal.sosfilt(self.sos, x, axis=0, zi=self._state)
        return y


class GapAwareFilter:
    """Wraps a ButterworthFilter for data with invalid frames.

    Each stretch of consecutive valid frames is filtered as one continuous signal; the
    filter state is reset at the start of every stretch that follows a gap, so it never
    filters across missing data. Invalid frames come out as NaN. Works across batches.
    """

    def __init__(self, inner: ButterworthFilter) -> None:
        self.inner = inner
        self._previous_frame_valid = False

    def process(self, x: FloatArray, valid: npt.NDArray[np.bool_]) -> FloatArray:
        y = np.full(x.shape, np.nan)
        for start, stop in valid_runs(valid):
            if start > 0 or not self._previous_frame_valid:
                self.inner.reset()
            y[start:stop] = self.inner.process(x[start:stop])
        if len(valid) > 0:
            self._previous_frame_valid = bool(valid[-1])
        return y


def valid_runs(valid: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[start, stop) index ranges of consecutive True values."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs
