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
