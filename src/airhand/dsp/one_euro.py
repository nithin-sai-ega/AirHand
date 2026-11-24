"""One Euro filter (Casiez, Roussel and Vogel, CHI 2012).

The pointer has two competing needs:
- when the hand moves slowly (aiming at a small target), jitter must be smoothed away;
- when the hand moves fast, the cursor must not lag behind.

A fixed low-pass filter can only trade one for the other. The One Euro filter is a
first-order low-pass whose cutoff frequency rises with the signal's speed:

    cutoff = min_cutoff + beta * |speed|

so it smooths heavily at rest and gets out of the way during fast motion. The speed itself
is estimated from the derivative, low-passed with a fixed cutoff `d_cutoff`.

A first-order low-pass with cutoff fc at sample period T has the update
    y = y_prev + alpha * (x - y_prev),   alpha = 1 / (1 + tau / T),   tau = 1 / (2 pi fc).

This implementation filters each channel independently and assumes a fixed sample rate
(the resampled grid), so T is a constant.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def smoothing_factor(cutoff_hz: FloatArray | float, sample_period_s: float) -> FloatArray:
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff_hz, dtype=np.float64))
    return 1.0 / (1.0 + tau / sample_period_s)


class OneEuroFilter:
    def __init__(
        self,
        sample_hz: float,
        min_cutoff_hz: float,
        beta: float,
        d_cutoff_hz: float,
        n_channels: int,
    ) -> None:
        self.period = 1.0 / sample_hz
        self.min_cutoff = min_cutoff_hz
        self.beta = beta
        self.d_alpha = smoothing_factor(d_cutoff_hz, self.period)
        self.n_channels = n_channels
        self._x_prev: FloatArray | None = None
        self._dx_prev = np.zeros(n_channels)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = np.zeros(self.n_channels)

    def filter(self, x: FloatArray) -> FloatArray:
        """Filter one sample (shape (n_channels,)) and return the filtered value."""
        if self._x_prev is None:
            self._x_prev = x.astype(np.float64).copy()
            return self._x_prev.copy()

        # 1. Speed estimate: raw derivative, smoothed with a fixed cutoff.
        dx = (x - self._x_prev) / self.period
        dx_hat = self._dx_prev + self.d_alpha * (dx - self._dx_prev)

        # 2. Cutoff grows with speed, then an ordinary first-order low-pass step.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = smoothing_factor(cutoff, self.period)
        x_hat = self._x_prev + alpha * (x - self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat.copy()
