"""Stillness detection and gyro bias estimation.

A gyroscope reports a small non-zero rate even when it is not rotating (its bias). If that
offset were fed to the pointer, the cursor would creep across the screen on its own. The
bias can only be measured when we know the true rate is zero, i.e. while the wrist is still.

Stillness here means: over the last `window` frames, both the rotation speed and the user
acceleration stayed below small thresholds. While still, the bias estimate is moved towards
the measured rate with an exponential moving average; during motion it is frozen.

Apple Core Motion already removes the gyro bias before Sensor Logger sends the data
(Phase 0 measured about 0 deg/s), so on the watch this estimate stays near zero. The stage
is still part of the pipeline because a raw IMU would need it.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class StillnessDetector:
    """True once the last `window_frames` frames were all below both thresholds."""

    def __init__(self, window_frames: int, gyro_threshold_rad_s: float, accel_threshold_g: float):
        if window_frames < 1:
            raise ValueError("window_frames must be at least 1")
        self.gyro_threshold = gyro_threshold_rad_s
        self.accel_threshold = accel_threshold_g
        self._quiet: deque[bool] = deque(maxlen=window_frames)

    def reset(self) -> None:
        self._quiet.clear()

    def update(self, gyro: FloatArray, user_accel: FloatArray) -> bool:
        quiet = (
            float(np.linalg.norm(gyro)) < self.gyro_threshold
            and float(np.linalg.norm(user_accel)) < self.accel_threshold
        )
        self._quiet.append(quiet)
        return len(self._quiet) == self._quiet.maxlen and all(self._quiet)


class GyroBiasEstimator:
    """Running gyro bias estimate, updated only while the detector reports stillness.

    Args:
        detector: Decides when the wrist is still.
        sample_hz: Frame rate.
        time_constant_s: Time constant of the moving average while still. The per-frame
            weight is alpha = 1 - exp(-dt / tau), the discrete form of a first-order low-pass.
    """

    def __init__(self, detector: StillnessDetector, sample_hz: float, time_constant_s: float):
        self.detector = detector
        self.alpha = 1.0 - float(np.exp(-1.0 / (sample_hz * time_constant_s)))
        self.bias = np.zeros(3)
        self.is_still = False

    def update(self, gyro: FloatArray, user_accel: FloatArray) -> FloatArray:
        """Feed one frame; return the bias-corrected gyro rate."""
        self.is_still = self.detector.update(gyro, user_accel)
        if self.is_still:
            self.bias += self.alpha * (gyro - self.bias)
        return gyro - self.bias
