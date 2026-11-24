"""The DSP stage: resampled frames in, cleaned signals and orientation out.

Per frame:
    gyro --> bias estimator (while still) --> low-pass ------------------> gyro (for pointer)
    gyro + gravity + user accel ----------> Madgwick ---------------------> madgwick_q
    gravity, user accel, device quaternion pass through unchanged

Invalid frames (gaps the resampler could not fill) produce NaN outputs and reset the
stateful filters, so nothing is computed from made-up data and no filter "rings" across a
gap.

The processor works on whole `Frames` batches and returns arrays, which keeps it easy to test
offline and cheap to call from the live loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from airhand.config import DspConfig
from airhand.dsp.filters import ButterworthFilter, GapAwareFilter
from airhand.dsp.madgwick import Madgwick
from airhand.dsp.stillness import GyroBiasEstimator, StillnessDetector
from airhand.ingest.chunk import Frames

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class DspOutput:
    """Per-frame DSP results; every array has one row per input frame."""

    time_ns: IntArray
    valid: BoolArray
    gyro_raw: FloatArray  # (N, 3) rad/s, as received
    gyro: FloatArray  # (N, 3) rad/s, bias-corrected and low-passed
    gyro_bias: FloatArray  # (N, 3) rad/s, current bias estimate
    still: BoolArray  # (N,)
    gravity: FloatArray  # (N, 3) g, points down, body frame
    user_accel: FloatArray  # (N, 3) g, body frame
    device_q: FloatArray  # (N, 4) the watch's own orientation
    madgwick_q: FloatArray  # (N, 4) our orientation estimate

    def __len__(self) -> int:
        return len(self.time_ns)


def _columns(frames: Frames, names: tuple[str, ...]) -> FloatArray:
    try:
        index = [frames.channels.index(name) for name in names]
    except ValueError as exc:
        raise ValueError(f"Frames are missing a channel from {names}") from exc
    return frames.values[:, index]


class DspProcessor:
    def __init__(self, config: DspConfig, sample_hz: float) -> None:
        self.config = config
        self.sample_hz = sample_hz
        window = max(1, round(config.stillness_window_s * sample_hz))
        self.bias = GyroBiasEstimator(
            StillnessDetector(
                window,
                gyro_threshold_rad_s=float(np.radians(config.stillness_gyro_dps)),
                accel_threshold_g=config.stillness_accel_g,
            ),
            sample_hz,
            config.bias_time_constant_s,
        )
        self.lowpass = GapAwareFilter(
            ButterworthFilter(
                "lowpass", config.lowpass_cutoff_hz, sample_hz, config.lowpass_order, n_channels=3
            )
        )
        self.madgwick: Madgwick | None = None  # created on the first valid frame

    def process(self, frames: Frames) -> DspOutput:
        n = len(frames)
        gyro_raw = _columns(frames, self.config.gyro_channels)
        user_accel = _columns(frames, self.config.user_accel_channels)
        gravity = _columns(frames, self.config.gravity_channels)
        device_q = _columns(frames, self.config.quaternion_channels)

        corrected = np.full((n, 3), np.nan)
        bias = np.full((n, 3), np.nan)
        still = np.zeros(n, dtype=bool)
        madgwick_q = np.full((n, 4), np.nan)
        dt = 1.0 / self.sample_hz

        for i in range(n):
            if not frames.valid[i]:
                # Stillness must be proven again after a gap. Madgwick keeps its orientation:
                # the pose after a short gap is close to the last one, and the accelerometer
                # pulls the tilt back quickly if it is not.
                self.bias.detector.reset()
                continue
            corrected[i] = self.bias.update(gyro_raw[i], user_accel[i])
            bias[i] = self.bias.bias
            still[i] = self.bias.is_still
            if self.madgwick is None:
                self.madgwick = Madgwick(self.config.madgwick_beta, initial=device_q[i])
            else:
                specific_force = -(gravity[i] + user_accel[i])
                self.madgwick.update(corrected[i], specific_force, dt)
            madgwick_q[i] = self.madgwick.q

        gyro = self.lowpass.process(corrected, frames.valid)

        return DspOutput(
            time_ns=frames.time_ns.copy(),
            valid=frames.valid.copy(),
            gyro_raw=gyro_raw,
            gyro=gyro,
            gyro_bias=bias,
            still=still,
            gravity=gravity,
            user_accel=user_accel,
            device_q=device_q,
            madgwick_q=madgwick_q,
        )

