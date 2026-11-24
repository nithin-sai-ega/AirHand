"""The pointer stage: DSP output in, integer cursor moves out.

Per frame:
    gyro + gravity --> WristToScreen --> angular velocity (deg/s, screen directions)
                   --> GainCurve      --> cursor velocity (px/s)
                   --> integrate      --> intended cursor position (px, floating point)
                   --> One Euro       --> smoothed position
                   --> round          --> integer position; dx, dy = change since last frame

The One Euro filter is applied to position, its intended use: at low speed it removes
jitter, at high speed it adapts to avoid lag. Rounding the smoothed *position* (instead of
rounding each small step) means sub-pixel motion is never lost: ten moves of 0.3 px add up
to 3 px.

user accel --> high-pass --> |.| --> DoubleKnockDetector --> toggles the clutch

While the clutch is disengaged the cursor gets no moves and the integrator and One Euro
filter are reset, so re-engaging never makes the cursor jump.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from airhand.config import PointerConfig
from airhand.dsp.filters import ButterworthFilter, GapAwareFilter
from airhand.dsp.one_euro import OneEuroFilter
from airhand.dsp.processor import DspOutput
from airhand.pointer.clutch import Clutch, DoubleKnockDetector, KnockTiming
from airhand.pointer.mapping import WristToScreen
from airhand.pointer.transfer import GainCurve

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PointerOutput:
    """Per-frame pointer results; every array has one row per input frame."""

    time_ns: IntArray
    angular_velocity_dps: FloatArray  # (N, 2) screen directions
    velocity_px_s: FloatArray  # (N, 2) after the gain curve
    position_raw: FloatArray  # (N, 2) integrated velocity (NaN when disengaged)
    position_smooth: FloatArray  # (N, 2) after One Euro (NaN when disengaged)
    dx: IntArray  # (N,) integer pixel moves to send
    dy: IntArray
    engaged: BoolArray  # (N,) clutch state used for this frame
    knock_signal_g: FloatArray  # (N,) high-passed acceleration magnitude
    toggled: BoolArray  # (N,) a double knock toggled the clutch on this frame

    def __len__(self) -> int:
        return len(self.time_ns)


class PointerProcessor:
    def __init__(self, config: PointerConfig, sample_hz: float) -> None:
        self.config = config
        self.sample_hz = sample_hz
        self.mapping = WristToScreen(config.forearm_axis, config.min_horizontal_norm)
        self.gain = GainCurve(
            dead_zone_dps=config.dead_zone_dps,
            low_px_per_deg=config.gain_low_px_per_deg,
            high_px_per_deg=config.gain_high_px_per_deg,
            mid_dps=config.gain_mid_dps,
            width_dps=config.gain_width_dps,
        )
        self.one_euro = OneEuroFilter(
            sample_hz,
            config.one_euro_min_cutoff_hz,
            config.one_euro_beta,
            config.one_euro_d_cutoff_hz,
            n_channels=2,
        )
        self.knock_highpass = GapAwareFilter(
            ButterworthFilter("highpass", config.knock_highpass_hz, sample_hz, 2, n_channels=3)
        )
        self.knocks = DoubleKnockDetector(
            KnockTiming(
                threshold_g=config.knock_threshold_g,
                min_spacing_s=config.knock_min_spacing_s,
                pair_min_s=config.knock_pair_min_s,
                pair_max_s=config.knock_pair_max_s,
                quiet_before_s=config.knock_quiet_before_s,
                quiet_after_s=config.knock_quiet_after_s,
                cooldown_s=config.knock_cooldown_s,
            )
        )
        self.clutch = Clutch(config.start_engaged)
        # Motion is integrated relative to where the clutch was engaged, starting at 0.
        self._position = np.zeros(2)
        self._last_emitted = np.zeros(2, dtype=np.int64)

    def toggle_clutch(self) -> None:
        """Hotkey path: toggle the clutch between batches."""
        self.clutch.toggle()

    def _release(self) -> None:
        """Forget the motion state so the next engaged frame starts from rest."""
        self._position = np.zeros(2)
        self._last_emitted = np.zeros(2, dtype=np.int64)
        self.one_euro.reset()

    def process(self, dsp: DspOutput) -> PointerOutput:
        n = len(dsp)
        angular = self.mapping.angular_velocity_dps(dsp.gyro, dsp.gravity)
        velocity = self.gain.apply(np.nan_to_num(angular))
        velocity[~dsp.valid] = np.nan
        knock_signal = np.linalg.norm(
            self.knock_highpass.process(dsp.user_accel, dsp.valid), axis=1
        )

        position_raw = np.full((n, 2), np.nan)
        position_smooth = np.full((n, 2), np.nan)
        dx = np.zeros(n, dtype=np.int64)
        dy = np.zeros(n, dtype=np.int64)
        engaged = np.zeros(n, dtype=bool)
        toggled = np.zeros(n, dtype=bool)

        for i in range(n):
            if not dsp.valid[i]:
                self.knocks.reset()
                self._release()
                continue
            if self.knocks.update(int(dsp.time_ns[i]), float(knock_signal[i])):
                self.clutch.toggle()
                toggled[i] = True
            engaged[i] = self.clutch.engaged
            if not engaged[i]:
                self._release()
                continue

            self._position = self._position + velocity[i] / self.sample_hz
            smooth = self.one_euro.filter(self._position)
            emitted = np.round(smooth).astype(np.int64)
            dx[i], dy[i] = emitted - self._last_emitted
            self._last_emitted = emitted
            position_raw[i] = self._position
            position_smooth[i] = smooth

        return PointerOutput(
            time_ns=dsp.time_ns.copy(),
            angular_velocity_dps=angular,
            velocity_px_s=velocity,
            position_raw=position_raw,
            position_smooth=position_smooth,
            dx=dx,
            dy=dy,
            engaged=engaged,
            knock_signal_g=knock_signal,
            toggled=toggled,
        )
