"""Numerical tests for the DSP building blocks against known signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from airhand.config import DspConfig
from airhand.dsp import quaternion as quat
from airhand.dsp.filters import ButterworthFilter
from airhand.dsp.madgwick import Madgwick
from airhand.dsp.one_euro import OneEuroFilter, smoothing_factor
from airhand.dsp.processor import DspProcessor
from airhand.dsp.stillness import GyroBiasEstimator, StillnessDetector
from airhand.ingest.chunk import Frames

REPO_ROOT = Path(__file__).resolve().parents[1]
FS = 100.0


# --- quaternions -----------------------------------------------------------------------


def test_rotation_vector_quarter_turn_about_z_maps_x_to_y() -> None:
    q = quat.from_rotation_vector(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(quat.rotate(q, np.array([1.0, 0.0, 0.0])), [0, 1, 0], atol=1e-12)


def test_multiply_composes_rotations() -> None:
    a = quat.from_rotation_vector(np.array([0.3, -0.2, 0.5]))
    b = quat.from_rotation_vector(np.array([-0.1, 0.4, 0.2]))
    v = np.array([0.2, -0.7, 0.4])
    np.testing.assert_allclose(
        quat.rotate(quat.multiply(a, b), v), quat.rotate(a, quat.rotate(b, v)), atol=1e-12
    )


def test_heading_and_tilt_differences_separate_cleanly() -> None:
    tilt = quat.from_rotation_vector(np.radians([20.0, 0.0, 0.0]))
    yaw = quat.from_rotation_vector(np.radians([0.0, 0.0, 30.0]))
    a = quat.multiply(yaw, tilt)  # same tilt, heading turned by 30 degrees
    assert quat.heading_difference_deg(a, tilt) == pytest.approx(30.0, abs=1e-9)
    assert quat.tilt_difference_deg(a, tilt) == pytest.approx(0.0, abs=1e-4)
    assert quat.tilt_difference_deg(tilt, quat.IDENTITY) == pytest.approx(20.0, abs=1e-9)


# --- Butterworth filters --------------------------------------------------------------


def sine(freq_hz: float, seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return np.sin(2 * np.pi * freq_hz * t)[:, np.newaxis]


def steady_state_gain(filt: ButterworthFilter, x: np.ndarray) -> float:
    y = filt.process(x)
    half = len(x) // 2  # ignore the start-up transient
    return float(np.std(y[half:]) / np.std(x[half:]))


def test_lowpass_passes_hand_motion_and_blocks_noise() -> None:
    def lp() -> ButterworthFilter:
        return ButterworthFilter("lowpass", 15.0, FS, 2, n_channels=1)

    assert steady_state_gain(lp(), sine(2.0)) == pytest.approx(1.0, abs=0.01)
    assert steady_state_gain(lp(), sine(15.0)) == pytest.approx(1 / np.sqrt(2), abs=0.03)
    assert steady_state_gain(lp(), sine(45.0)) < 0.1


def test_highpass_removes_constant_offset() -> None:
    hp = ButterworthFilter("highpass", 5.0, FS, 2, n_channels=1)
    y = hp.process(np.full((300, 1), 0.7))
    assert np.max(np.abs(y)) < 1e-9


def test_lowpass_starts_without_a_jump_on_constant_input() -> None:
    lp = ButterworthFilter("lowpass", 15.0, FS, 2, n_channels=3)
    x = np.tile([0.1, -0.2, 0.3], (50, 1))
    np.testing.assert_allclose(lp.process(x), x, atol=1e-12)


def test_filter_output_does_not_depend_on_chunking() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    whole = ButterworthFilter("lowpass", 10.0, FS, 4, n_channels=3).process(x)
    chunked_filter = ButterworthFilter("lowpass", 10.0, FS, 4, n_channels=3)
    parts = [chunked_filter.process(x[a:b]) for a, b in [(0, 7), (7, 151), (151, 152), (152, 500)]]
    np.testing.assert_allclose(np.vstack(parts), whole, atol=1e-12)


# --- stillness and gyro bias ------------------------------------------------------------


def make_bias_estimator() -> GyroBiasEstimator:
    detector = StillnessDetector(50, gyro_threshold_rad_s=np.radians(2.0), accel_threshold_g=0.02)
    return GyroBiasEstimator(detector, FS, time_constant_s=1.0)


def test_bias_is_learned_while_still() -> None:
    est = make_bias_estimator()
    true_bias = np.array([0.01, -0.005, 0.002])  # rad/s, well under the stillness threshold
    for _ in range(int(10 * FS)):
        corrected = est.update(true_bias, np.zeros(3))
    np.testing.assert_allclose(est.bias, true_bias, atol=1e-5)
    np.testing.assert_allclose(corrected, 0.0, atol=1e-5)


def test_bias_is_frozen_during_motion() -> None:
    est = make_bias_estimator()
    for _ in range(int(5 * FS)):
        est.update(np.array([0.01, 0.0, 0.0]), np.zeros(3))
    learned = est.bias.copy()
    for _ in range(int(2 * FS)):
        est.update(np.array([1.5, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]))  # waving the arm
    np.testing.assert_array_equal(est.bias, learned)
    assert not est.is_still


# --- Madgwick ---------------------------------------------------------------------------


def specific_force_for(q: np.ndarray) -> np.ndarray:
    """What an accelerometer at rest measures in orientation q: world up in the body frame."""
    return -quat.world_down_in_body(q)


def test_madgwick_converges_to_true_tilt_from_a_wrong_start() -> None:
    truth = quat.from_rotation_vector(np.radians([30.0, -15.0, 0.0]))
    filt = Madgwick(beta=0.1)  # starts at identity, 30+ degrees off
    for _ in range(int(20 * FS)):
        filt.update(np.zeros(3), specific_force_for(truth), 1 / FS)
    assert quat.tilt_difference_deg(filt.q, truth) < 0.5


def test_madgwick_integrates_a_constant_yaw_rate() -> None:
    filt = Madgwick(beta=0.04)
    rate = np.radians(90.0)  # deg/s about the vertical axis, device lying flat
    for _ in range(int(FS)):  # one second
        filt.update(np.array([0.0, 0.0, rate]), np.array([0.0, 0.0, 1.0]), 1 / FS)
    assert quat.heading_difference_deg(filt.q, quat.IDENTITY) == pytest.approx(90.0, abs=0.5)
    assert quat.tilt_difference_deg(filt.q, quat.IDENTITY) < 0.1


# --- One Euro ---------------------------------------------------------------------------


def test_one_euro_with_beta_zero_is_a_first_order_low_pass() -> None:
    filt = OneEuroFilter(FS, min_cutoff_hz=2.0, beta=0.0, d_cutoff_hz=1.0, n_channels=1)
    alpha = float(smoothing_factor(2.0, 1 / FS))
    filt.filter(np.array([0.0]))
    outputs = [filt.filter(np.array([1.0]))[0] for _ in range(50)]
    expected = [1 - (1 - alpha) ** n for n in range(1, 51)]
    np.testing.assert_allclose(outputs, expected, rtol=1e-12)


def test_one_euro_holds_a_constant_signal() -> None:
    filt = OneEuroFilter(FS, 1.0, 0.01, 1.0, n_channels=2)
    for _ in range(100):
        y = filt.filter(np.array([3.0, -4.0]))
    np.testing.assert_allclose(y, [3.0, -4.0])


def test_one_euro_beta_reduces_lag_during_fast_motion() -> None:
    ramp = np.arange(200) * 10.0  # 1000 units/s

    def final_lag(beta: float) -> float:
        filt = OneEuroFilter(FS, 1.0, beta, 1.0, n_channels=1)
        y = [filt.filter(np.array([v]))[0] for v in ramp]
        return float(ramp[-1] - y[-1])

    assert final_lag(0.05) < 0.2 * final_lag(0.0)


# --- DSP processor ----------------------------------------------------------------------

CHANNELS = (
    "rotationRateX", "rotationRateY", "rotationRateZ",
    "accelerationX", "accelerationY", "accelerationZ",
    "gravityX", "gravityY", "gravityZ",
    "quaternionW", "quaternionX", "quaternionY", "quaternionZ",
)  # fmt: skip


def synthetic_frames(n: int, invalid: slice | None = None) -> Frames:
    """Watch lying flat, turning slowly about the vertical axis, with noise."""
    rng = np.random.default_rng(1)
    t = np.arange(n, dtype=np.int64) * 10_000_000
    values = np.zeros((n, len(CHANNELS)))
    values[:, 0:3] = rng.normal(0, 0.01, (n, 3))
    values[:, 2] += 0.5
    values[:, 3:6] = rng.normal(0, 0.005, (n, 3))
    values[:, 8] = -1.0
    values[:, 9] = 1.0
    valid = np.ones(n, dtype=bool)
    if invalid is not None:
        valid[invalid] = False
        values[invalid] = np.nan
    return Frames(time_ns=t, values=values, valid=valid, channels=CHANNELS)


def dsp_processor() -> DspProcessor:
    return DspProcessor(DspConfig.from_yaml(REPO_ROOT / "configs" / "dsp.yaml"), FS)


def test_processor_output_does_not_depend_on_batching() -> None:
    frames = synthetic_frames(400, invalid=slice(150, 170))
    whole = dsp_processor().process(frames)
    proc = dsp_processor()
    cuts = [0, 33, 160, 161, 299, 400]
    parts = [
        proc.process(
            Frames(
                frames.time_ns[a:b], frames.values[a:b], frames.valid[a:b], frames.channels
            )
        )
        for a, b in zip(cuts[:-1], cuts[1:], strict=True)
    ]
    for name in ("gyro", "gyro_bias", "madgwick_q"):
        np.testing.assert_allclose(
            np.vstack([getattr(p, name) for p in parts]), getattr(whole, name), atol=1e-12
        )


def test_processor_marks_gaps_as_nan_and_recovers() -> None:
    out = dsp_processor().process(synthetic_frames(400, invalid=slice(150, 170)))
    assert np.isnan(out.gyro[150:170]).all()
    assert np.isfinite(out.gyro[170:]).all()
    assert np.isfinite(out.gyro[:150]).all()


def test_processor_madgwick_tracks_the_turn() -> None:
    out = dsp_processor().process(synthetic_frames(400))
    turned = quat.heading_difference_deg(out.madgwick_q[-1], out.madgwick_q[0])
    assert turned == pytest.approx(np.degrees(0.5 * 3.99), abs=1.0)
