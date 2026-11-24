"""Tests for the pointer stage: mapping, gain curve, knock clutch and the full processor."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from airhand.config import PointerConfig
from airhand.dsp import quaternion as quat
from airhand.dsp.processor import DspOutput
from airhand.pointer.clutch import DoubleKnockDetector, KnockTiming
from airhand.pointer.mapping import WristToScreen
from airhand.pointer.processor import PointerProcessor
from airhand.pointer.transfer import GainCurve

REPO_ROOT = Path(__file__).resolve().parents[1]
FS = 100.0
NS = 1_000_000_000

# --- mapping ----------------------------------------------------------------------------

FLAT_GRAVITY = np.array([[0.0, 0.0, -1.0]])  # watch face up, body z = world up


def test_turning_left_moves_the_cursor_left() -> None:
    m = WristToScreen((1.0, 0.0, 0.0), 0.3)
    w = m.angular_velocity_dps(np.array([[0.0, 0.0, np.radians(10.0)]]), FLAT_GRAVITY)
    np.testing.assert_allclose(w, [[-10.0, 0.0]], atol=1e-12)


def test_tilting_the_hand_up_moves_the_cursor_up() -> None:
    m = WristToScreen((1.0, 0.0, 0.0), 0.3)
    # Hand up = forearm (+x) turning towards up (+z) = negative rotation about left (+y).
    w = m.angular_velocity_dps(np.array([[0.0, -np.radians(10.0), 0.0]]), FLAT_GRAVITY)
    np.testing.assert_allclose(w, [[0.0, -10.0]], atol=1e-12)


@pytest.mark.parametrize("roll_deg", [0.0, 35.0, 90.0, -60.0])
def test_mapping_does_not_depend_on_wrist_roll(roll_deg: float) -> None:
    """The same world-frame motion gives the same cursor motion however the wrist is rolled."""
    m = WristToScreen((1.0, 0.0, 0.0), 0.3)
    roll = quat.from_rotation_vector(np.radians([roll_deg, 0.0, 0.0]))  # about the forearm
    gravity_body = quat.world_down_in_body(roll)[np.newaxis, :]
    world_rate = np.radians(np.array([0.0, -5.0, 10.0]))  # turn left 10 deg/s, hand up 5
    body_rate = quat.rotate(quat.conjugate(roll), world_rate)[np.newaxis, :]
    w = m.angular_velocity_dps(body_rate, gravity_body)
    np.testing.assert_allclose(w, [[-10.0, -5.0]], atol=1e-9)


def test_wrist_roll_alone_does_not_move_the_cursor() -> None:
    m = WristToScreen((1.0, 0.0, 0.0), 0.3)
    w = m.angular_velocity_dps(np.array([[np.radians(90.0), 0.0, 0.0]]), FLAT_GRAVITY)
    np.testing.assert_allclose(w, [[0.0, 0.0]], atol=1e-12)


def test_vertical_forearm_disables_vertical_motion() -> None:
    m = WristToScreen((1.0, 0.0, 0.0), 0.3)
    gravity = np.array([[-1.0, 0.0, 0.0]])  # forearm pointing straight up
    w = m.angular_velocity_dps(np.array([[0.0, np.radians(20.0), 0.0]]), gravity)
    assert w[0, 1] == 0.0


# --- gain curve -------------------------------------------------------------------------

CURVE = GainCurve(
    dead_zone_dps=1.5, low_px_per_deg=3.0, high_px_per_deg=20.0, mid_dps=40.0, width_dps=15.0
)


def test_dead_zone_outputs_zero_without_a_jump() -> None:
    v = CURVE.apply(np.array([[1.0, 0.0], [0.0, -1.5], [1.6, 0.0]]))
    np.testing.assert_allclose(v[:2], 0.0)
    assert 0 < v[2, 0] < 1.0  # just above the dead zone: tiny, not a jump


def test_cursor_speed_increases_with_wrist_speed_and_keeps_direction() -> None:
    speeds = np.linspace(0, 300, 301)
    out = np.linalg.norm(CURVE.apply(np.column_stack([speeds, np.zeros_like(speeds)])), axis=1)
    assert np.all(np.diff(out) >= 0)
    v = CURVE.apply(np.array([[30.0, -40.0]]))
    np.testing.assert_allclose(v[0] / np.linalg.norm(v[0]), [0.6, -0.8])


def test_gain_goes_from_low_to_high() -> None:
    assert 3.0 < CURVE.gain(np.array(0.0)) < 4.5
    assert CURVE.gain(np.array(40.0)) == pytest.approx((3.0 + 20.0) / 2)  # mid point
    assert CURVE.gain(np.array(200.0)) == pytest.approx(20.0, abs=0.01)


# --- double knock -----------------------------------------------------------------------

TIMING = KnockTiming(
    threshold_g=0.6,
    min_spacing_s=0.08,
    pair_min_s=0.12,
    pair_max_s=0.5,
    quiet_before_s=0.4,
    quiet_after_s=0.3,
    cooldown_s=1.0,
)


def knock_toggles(knock_times_s: list[float], duration_s: float = 6.0) -> list[float]:
    """Feed a 100 Hz signal with 2-frame spikes at the given times; return toggle times."""
    det = DoubleKnockDetector(TIMING)
    n = int(duration_s * FS)
    signal = np.full(n, 0.05)
    for t in knock_times_s:
        i = round(t * FS)
        signal[i : i + 2] = 1.5
    return [i / FS for i in range(n) if det.update(round(i * NS / FS), float(signal[i]))]


def test_clean_double_knock_toggles_after_the_quiet_period() -> None:
    assert knock_toggles([1.0, 1.25]) == [pytest.approx(1.25 + 0.3)]


def test_single_and_triple_knocks_do_not_toggle() -> None:
    assert knock_toggles([1.0]) == []
    assert knock_toggles([1.0, 1.25, 1.45]) == []


def test_knocks_too_close_or_too_far_apart_do_not_toggle() -> None:
    assert knock_toggles([1.0, 1.05]) == []  # merged into one knock
    assert knock_toggles([1.0, 1.8]) == []


def test_knock_right_before_the_pair_rejects_it() -> None:
    assert knock_toggles([0.8, 1.0, 1.25]) == []


def test_cooldown_ignores_knocks_right_after_a_toggle() -> None:
    toggles = knock_toggles([1.0, 1.25, 1.8, 2.0, 4.0, 4.25])
    assert toggles == [pytest.approx(1.55), pytest.approx(4.55)]


# --- full processor ---------------------------------------------------------------------


def pointer_config(**overrides: object) -> PointerConfig:
    config = PointerConfig.from_yaml(REPO_ROOT / "configs" / "pointer.yaml")
    return dataclasses.replace(config, **overrides)


def dsp_output(gyro_z_dps: np.ndarray, user_accel: np.ndarray | None = None) -> DspOutput:
    """Watch lying flat with the given rotation rate about the vertical axis."""
    n = len(gyro_z_dps)
    gyro = np.zeros((n, 3))
    gyro[:, 2] = np.radians(gyro_z_dps)
    q = np.tile(quat.IDENTITY, (n, 1))
    return DspOutput(
        time_ns=np.arange(n, dtype=np.int64) * 10_000_000,
        valid=np.ones(n, dtype=bool),
        gyro_raw=gyro,
        gyro=gyro,
        gyro_bias=np.zeros((n, 3)),
        still=np.zeros(n, dtype=bool),
        gravity=np.tile([0.0, 0.0, -1.0], (n, 1)),
        user_accel=np.zeros((n, 3)) if user_accel is None else user_accel,
        device_q=q,
        madgwick_q=q,
    )


def test_disengaged_pointer_sends_nothing() -> None:
    out = PointerProcessor(pointer_config(start_engaged=False), FS).process(
        dsp_output(np.full(300, 50.0))
    )
    assert not out.engaged.any()
    assert out.dx.sum() == 0 and out.dy.sum() == 0


def test_total_cursor_travel_matches_the_gain_curve() -> None:
    config = pointer_config(start_engaged=True)
    rate = np.r_[np.full(100, 60.0), np.zeros(200)]  # turn left 60 deg/s for 1 s, then stop
    proc = PointerProcessor(config, FS)
    out = proc.process(dsp_output(rate))
    speed_px_s = proc.gain.apply(np.array([[60.0, 0.0]]))[0, 0]
    assert out.dx.sum() == pytest.approx(-speed_px_s * 1.0, abs=1)  # 1 s, leftwards
    assert out.dy.sum() == 0


def test_sub_pixel_motion_is_not_lost() -> None:
    config = pointer_config(start_engaged=True, dead_zone_dps=0.0)
    proc = PointerProcessor(config, FS)
    slow = 0.2  # deg/s: only a fraction of a pixel per frame
    out = proc.process(dsp_output(np.full(1000, -slow)))  # turning right, slowly
    per_frame_px = proc.gain.apply(np.array([[slow, 0.0]]))[0, 0] / FS
    assert per_frame_px < 0.1
    assert out.dx.sum() == pytest.approx(per_frame_px * 1000, abs=1.5)


def test_processor_output_does_not_depend_on_batching() -> None:
    rng = np.random.default_rng(3)
    rate = rng.normal(0, 40, 500)
    accel = rng.normal(0, 0.05, (500, 3))
    whole = PointerProcessor(pointer_config(start_engaged=True), FS).process(
        dsp_output(rate, accel)
    )
    proc = PointerProcessor(pointer_config(start_engaged=True), FS)
    full = dsp_output(rate, accel)
    parts = []
    for a, b in [(0, 1), (1, 120), (120, 333), (333, 500)]:
        part = DspOutput(**{f.name: getattr(full, f.name)[a:b] for f in dataclasses.fields(full)})
        parts.append(proc.process(part))
    np.testing.assert_array_equal(np.concatenate([p.dx for p in parts]), whole.dx)
    np.testing.assert_array_equal(np.concatenate([p.dy for p in parts]), whole.dy)


def test_double_knock_in_the_acceleration_engages_the_clutch() -> None:
    n = 400
    accel = np.zeros((n, 3))
    for t in (1.0, 1.25):  # two sharp spikes, like knuckles on a desk
        accel[round(t * FS), 2] = 2.0
    out = PointerProcessor(pointer_config(start_engaged=False), FS).process(
        dsp_output(np.full(n, 30.0), accel)
    )
    toggle_index = int(np.flatnonzero(out.toggled)[0])
    assert toggle_index / FS == pytest.approx(1.55, abs=0.02)
    assert not out.engaged[:toggle_index].any()
    assert out.engaged[toggle_index:].all()
    assert out.dx[toggle_index:].sum() < 0  # now turning left moves the cursor left
