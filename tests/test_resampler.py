"""Numerical tests for the streaming resampler against signals with known values."""

from __future__ import annotations

import numpy as np
import pytest

from airhand.ingest.chunk import Frames, SampleChunk
from airhand.ingest.resampler import Resampler, resample_offline

MS = 1_000_000
T0 = 1_795_463_355_000_000_000  # a multiple of 10 ms, like real epoch nanoseconds
QUAT = ("qw", "qx", "qy", "qz")


def make_resampler(
    channels: tuple[str, ...] = ("a", "b"),
    quaternion: tuple[str, ...] = (),
    max_gap_ms: float = 50.0,
) -> Resampler:
    return Resampler(
        input_channels=channels,
        output_channels=channels,
        quaternion_channels=quaternion,
        grid_hz=100.0,
        max_gap_ms=max_gap_ms,
    )


def chunk(
    times: np.ndarray, values: np.ndarray, channels: tuple[str, ...] = ("a", "b")
) -> SampleChunk:
    return SampleChunk(
        packet_seq=0,
        recv_wall_ns=0,
        recv_mono_ns=0,
        time_ns=np.asarray(times, dtype=np.int64),
        values=np.asarray(values, dtype=np.float64),
        channels=channels,
    )


def split(c: SampleChunk, sizes: list[int]) -> list[SampleChunk]:
    """Cut one chunk into consecutive chunks of the given sizes (like separate packets)."""
    edges = np.cumsum([0, *sizes])
    return [
        chunk(c.time_ns[a:b], c.values[a:b], c.channels)
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


def test_samples_already_on_the_grid_pass_through_unchanged() -> None:
    times = T0 + np.arange(50) * 10 * MS
    values = np.column_stack([np.arange(50.0), -np.arange(50.0)])
    frames = make_resampler().push(chunk(times, values))
    np.testing.assert_array_equal(frames.time_ns, times)
    np.testing.assert_array_equal(frames.values, values)
    assert frames.valid.all()


def test_linear_signal_is_reproduced_exactly_from_99_9_hz() -> None:
    """Linear interpolation is exact for a straight line, whatever the sample times."""
    times = T0 + 3 * MS + np.round(np.arange(1000) * 10.01 * MS).astype(np.int64)
    seconds = (times - T0) / 1e9
    values = np.column_stack([2.0 * seconds + 1.0, -0.5 * seconds])
    frames = make_resampler().push(chunk(times, values))

    grid_seconds = (frames.time_ns - T0) / 1e9
    np.testing.assert_allclose(frames.values[:, 0], 2.0 * grid_seconds + 1.0, atol=1e-12)
    np.testing.assert_allclose(frames.values[:, 1], -0.5 * grid_seconds, atol=1e-12)
    assert np.all(frames.time_ns % (10 * MS) == 0), "grid points must be multiples of 10 ms"
    assert frames.time_ns[0] >= times[0] and frames.time_ns[-1] <= times[-1], "no extrapolation"


def test_jittered_sine_error_is_within_the_interpolation_bound() -> None:
    """2 Hz sine sampled at 99.9 Hz with +/-0.5 ms timestamp jitter (true times known).

    Linear interpolation error is at most h^2/8 * max|f''|, with h the largest spacing.
    """
    rng = np.random.default_rng(0)
    nominal = np.arange(5000) * 10.01 * MS
    times = (T0 + nominal + rng.uniform(-0.5, 0.5, size=5000) * MS).astype(np.int64)
    freq = 2.0

    def signal(t_ns: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * freq * (t_ns - T0) / 1e9)

    values = np.column_stack([signal(times), np.zeros(len(times))])
    frames = make_resampler().push(chunk(times, values))

    h = np.diff(times).max() / 1e9
    bound = h**2 / 8 * (2 * np.pi * freq) ** 2
    error = np.abs(frames.values[:, 0] - signal(frames.time_ns))
    assert frames.valid.all()
    assert error.max() <= bound
    assert error.max() > 0  # sanity: the test signal really needs interpolation


def test_long_gap_is_marked_invalid_and_short_gap_is_interpolated() -> None:
    times = T0 + np.arange(100) * 10 * MS
    values = np.column_stack([np.arange(100.0), np.zeros(100)])
    # Long gap: samples 20-29 missing (110 ms between neighbours > 50 ms).
    # Short gap: samples 60-61 missing (30 ms between neighbours <= 50 ms).
    keep = np.ones(100, dtype=bool)
    keep[20:30] = False
    keep[60:62] = False
    frames = make_resampler(max_gap_ms=50.0).push(chunk(times[keep], values[keep]))

    np.testing.assert_array_equal(frames.time_ns, times)  # the grid itself has no holes
    assert not frames.valid[20:30].any()
    assert np.isnan(frames.values[20:30]).all()
    assert frames.valid[60:62].all()
    np.testing.assert_allclose(frames.values[60:62, 0], [60.0, 61.0])  # linear data
    assert frames.valid.sum() == 90


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_streaming_in_packets_equals_offline_resampling(seed: int) -> None:
    """Packet boundaries must not change the output (determinism of the streaming path)."""
    rng = np.random.default_rng(seed)
    n = 3000
    times = (T0 + 7 * MS + np.arange(n) * 10.01 * MS + rng.uniform(-1, 1, n) * MS).astype(np.int64)
    values = rng.normal(size=(n, 2))
    gap = slice(1000, 1015)  # a long gap somewhere in the middle
    keep = np.ones(n, dtype=bool)
    keep[gap] = False
    whole = chunk(times[keep], values[keep])

    sizes = []
    remaining = len(whole)
    while remaining > 0:
        size = int(min(remaining, rng.integers(1, 160)))
        sizes.append(size)
        remaining -= size

    streaming = make_resampler()
    parts = [streaming.push(c) for c in split(whole, sizes)]
    streamed = Frames.concatenate(parts, ("a", "b"))
    offline = resample_offline([whole], make_resampler())

    np.testing.assert_array_equal(streamed.time_ns, offline.time_ns)
    np.testing.assert_array_equal(streamed.valid, offline.valid)
    np.testing.assert_array_equal(streamed.values, offline.values)  # NaNs compare equal here


def test_duplicate_and_out_of_order_samples_are_dropped_and_counted() -> None:
    times = T0 + np.array([0, 10, 10, 20, 15, 30, 40]) * MS
    values = np.column_stack([np.arange(7.0), np.zeros(7)])
    resampler = make_resampler()
    frames = resampler.push(chunk(times, values))
    assert resampler.n_dropped == 2
    np.testing.assert_array_equal(frames.time_ns, T0 + np.array([0, 10, 20, 30, 40]) * MS)
    np.testing.assert_array_equal(frames.values[:, 0], [0, 1, 3, 5, 6])

    # A later packet that goes back in time is dropped too.
    frames = resampler.push(chunk(T0 + np.array([35, 50]) * MS, np.ones((2, 2))))
    assert resampler.n_dropped == 3
    np.testing.assert_array_equal(frames.time_ns, [T0 + 50 * MS])


def test_quaternion_is_interpolated_on_the_unit_sphere_across_a_sign_flip() -> None:
    """q and -q are the same rotation; a sign flip between samples must not distort it."""
    angle = np.deg2rad(np.array([0.0, 2.0]))  # small rotation about z between two samples
    q = np.column_stack([np.cos(angle / 2), np.zeros(2), np.zeros(2), np.sin(angle / 2)])
    q[1] = -q[1]  # second sample stored with the opposite sign
    times = T0 + 5 * MS + np.array([0, 10]) * MS  # grid point T0 + 10 ms falls in between
    frames = make_resampler(QUAT, QUAT).push(chunk(times, q, QUAT))

    assert len(frames) == 1
    result = frames.values[0]
    assert np.linalg.norm(result) == pytest.approx(1.0)
    expected = np.array([np.cos(np.deg2rad(1.0) / 2), 0, 0, np.sin(np.deg2rad(1.0) / 2)])
    # Halfway between 0 and 2 degrees is 1 degree (up to the overall sign).
    assert abs(np.dot(result, expected)) == pytest.approx(1.0, abs=1e-9)


def test_output_channels_can_be_a_reordered_subset() -> None:
    resampler = Resampler(
        input_channels=("a", "b", "c"),
        output_channels=("c", "a"),
        quaternion_channels=(),
        grid_hz=100.0,
        max_gap_ms=50.0,
    )
    times = T0 + np.arange(3) * 10 * MS
    values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    frames = resampler.push(chunk(times, values, ("a", "b", "c")))
    assert frames.channels == ("c", "a")
    np.testing.assert_array_equal(frames.values, [[3, 1], [6, 4], [9, 7]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_channels": ("zzz",)},
        {"quaternion_channels": ("a", "b")},
        {"grid_hz": 0.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    settings: dict[str, object] = {
        "input_channels": ("a", "b"),
        "output_channels": ("a", "b"),
        "quaternion_channels": (),
        "grid_hz": 100.0,
        "max_gap_ms": 50.0,
    }
    settings.update(kwargs)
    with pytest.raises(ValueError):
        Resampler(**settings)  # type: ignore[arg-type]


def test_chunk_with_other_channels_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_resampler().push(chunk(np.array([T0]), np.zeros((1, 1)), ("x",)))
