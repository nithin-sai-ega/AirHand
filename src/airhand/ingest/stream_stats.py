"""Timing statistics for a recorded sensor stream.

These functions answer the Phase 0 questions about the real stream: how fast samples
really arrive, how regular they are, whether any are missing, how they are batched into
packets, and how old data is by the time it reaches the laptop. They are pure numpy
functions so they can be tested against synthetic streams with known answers.

All times are integer nanoseconds unless a name says otherwise (_ms, _s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IntervalStats:
    """How regularly samples are spaced, from the sensor's own timestamps."""

    n_samples: int
    span_s: float
    median_dt_ms: float
    mean_dt_ms: float
    std_dt_ms: float
    rate_from_median_hz: float
    rate_from_count_hz: float
    n_non_increasing: int


def interval_stats(time_ns: IntArray) -> IntervalStats:
    """Sample spacing statistics.

    Two rate estimates are reported because they fail differently: the median spacing
    ignores gaps (it gives the rate the sensor runs at), while count/span includes them
    (it gives the rate actually delivered). If they disagree, samples are missing.

    `time_ns` must be in arrival order; non-increasing steps (duplicates or samples out of
    order) are counted rather than hidden, then the times are sorted for the statistics.
    """
    if len(time_ns) < 2:
        raise ValueError("Need at least two samples to measure intervals")
    raw_dt = np.diff(time_ns)
    n_non_increasing = int(np.count_nonzero(raw_dt <= 0))

    dt_ms = np.diff(np.sort(time_ns)) / NS_PER_MS
    span_s = float(time_ns.max() - time_ns.min()) / NS_PER_S
    median_dt_ms = float(np.median(dt_ms))
    return IntervalStats(
        n_samples=len(time_ns),
        span_s=span_s,
        median_dt_ms=median_dt_ms,
        mean_dt_ms=float(np.mean(dt_ms)),
        std_dt_ms=float(np.std(dt_ms)),
        rate_from_median_hz=1000.0 / median_dt_ms,
        rate_from_count_hz=(len(time_ns) - 1) / span_s,
        n_non_increasing=n_non_increasing,
    )


@dataclass(frozen=True)
class GapStats:
    """Stretches where samples are missing."""

    nominal_dt_ms: float
    threshold_ms: float
    n_gaps: int
    total_gap_s: float
    fraction_of_time_in_gaps: float
    estimated_missing_samples: int


def gap_stats(time_ns: IntArray, gap_factor: float) -> GapStats:
    """Find gaps: spacings larger than gap_factor x the nominal (median) spacing.

    The time "in gaps" counts only the extra time beyond one nominal period, and the
    missing-sample estimate is how many nominal periods fit into each gap minus one.
    """
    sorted_ns = np.sort(time_ns)
    dt_ms = np.diff(sorted_ns) / NS_PER_MS
    nominal_dt_ms = float(np.median(dt_ms))
    threshold_ms = gap_factor * nominal_dt_ms
    gap_dt_ms = dt_ms[dt_ms > threshold_ms]

    span_s = float(sorted_ns[-1] - sorted_ns[0]) / NS_PER_S
    extra_s = float(np.sum(gap_dt_ms - nominal_dt_ms)) / 1000.0
    missing = int(np.sum(np.round(gap_dt_ms / nominal_dt_ms) - 1))
    return GapStats(
        nominal_dt_ms=nominal_dt_ms,
        threshold_ms=threshold_ms,
        n_gaps=len(gap_dt_ms),
        total_gap_s=extra_s,
        fraction_of_time_in_gaps=extra_s / span_s if span_s > 0 else 0.0,
        estimated_missing_samples=missing,
    )


def count_missing_message_ids(message_ids: IntArray) -> int:
    """Messages lost in transit, assuming ids increase by exactly 1 per message."""
    ids = np.unique(message_ids)
    if len(ids) == 0:
        return 0
    expected = int(ids[-1] - ids[0] + 1)
    return expected - len(ids)


def inter_arrival_ms(recv_mono_ns: IntArray) -> FloatArray:
    """Time between consecutive packet arrivals on the laptop (monotonic clock)."""
    return np.diff(np.sort(recv_mono_ns)) / NS_PER_MS


def sample_age_ms(sample_time_ns: IntArray, recv_wall_ns: IntArray) -> FloatArray:
    """How old each sample is when its packet arrives: receive time minus sample time.

    This compares two different clocks (sender and laptop), so it equals
    true delay + clock offset. Differences between streams in the same session are
    unaffected by the offset; absolute values are only as good as the clock sync.
    """
    return (recv_wall_ns - sample_time_ns) / NS_PER_MS


@dataclass(frozen=True)
class LineFit:
    """y = intercept + slope * x."""

    slope: float
    intercept: float


def lower_envelope(
    t_s: FloatArray, y: FloatArray, window_s: float
) -> tuple[FloatArray, FloatArray]:
    """Minimum of y in consecutive windows of length window_s.

    Network delay varies but never drops below some floor, so the per-window minimum of
    (receive time - sample time) tracks clock offset + minimum delay while ignoring the
    random extra delay of individual packets.
    """
    if len(t_s) == 0:
        return np.array([]), np.array([])
    start = float(np.min(t_s))
    window_index = np.floor((t_s - start) / window_s).astype(np.int64)
    centers: list[float] = []
    minima: list[float] = []
    for index in np.unique(window_index):
        in_window = window_index == index
        best = int(np.argmin(np.where(in_window, y, np.inf)))
        centers.append(float(t_s[best]))
        minima.append(float(y[best]))
    return np.array(centers), np.array(minima)


def fit_line(x: FloatArray, y: FloatArray) -> LineFit:
    """Least-squares straight line through (x, y)."""
    if len(x) < 2:
        raise ValueError("Need at least two points to fit a line")
    slope, intercept = np.polyfit(x, y, deg=1)
    return LineFit(slope=float(slope), intercept=float(intercept))


def drift_ppm(fit_ms_per_s: LineFit) -> float:
    """Clock drift in parts per million from a fit of offset (ms) against time (s).

    1 ms of offset change per second is 1000 ppm, so ppm = slope * 1000.
    """
    return fit_ms_per_s.slope * 1000.0
