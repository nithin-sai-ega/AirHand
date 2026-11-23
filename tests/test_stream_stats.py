"""Stream statistics checked against synthetic streams with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from airhand.ingest.stream_stats import (
    NS_PER_MS,
    NS_PER_S,
    count_missing_message_ids,
    drift_ppm,
    fit_line,
    gap_stats,
    inter_arrival_ms,
    interval_stats,
    lower_envelope,
    sample_age_ms,
)

T0 = 1_795_463_355_000_000_000
PERIOD_NS = 10 * NS_PER_MS  # 100 Hz


def regular_stream(n: int, period_ns: int = PERIOD_NS) -> np.ndarray:
    return T0 + np.arange(n, dtype=np.int64) * period_ns


def test_regular_stream_rate() -> None:
    stats = interval_stats(regular_stream(1001))
    assert stats.n_samples == 1001
    assert stats.span_s == pytest.approx(10.0)
    assert stats.median_dt_ms == pytest.approx(10.0)
    assert stats.std_dt_ms == pytest.approx(0.0)
    assert stats.rate_from_median_hz == pytest.approx(100.0)
    assert stats.rate_from_count_hz == pytest.approx(100.0)
    assert stats.n_non_increasing == 0


def test_jitter_shows_up_in_std_but_not_in_rate() -> None:
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.5 * NS_PER_MS, size=10_001).astype(np.int64)
    stats = interval_stats(regular_stream(10_001) + jitter)
    assert stats.rate_from_count_hz == pytest.approx(100.0, rel=1e-3)
    # Difference of two independent N(0, 0.5 ms) values has std 0.5 * sqrt(2) ms.
    assert stats.std_dt_ms == pytest.approx(0.5 * np.sqrt(2), rel=0.05)


def test_duplicates_and_out_of_order_samples_are_counted() -> None:
    t = regular_stream(10)
    t[5] = t[4]  # duplicate
    t[8], t[7] = t[7], t[8]  # swapped
    assert interval_stats(t).n_non_increasing == 2


def test_interval_stats_needs_two_samples() -> None:
    with pytest.raises(ValueError):
        interval_stats(regular_stream(1))


def test_gaps_are_found_and_sized() -> None:
    t = regular_stream(1000)
    # Remove 9 samples (one gap of 10 periods) and 1 sample (one gap of 2 periods).
    t = np.delete(t, list(range(100, 109)) + [500])
    gaps = gap_stats(t, gap_factor=1.5)
    assert gaps.nominal_dt_ms == pytest.approx(10.0)
    assert gaps.n_gaps == 2
    assert gaps.estimated_missing_samples == 10
    assert gaps.total_gap_s == pytest.approx(0.10)  # 10 missing periods of 10 ms
    assert gaps.fraction_of_time_in_gaps == pytest.approx(0.10 / 9.99, rel=1e-3)


def test_no_gaps_in_regular_stream() -> None:
    gaps = gap_stats(regular_stream(500), gap_factor=1.5)
    assert gaps.n_gaps == 0
    assert gaps.estimated_missing_samples == 0


def test_missing_message_ids() -> None:
    assert count_missing_message_ids(np.array([10, 11, 12, 13])) == 0
    assert count_missing_message_ids(np.array([10, 11, 14, 15])) == 2
    assert count_missing_message_ids(np.array([], dtype=np.int64)) == 0


def test_inter_arrival() -> None:
    recv = np.array([0, 1000, 2100, 3000], dtype=np.int64) * NS_PER_MS
    np.testing.assert_allclose(inter_arrival_ms(recv), [1000, 1100, 900])


def test_sample_age() -> None:
    sample_time = np.array([T0, T0 + 500 * NS_PER_MS])
    recv = np.array([T0 + 1200 * NS_PER_MS, T0 + 1200 * NS_PER_MS])
    np.testing.assert_allclose(sample_age_ms(sample_time, recv), [1200, 700])


def test_lower_envelope_and_drift_recover_known_clock() -> None:
    """Offset grows by 50 ppm; random positive delay sits on top of a 30 ms floor."""
    rng = np.random.default_rng(1)
    t_s = np.arange(0, 600, 0.5)
    true_floor_ms = 30.0 + 0.05 * t_s  # 50 ppm = 0.05 ms per second
    observed = true_floor_ms + rng.exponential(scale=40.0, size=len(t_s))

    centers, minima = lower_envelope(t_s, observed, window_s=20.0)
    fit = fit_line(centers, minima)

    assert len(centers) == 30
    assert drift_ppm(fit) == pytest.approx(50.0, abs=5.0)
    assert fit.intercept == pytest.approx(30.0, abs=3.0)


def test_fit_line_needs_two_points() -> None:
    with pytest.raises(ValueError):
        fit_line(np.array([1.0]), np.array([1.0]))


def test_nanosecond_constants() -> None:
    assert NS_PER_S == 1000 * NS_PER_MS
