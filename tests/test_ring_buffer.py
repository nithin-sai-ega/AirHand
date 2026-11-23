"""Tests for the frame ring buffer, including wraparound."""

from __future__ import annotations

import numpy as np
import pytest

from airhand.ingest.chunk import Frames
from airhand.ingest.ring_buffer import RingBuffer

CHANNELS = ("a", "b")


def frames(start: int, n: int) -> Frames:
    """n frames whose time and values encode their index, so order is easy to check."""
    idx = np.arange(start, start + n)
    return Frames(
        time_ns=idx.astype(np.int64),
        values=np.column_stack([idx, -idx]).astype(np.float64),
        valid=(idx % 3 != 0),
        channels=CHANNELS,
    )


def test_empty_buffer() -> None:
    ring = RingBuffer(5, CHANNELS)
    assert len(ring) == 0
    assert len(ring.latest(3)) == 0


def test_latest_before_full() -> None:
    ring = RingBuffer(5, CHANNELS)
    ring.extend(frames(0, 3))
    out = ring.latest(10)
    np.testing.assert_array_equal(out.time_ns, [0, 1, 2])
    np.testing.assert_array_equal(out.values[:, 1], [0, -1, -2])
    np.testing.assert_array_equal(out.valid, [False, True, True])


def test_wraparound_keeps_newest_in_order() -> None:
    ring = RingBuffer(5, CHANNELS)
    for start in range(0, 12, 3):  # four batches of 3 -> frames 0..11
        ring.extend(frames(start, 3))
    assert len(ring) == 5
    assert ring.total_written == 12
    np.testing.assert_array_equal(ring.latest(5).time_ns, [7, 8, 9, 10, 11])
    np.testing.assert_array_equal(ring.latest(2).time_ns, [10, 11])


def test_batch_larger_than_capacity() -> None:
    ring = RingBuffer(4, CHANNELS)
    ring.extend(frames(0, 2))
    ring.extend(frames(2, 10))
    np.testing.assert_array_equal(ring.latest(4).time_ns, [8, 9, 10, 11])
    ring.extend(frames(12, 1))
    np.testing.assert_array_equal(ring.latest(4).time_ns, [9, 10, 11, 12])


def test_latest_returns_a_copy() -> None:
    ring = RingBuffer(3, CHANNELS)
    ring.extend(frames(0, 3))
    out = ring.latest(3)
    out.values[:] = 99
    assert ring.latest(3).values[0, 0] == 0


def test_matches_a_plain_list_over_many_random_batches() -> None:
    """Reference check: the buffer must always equal the tail of everything written."""
    rng = np.random.default_rng(0)
    ring = RingBuffer(37, CHANNELS)
    written: list[int] = []
    start = 0
    for _ in range(200):
        n = int(rng.integers(0, 60))
        ring.extend(frames(start, n))
        written.extend(range(start, start + n))
        start += n
        k = int(rng.integers(0, 40))
        expected = written[-k:] if k else []
        np.testing.assert_array_equal(ring.latest(k).time_ns, expected[-ring.capacity :])


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        RingBuffer(0, CHANNELS)
    ring = RingBuffer(3, CHANNELS)
    other = Frames.empty(("x",))
    with pytest.raises(ValueError):
        ring.extend(other)
