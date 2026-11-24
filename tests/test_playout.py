"""Tests for the playout (jitter) buffer, driven by a fake clock."""

from __future__ import annotations

import numpy as np

from airhand.ingest.chunk import Frames
from airhand.ingest.playout import PlayoutBuffer

MS = 1_000_000
PERIOD = 10 * MS
CHANNELS = ("a",)


def grid_frames(first_index: int, count: int) -> Frames:
    idx = np.arange(first_index, first_index + count, dtype=np.int64)
    return Frames(
        time_ns=idx * PERIOD,
        values=idx.astype(np.float64)[:, np.newaxis],
        valid=np.ones(count, dtype=bool),
        channels=CHANNELS,
    )


def run(
    arrivals: list[tuple[int, Frames]],
    end_ms: int,
    delay_s: float = 1.5,
    window_s: float = 30.0,
) -> tuple[PlayoutBuffer, list[tuple[int, int]]]:
    """Tick every 1 ms; return (buffer, [(release time ms, frame index)])."""
    buf = PlayoutBuffer(CHANNELS, PERIOD, delay_s=delay_s, offset_window_s=window_s)
    pending = sorted(arrivals, key=lambda a: a[0])
    released: list[tuple[int, int]] = []
    for now_ms in range(end_ms):
        while pending and pending[0][0] <= now_ms:
            buf.push(pending.pop(0)[1], now_ms * MS)
        out = buf.pop_due(now_ms * MS)
        released += [(now_ms, int(v)) for v in out.values[:, 0]]
    return buf, released


def packets_on_time(sizes: list[int], network_ms: int = 300) -> tuple[list, int]:
    """Packets sent as soon as their last sample exists, arriving `network_ms` later."""
    arrivals, first = [], 0
    for n in sizes:
        newest_ms = (first + n - 1) * 10
        arrivals.append((newest_ms + network_ms, grid_frames(first, n)))
        first += n
    return arrivals, first


def test_every_frame_plays_at_its_time_plus_floor_plus_delay() -> None:
    arrivals, total = packets_on_time([100] * 5)
    buf, released = run(arrivals, end_ms=8000)
    assert [i for _, i in released] == list(range(total))  # each frame once, in order
    for t_ms, i in released:
        assert t_ms == i * 10 + 300 + 1500  # sensor time + network floor + delay
    assert buf.n_underruns == 1  # only when the recording runs out


def test_variable_packet_sizes_like_the_real_stream_play_evenly() -> None:
    arrivals, total = packets_on_time([102, 153, 102, 153, 153, 102, 153])
    buf, released = run(arrivals, end_ms=12000, delay_s=1.6)
    times = np.array([t for t, _ in released])
    assert len(times) == total
    assert set(np.diff(times)) == {10}
    assert buf.n_underruns == 1  # end of recording


def test_jitter_within_the_margin_is_absorbed() -> None:
    arrivals, _ = packets_on_time([100] * 6)
    late_ms = [0, 200, 0, 350, 100, 0]  # extra network delay per packet
    arrivals = [(t + late, f) for (t, f), late in zip(arrivals, late_ms, strict=True)]
    # The oldest frame of a 1 s packet needs delay >= 1 s + 0.35 s of jitter.
    buf, released = run(arrivals, end_ms=10000, delay_s=1.5)
    times = np.array([t for t, _ in released])
    assert set(np.diff(times)) == {10}
    assert buf.n_underruns == 1


def test_late_packet_causes_one_underrun_then_catches_up() -> None:
    arrivals, total = packets_on_time([100] * 4)
    # Packet 2 is held up 1.5 s (far beyond the margin); packet 3 is queued behind it, as
    # packets are sent in order.
    stalled_until = arrivals[2][0] + 1500
    arrivals[2] = (stalled_until, arrivals[2][1])
    arrivals[3] = (max(arrivals[3][0], stalled_until), arrivals[3][1])
    buf, released = run(arrivals, end_ms=9000, delay_s=1.5)
    assert buf.n_underruns == 2  # the late packet, then the end of the recording
    assert [i for _, i in released] == list(range(total))
    # After catching up, frames are back on the normal schedule.
    play_time = {i: t for t, i in released}
    assert play_time[399] == 3990 + 300 + 1500


def test_slow_clock_drift_is_followed_without_underruns() -> None:
    # Arrival offset grows by 1 ms per second of data (Phase 0 measured about this much).
    arrivals, total = packets_on_time([100] * 300)  # 5 minutes
    arrivals = [(t + k, f) for k, (t, f) in enumerate(arrivals)]  # +1 ms per packet
    fixed_floor, _ = run(arrivals, end_ms=310_000, delay_s=1.02, window_s=10_000.0)
    sliding_floor, released = run(arrivals, end_ms=310_000, delay_s=1.02, window_s=10.0)
    # A floor fixed at the start drifts out of the 20 ms margin and keeps stalling; a
    # sliding floor follows the drift.
    assert fixed_floor.n_underruns > 10
    assert sliding_floor.n_underruns == 1  # end of recording
    assert len(released) == total


def test_nothing_is_released_before_any_data() -> None:
    buf = PlayoutBuffer(CHANNELS, PERIOD, delay_s=0.5, offset_window_s=10.0)
    assert len(buf.pop_due(10**9)) == 0
    assert buf.n_underruns == 0
