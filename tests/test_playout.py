"""Tests for the playout buffer, driven by a fake clock."""

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
    arrivals: list[tuple[int, Frames]], end_ms: int, delay_s: float = 0.8
) -> tuple[PlayoutBuffer, list[tuple[int, int]]]:
    """Tick every 1 ms; return (buffer, [(release time ms, frame index)])."""
    buf = PlayoutBuffer(CHANNELS, PERIOD, delay_s=delay_s, max_excess_s=1.0)
    pending = sorted(arrivals, key=lambda a: a[0])
    released: list[tuple[int, int]] = []
    for now_ms in range(end_ms):
        while pending and pending[0][0] == now_ms:
            buf.push(pending.pop(0)[1], now_ms * MS)
        out = buf.pop_due(now_ms * MS)
        released += [(now_ms, int(v)) for v in out.values[:, 0]]
    return buf, released


def test_one_second_bursts_come_out_evenly_spaced() -> None:
    # Five packets of 100 frames, arriving once per second.
    arrivals = [(1000 * k, grid_frames(100 * k, 100)) for k in range(5)]
    buf, released = run(arrivals, end_ms=6000)

    assert [i for _, i in released] == list(range(500))  # each frame once, in order
    times = np.array([t for t, _ in released])
    assert times[0] == 800  # first frame plays `delay` after it arrived
    assert set(np.diff(times)) == {10}  # then exactly one frame per 10 ms
    # The only underrun is at the very end, when the recording has run out.
    assert buf.n_underruns == 1
    assert buf.n_skips == 0


def test_jitter_smaller_than_the_delay_is_absorbed() -> None:
    late_ms = [0, 0, 500, 0, 300]  # packet k arrives up to 0.5 s late
    arrivals = [(1000 * k + late_ms[k], grid_frames(100 * k, 100)) for k in range(5)]
    buf, released = run(arrivals, end_ms=6000)
    times = np.array([t for t, _ in released])
    assert set(np.diff(times)) == {10}
    assert buf.n_underruns == 1  # only when the recording runs out


def test_packet_later_than_the_delay_causes_one_underrun_then_recovers() -> None:
    arrivals = [(0, grid_frames(0, 100)), (2500, grid_frames(100, 100))]  # 1.5 s late
    buf, released = run(arrivals, end_ms=5000)
    assert buf.n_underruns == 2  # the late packet, then the end of the recording
    assert [i for _, i in released] == list(range(200))
    second_start = next(t for t, i in released if i == 100)
    assert second_start == 2500 + 800  # restarts `delay` after the late packet arrived
    later = np.array([t for t, i in released if i >= 100])
    assert set(np.diff(later)) == {10}


def test_backlog_after_a_stall_is_skipped_to_keep_latency_bounded() -> None:
    # Packet 0 on time; packets 1-4 held up by a 4 s stall and delivered together at 5 s;
    # packet 5 on time again at 6 s.
    arrivals = [(0, grid_frames(0, 100))]
    arrivals += [(5000, grid_frames(100 * k, 100)) for k in range(1, 5)]
    arrivals += [(6000, grid_frames(500, 100))]
    buf, released = run(arrivals, end_ms=8000)
    assert buf.n_skips >= 1
    indices = [i for _, i in released]
    assert indices == sorted(indices)
    assert len(indices) < 600  # part of the backlog was dropped
    # Without skipping, packet 5 would wait behind 4 s of backlog. With it, no frame is ever
    # played more than delay + max_excess after it arrived.
    assert next(t for t, i in released if i == 500) <= 6000 + 800 + 1000


def test_one_large_packet_is_played_not_skipped() -> None:
    arrivals = [(0, grid_frames(0, 100)), (1000, grid_frames(100, 500))]  # 5 s in one packet
    buf, released = run(arrivals, end_ms=8000)
    assert buf.n_skips == 0
    assert [i for _, i in released] == list(range(600))


def test_packets_longer_than_one_second_do_not_cause_skips() -> None:
    # Like the real stream: 1.02 or 1.53 s of data per packet, one packet every 1-1.5 s.
    sizes = [102, 153, 102, 153, 153, 102, 153]
    arrivals, first, now = [], 0, 0
    for n in sizes:
        arrivals.append((now, grid_frames(first, n)))
        first += n
        now += n * 10  # the phone sends once it has collected the packet's samples
    buf, released = run(arrivals, end_ms=now + 3000)
    assert buf.n_skips == 0
    assert [i for _, i in released] == list(range(first))
    times = np.array([t for t, _ in released])
    assert set(np.diff(times)) == {10}


def test_nothing_is_released_before_any_data() -> None:
    buf = PlayoutBuffer(CHANNELS, PERIOD, delay_s=0.5, max_excess_s=1.0)
    assert len(buf.pop_due(10**9)) == 0
    assert buf.n_underruns == 0
