"""Recorder/reader round trip and replay timing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from airhand.ingest.chunk import SampleChunk
from airhand.ingest.recorder import (
    META_FILE,
    PART_PATTERN,
    SAMPLES_DIR,
    SessionRecorder,
    read_session,
)
from airhand.ingest.replayer import replay

CHANNELS = ("a", "b", "c")
UNITS = {"a": "rad/s", "b": "g", "c": "1"}
T0 = 1_795_463_355_000_000_000
MS = 1_000_000


def make_chunks(n_packets: int, seed: int = 0) -> list[SampleChunk]:
    """Packets of varying size, ~1 s apart, with random values."""
    rng = np.random.default_rng(seed)
    chunks = []
    t = T0
    for seq in range(n_packets):
        n = int(rng.integers(1, 160))
        times = t + np.arange(n, dtype=np.int64) * 10 * MS
        t = int(times[-1]) + 10 * MS
        chunks.append(
            SampleChunk(
                packet_seq=seq,
                recv_wall_ns=t + 800 * MS,
                recv_mono_ns=seq * 1_050 * MS,
                time_ns=times,
                values=rng.normal(size=(n, len(CHANNELS))),
                channels=CHANNELS,
            )
        )
    return chunks


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def record(path: Path, chunks: list[SampleChunk], **kwargs: object) -> None:
    settings: dict[str, object] = {
        "channels": CHANNELS,
        "units": UNITS,
        "info": {"note": "unit test", "subject_id": None},
        "flush_interval_s": 5.0,
    }
    settings.update(kwargs)
    with SessionRecorder(path, **settings) as recorder:  # type: ignore[arg-type]
        for chunk in chunks:
            recorder.append(chunk)


def assert_chunks_equal(a: list[SampleChunk], b: list[SampleChunk]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert (x.packet_seq, x.recv_wall_ns, x.recv_mono_ns, x.channels) == (
            y.packet_seq,
            y.recv_wall_ns,
            y.recv_mono_ns,
            y.channels,
        )
        np.testing.assert_array_equal(x.time_ns, y.time_ns)
        np.testing.assert_array_equal(x.values, y.values)


def test_round_trip_is_exact(tmp_path: Path) -> None:
    chunks = make_chunks(25)
    record(tmp_path / "s", chunks)
    session = read_session(tmp_path / "s")
    assert session.channels == CHANNELS
    assert_chunks_equal(session.chunks, chunks)


def test_metadata_is_complete(tmp_path: Path) -> None:
    chunks = make_chunks(5)
    record(tmp_path / "s", chunks)
    meta = json.loads((tmp_path / "s" / META_FILE).read_text(encoding="utf-8"))
    assert meta["status"] == "complete"
    assert meta["schema_version"] == 1
    assert meta["channels"] == list(CHANNELS)
    assert meta["units"] == UNITS
    assert meta["note"] == "unit test"
    assert meta["n_packets"] == 5
    assert meta["n_samples"] == sum(len(c) for c in chunks)
    assert meta["first_time_ns"] == int(chunks[0].time_ns[0])
    assert meta["last_time_ns"] == int(chunks[-1].time_ns[-1])


def test_flushes_part_files_on_the_interval(tmp_path: Path) -> None:
    clock = FakeClock()
    chunks = make_chunks(6)
    with SessionRecorder(
        tmp_path / "s", channels=CHANNELS, units=UNITS, info={}, flush_interval_s=2.0, clock=clock
    ) as recorder:
        for chunk in chunks:
            recorder.append(chunk)
            clock.now += 1.0  # one packet per second
    # Flushes run when packets 2 and 4 arrive (t = 2 s, 4 s) and close() writes packet 5: 3 parts.
    parts = sorted((tmp_path / "s" / SAMPLES_DIR).glob("*.parquet"))
    assert [p.name for p in parts] == [PART_PATTERN.format(i) for i in range(3)]
    assert_chunks_equal(read_session(tmp_path / "s").chunks, chunks)


def test_flushed_data_survives_a_crash(tmp_path: Path) -> None:
    """If the process dies without close(), everything flushed so far is still readable."""
    clock = FakeClock()
    chunks = make_chunks(6)
    recorder = SessionRecorder(
        tmp_path / "s", channels=CHANNELS, units=UNITS, info={}, flush_interval_s=2.0, clock=clock
    )
    for chunk in chunks:
        recorder.append(chunk)
        clock.now += 1.0
    # No close(): flushes ran when packets 2 and 4 arrived (t = 2 s, 4 s), so packets 0-4 are
    # on disk and packet 5 was still in memory when the process "died".
    session = read_session(tmp_path / "s")
    assert session.meta["status"] == "recording"
    assert_chunks_equal(session.chunks, chunks[:5])


def test_update_info_adds_metadata(tmp_path: Path) -> None:
    with SessionRecorder(
        tmp_path / "s", channels=CHANNELS, units=UNITS, info={"note": "x"}, flush_interval_s=1.0
    ) as recorder:
        recorder.update_info({"user_agent": "SensorLogger/14"})
    meta = read_session(tmp_path / "s").meta
    assert meta["user_agent"] == "SensorLogger/14" and meta["note"] == "x"


def test_empty_session_is_valid(tmp_path: Path) -> None:
    record(tmp_path / "s", [])
    session = read_session(tmp_path / "s")
    assert session.chunks == []
    assert session.meta["n_samples"] == 0


def test_never_overwrites_an_existing_session(tmp_path: Path) -> None:
    record(tmp_path / "s", make_chunks(1))
    with pytest.raises(FileExistsError):
        record(tmp_path / "s", make_chunks(1))


def test_rejects_chunks_with_other_channels(tmp_path: Path) -> None:
    chunk = make_chunks(1)[0]
    other = SampleChunk(
        packet_seq=0,
        recv_wall_ns=0,
        recv_mono_ns=0,
        time_ns=chunk.time_ns,
        values=chunk.values[:, :2],
        channels=("a", "b"),
    )
    with (
        pytest.raises(ValueError),
        SessionRecorder(
            tmp_path / "s", channels=CHANNELS, units=UNITS, info={}, flush_interval_s=1.0
        ) as recorder,
    ):
        recorder.append(other)


def drain(queue: asyncio.Queue[SampleChunk | None]) -> list[SampleChunk]:
    items = []
    while True:
        item = queue.get_nowait()
        if item is None:
            return items
        items.append(item)


def test_fast_replay_puts_every_chunk_then_end_marker() -> None:
    chunks = make_chunks(10)

    async def main() -> list[SampleChunk]:
        queue: asyncio.Queue[SampleChunk | None] = asyncio.Queue()
        await replay(chunks, queue, speed=None)
        return drain(queue)

    assert_chunks_equal(asyncio.run(main()), chunks)


def test_real_time_replay_follows_recorded_timing() -> None:
    """Chunks recorded 1.05 s apart are released 1.05 s apart (0.525 s at 2x speed)."""
    chunks = make_chunks(5)

    for speed, expected_gap in [(1.0, 1.05), (2.0, 0.525)]:
        clock = FakeClock()
        sleeps: list[float] = []

        async def fake_sleep(
            seconds: float, clock: FakeClock = clock, log: list[float] = sleeps
        ) -> None:
            log.append(seconds)
            clock.now += seconds

        async def main(speed: float = speed, clock: FakeClock = clock) -> None:
            queue: asyncio.Queue[SampleChunk | None] = asyncio.Queue()
            await replay(chunks, queue, speed=speed, sleep=fake_sleep, clock=clock)

        asyncio.run(main())
        np.testing.assert_allclose(sleeps, [expected_gap] * 4)


def test_replay_rejects_bad_speed() -> None:
    async def main() -> None:
        await replay([], asyncio.Queue(), speed=0.0)

    with pytest.raises(ValueError):
        asyncio.run(main())
