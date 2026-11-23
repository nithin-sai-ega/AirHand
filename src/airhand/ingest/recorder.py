"""Session recording to Parquet + JSON metadata, and reading sessions back.

A session folder contains:

    samples.parquet   one row per raw sample, in arrival order:
                      packet_seq, recv_wall_ns, recv_mono_ns, time_ns, <one column per channel>
    meta.json         schema version, channels and units, source, subject/notes, counts

Raw samples are recorded, not resampled frames: resampling is deterministic, so frames can
always be regenerated, and keeping the raw data lets us change the resampler later. Each
row also keeps its packet number and receive times, so a replay can reproduce the original
packet timing and latency can be analysed after the fact.

Parquet stores data column by column with compression, which suits this data well: reading
one channel for a plot touches only that column, and long runs of similar numbers compress.
Rows are flushed in groups every few seconds, so a crash loses at most the last few seconds.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from airhand.ingest.chunk import SampleChunk

SCHEMA_VERSION = 1
SAMPLES_FILE = "samples.parquet"
META_FILE = "meta.json"
PACKET_COLUMNS = ("packet_seq", "recv_wall_ns", "recv_mono_ns", "time_ns")


def _schema(channels: tuple[str, ...]) -> pa.Schema:
    return pa.schema(
        [pa.field(name, pa.int64()) for name in PACKET_COLUMNS]
        + [pa.field(name, pa.float64()) for name in channels]
    )


def _chunks_to_table(chunks: list[SampleChunk], schema: pa.Schema) -> pa.Table:
    """Flatten chunks into one table with the packet fields repeated on every row."""
    lengths = [len(c) for c in chunks]
    columns: dict[str, np.ndarray] = {
        "packet_seq": np.repeat([c.packet_seq for c in chunks], lengths).astype(np.int64),
        "recv_wall_ns": np.repeat([c.recv_wall_ns for c in chunks], lengths).astype(np.int64),
        "recv_mono_ns": np.repeat([c.recv_mono_ns for c in chunks], lengths).astype(np.int64),
        "time_ns": np.concatenate([c.time_ns for c in chunks]),
    }
    values = np.concatenate([c.values for c in chunks])
    for col, name in enumerate(chunks[0].channels):
        columns[name] = values[:, col]
    return pa.Table.from_pydict(columns, schema=schema)


class SessionRecorder:
    """Write raw sample chunks of one session to a new session folder.

    Args:
        session_dir: Folder to create; must not exist yet (never overwrite a recording).
        channels: Channel order of the chunks that will be appended.
        units: Unit of each channel, stored in the metadata.
        info: Free-form descriptive metadata (source app, sensor, subject id, note, ...).
        flush_interval_s: How often buffered rows are written to disk.
        clock: Monotonic clock in seconds; injectable so tests can control flushing.
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        channels: tuple[str, ...],
        units: dict[str, str],
        info: dict[str, Any],
        flush_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        session_dir.mkdir(parents=True, exist_ok=False)
        self.session_dir = session_dir
        self.channels = channels
        self._schema = _schema(channels)
        self._writer = pq.ParquetWriter(session_dir / SAMPLES_FILE, self._schema)
        self._pending: list[SampleChunk] = []
        self._flush_interval_s = flush_interval_s
        self._clock = clock
        self._last_flush = clock()

        self.n_packets = 0
        self.n_samples = 0
        self._first_time_ns: int | None = None
        self._last_time_ns: int | None = None
        self._meta: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "recording",
            "created_at": datetime.now().astimezone().isoformat(),
            "channels": list(channels),
            "units": {c: units[c] for c in channels},
            **info,
        }
        self._write_meta()

    def append(self, chunk: SampleChunk) -> None:
        if chunk.channels != self.channels:
            raise ValueError("Chunk channels do not match the recorder's channels")
        if len(chunk) == 0:
            return
        self._pending.append(chunk)
        self.n_packets += 1
        self.n_samples += len(chunk)
        if self._first_time_ns is None:
            self._first_time_ns = int(chunk.time_ns[0])
        self._last_time_ns = int(chunk.time_ns[-1])
        if self._clock() - self._last_flush >= self._flush_interval_s:
            self.flush()

    def flush(self) -> None:
        """Write buffered chunks as one Parquet row group."""
        if self._pending:
            self._writer.write_table(_chunks_to_table(self._pending, self._schema))
            self._pending.clear()
        self._last_flush = self._clock()

    def close(self) -> None:
        self.flush()
        self._writer.close()
        self._meta.update(
            {
                "status": "complete",
                "ended_at": datetime.now().astimezone().isoformat(),
                "n_packets": self.n_packets,
                "n_samples": self.n_samples,
                "first_time_ns": self._first_time_ns,
                "last_time_ns": self._last_time_ns,
            }
        )
        self._write_meta()

    def _write_meta(self) -> None:
        (self.session_dir / META_FILE).write_text(
            json.dumps(self._meta, indent=2), encoding="utf-8"
        )

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True)
class RecordedSession:
    """A session read back from disk, as the chunks (packets) it was recorded from."""

    path: Path
    meta: dict[str, Any]
    chunks: list[SampleChunk]

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self.meta["channels"])


def read_session(session_dir: Path) -> RecordedSession:
    """Read a session folder back into chunks, one per original packet, in arrival order."""
    meta = json.loads((session_dir / META_FILE).read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported session schema version: {meta.get('schema_version')}")
    channels = tuple(meta["channels"])
    table = pq.read_table(session_dir / SAMPLES_FILE)

    seq = table.column("packet_seq").to_numpy()
    if len(seq) == 0:
        return RecordedSession(path=session_dir, meta=meta, chunks=[])
    recv_wall = table.column("recv_wall_ns").to_numpy()
    recv_mono = table.column("recv_mono_ns").to_numpy()
    times = table.column("time_ns").to_numpy()
    values = np.column_stack([table.column(c).to_numpy() for c in channels])

    # Rows of one packet are contiguous; a new packet starts wherever packet_seq changes.
    starts = np.concatenate(([0], np.flatnonzero(np.diff(seq)) + 1))
    ends = np.concatenate((starts[1:], [len(seq)]))
    chunks = [
        SampleChunk(
            packet_seq=int(seq[a]),
            recv_wall_ns=int(recv_wall[a]),
            recv_mono_ns=int(recv_mono[a]),
            time_ns=times[a:b].astype(np.int64),
            values=values[a:b].astype(np.float64),
            channels=channels,
        )
        for a, b in zip(starts, ends, strict=True)
    ]
    return RecordedSession(path=session_dir, meta=meta, chunks=chunks)
