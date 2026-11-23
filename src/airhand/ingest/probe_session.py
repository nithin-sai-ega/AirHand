"""Turn a recorded probe session into tables for analysis.

A probe session folder holds packets.jsonl (raw envelopes) and session.json (metadata).
This module keeps only Sensor Logger data packets (HTTP POSTs), parses them, and builds:

- a packet table: one row per received message, with receive times and message ids,
- a sample table per sensor: one row per sample, linked back to the packet it came in.

Linking every sample to its packet is what makes batching and latency measurable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from airhand.ingest.raw_log import read_raw_log
from airhand.ingest.sensor_logger import parse_message


@dataclass(frozen=True)
class ProbeSession:
    """Parsed contents of one probe session folder."""

    name: str
    metadata: dict[str, Any]
    packets: pd.DataFrame
    samples: pd.DataFrame

    @property
    def note(self) -> str:
        return str(self.metadata.get("note", ""))

    def sensor(self, sensor_name: str) -> pd.DataFrame:
        """Samples of one sensor, sorted by sensor time, with value fields as columns."""
        rows = self.samples[self.samples["sensor"] == sensor_name]
        return rows.dropna(axis="columns", how="all").sort_values("time_ns", kind="stable")


def load_probe_session(session_dir: Path) -> ProbeSession:
    """Load and parse a probe session folder.

    Only POST requests are treated as Sensor Logger data; other requests (e.g. the
    connectivity check GET /test from Safari) are ignored.
    """
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

    packet_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for envelope in read_raw_log(session_dir / "packets.jsonl"):
        if envelope.transport != "http" or envelope.method != "POST":
            continue
        message = parse_message(envelope.body)
        packet_rows.append(
            {
                "packet_seq": envelope.seq,
                "message_id": message.message_id,
                "session_id": message.session_id,
                "recv_wall_ns": envelope.recv_wall_ns,
                "recv_mono_ns": envelope.recv_mono_ns,
                "body_len": len(envelope.body),
                "n_samples": len(message.samples),
            }
        )
        for sample in message.samples:
            sample_rows.append(
                {
                    "packet_seq": envelope.seq,
                    "sensor": sample.name,
                    "time_ns": sample.time_ns,
                    **sample.values,
                }
            )

    packets = pd.DataFrame(
        packet_rows,
        columns=[
            "packet_seq",
            "message_id",
            "session_id",
            "recv_wall_ns",
            "recv_mono_ns",
            "body_len",
            "n_samples",
        ],
    )
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        samples = pd.DataFrame(columns=["packet_seq", "sensor", "time_ns"])
    return ProbeSession(name=session_dir.name, metadata=metadata, packets=packets, samples=samples)
