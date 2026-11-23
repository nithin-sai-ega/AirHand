"""Convert a Phase 0 probe recording into a Phase 1 session folder.

Usage (from the repository root, with the venv active):

    python scripts/convert_probe_session.py data/raw/probe/<session> [more sessions...]

Reads packets.jsonl (verbatim HTTP bodies), extracts the watch samples exactly as the live
receiver would, and writes data/raw/sessions/probe-<session>/ with samples and metadata.
The original receive timestamps are kept, so the converted session replays with its real
packet timing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from airhand.config import IngestConfig
from airhand.ingest.raw_log import read_raw_log
from airhand.ingest.recorder import SessionRecorder
from airhand.ingest.sensor_logger import WRIST_MOTION_UNITS, parse_message, to_chunk

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ingest.yaml"


def convert(probe_dir: Path, config: IngestConfig) -> Path:
    probe_meta = json.loads((probe_dir / "session.json").read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / config.output_root / f"probe-{probe_dir.name}"
    info = {
        "note": probe_meta.get("note", ""),
        "subject_id": None,
        "wrist": None,
        "source_app": config.source_app,
        "sensor": config.sensor,
        "origin": f"converted from phase 0 probe session {probe_dir.name}",
    }
    with SessionRecorder(
        out_dir,
        channels=config.record_channels,
        units=WRIST_MOTION_UNITS,
        info=info,
        flush_interval_s=float("inf"),  # write one part at close; nothing is live here
    ) as recorder:
        user_agent = None
        for envelope in read_raw_log(probe_dir / "packets.jsonl"):
            if envelope.transport != "http" or envelope.method != "POST":
                continue
            user_agent = user_agent or envelope.headers.get("User-Agent")
            chunk = to_chunk(
                parse_message(envelope.body),
                sensor=config.sensor,
                channels=config.record_channels,
                packet_seq=envelope.seq,
                recv_wall_ns=envelope.recv_wall_ns,
                recv_mono_ns=envelope.recv_mono_ns,
            )
            if chunk is not None:
                recorder.append(chunk)
        recorder.update_info({"user_agent": user_agent})
    counts = f"{recorder.n_samples} samples in {recorder.n_packets} packets"
    print(f"{probe_dir.name}: {counts} -> {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probe_sessions", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = IngestConfig.from_yaml(args.config)
    for probe_dir in args.probe_sessions:
        convert(probe_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
