"""Receive the live watch stream, record it, and run the ingest pipeline.

Usage (from the repository root, with the venv active):

    python scripts/ingest_live.py --note "desk test" [--subject S01 --wrist L]

Point Sensor Logger's HTTP push at http://<laptop-ip>:8000/data (see docs/SETUP_NETWORK.md).
Each run records a new session to data/raw/sessions/<YYYYmmdd-HHMMSS>/. Press Ctrl+C to stop;
the recording is closed cleanly and its metadata completed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from datetime import datetime
from pathlib import Path

from airhand.config import IngestConfig
from airhand.ingest.live import LiveIngest, push_urls
from airhand.ingest.recorder import SessionRecorder
from airhand.ingest.sensor_logger import WRIST_MOTION_UNITS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ingest.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--note", default="", help="What this recording is")
    parser.add_argument("--subject", default=None, help="Anonymous subject id, e.g. S01")
    parser.add_argument("--wrist", choices=["L", "R"], default=None, help="Wrist wearing the watch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = IngestConfig.from_yaml(args.config)
    session_dir = REPO_ROOT / config.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    info = {
        "note": args.note,
        "subject_id": args.subject,
        "wrist": args.wrist,
        "source_app": config.source_app,
        "sensor": config.sensor,
        "origin": "live",
    }

    if sys.platform == "win32":
        # Treat Ctrl+Break like Ctrl+C so the recording is also closed cleanly then.
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    recorder = SessionRecorder(
        session_dir,
        channels=config.record_channels,
        units=WRIST_MOTION_UNITS,
        info=info,
        flush_interval_s=config.flush_interval_s,
    )
    with recorder, asyncio.Runner() as runner:
        live = LiveIngest(config, recorder)
        runner.run(live.start())
        print(f"Recording to {recorder.session_dir}")
        for url in push_urls(config):
            print(f"  Sensor Logger push URL: {url}")
        print("Press Ctrl+C to stop.\n", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            runner.run(live.wait_forever())  # returns only when Ctrl+C interrupts it
        runner.run(live.stop())
        print("\n" + live.status())
    print(f"Saved {recorder.n_samples} samples in {recorder.n_packets} packets to {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
