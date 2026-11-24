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
import socket
import sys
from datetime import datetime
from pathlib import Path

from aiohttp import web

from airhand.config import IngestConfig
from airhand.ingest.chunk import SampleChunk
from airhand.ingest.http_server import SensorLoggerReceiver
from airhand.ingest.pipeline import IngestPipeline, build_pipeline
from airhand.ingest.recorder import SessionRecorder
from airhand.ingest.sensor_logger import WRIST_MOTION_UNITS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ingest.yaml"


def lan_ipv4_addresses() -> list[str]:
    try:
        _, _, addresses = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        return []
    return sorted(a for a in addresses if not a.startswith("127."))


def status_line(pipeline: IngestPipeline, receiver: SensorLoggerReceiver) -> str:
    s = pipeline.stats
    age = f"{s.last_sample_age_ms:.0f} ms" if s.last_sample_age_ms is not None else "-"
    return (
        f"packets {s.chunks} | samples {s.samples} | frames {s.frames} "
        f"(invalid {s.invalid_frames}) | dropped {pipeline.n_dropped_samples} | "
        f"rejected {receiver.n_rejected} | no-watch {receiver.n_without_sensor} | "
        f"newest sample age {age} | buffer {len(pipeline.ring)}/{pipeline.ring.capacity}"
    )


class LiveIngest:
    """Live server + pipeline with an explicit start / stop.

    Shutdown is a separate step instead of a `finally` inside a long-running coroutine:
    when Ctrl+C interrupts the event loop, `stop()` is then run on the same loop, so the
    queue is always drained and the metadata completed, however the wait was interrupted.
    """

    def __init__(self, config: IngestConfig, recorder: SessionRecorder) -> None:
        self.config = config
        self.recorder = recorder
        self.queue: asyncio.Queue[SampleChunk | None] = asyncio.Queue(
            maxsize=config.queue_max_chunks
        )
        self.receiver = SensorLoggerReceiver(
            self.queue,
            sensor=config.sensor,
            channels=config.record_channels,
            data_path=config.data_path,
        )
        self.pipeline = build_pipeline(config, recorder)
        self._runner = web.AppRunner(self.receiver.make_app())
        self._pipeline_task: asyncio.Task[None] | None = None
        self._report_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._runner.setup()
        await web.TCPSite(self._runner, self.config.bind_host, self.config.http_port).start()
        self._pipeline_task = asyncio.create_task(self.pipeline.run(self.queue))
        self._report_task = asyncio.create_task(self._report())

        print(f"Recording to {self.recorder.session_dir}")
        for ip in lan_ipv4_addresses() or ["<laptop-ip>"]:
            url = f"http://{ip}:{self.config.http_port}{self.config.data_path}"
            print(f"  Sensor Logger push URL: {url}")
        print("Press Ctrl+C to stop.\n", flush=True)

    async def _report(self) -> None:
        while True:
            await asyncio.sleep(self.config.stats_interval_s)
            print(status_line(self.pipeline, self.receiver), flush=True)

    async def wait_forever(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        """Stop accepting packets, process everything already queued, complete metadata."""
        if self._report_task is not None:
            self._report_task.cancel()
        await self._runner.cleanup()
        await self.queue.put(None)
        if self._pipeline_task is not None:
            await self._pipeline_task
        self.recorder.update_info({"user_agent": self.receiver.user_agent})
        print("\n" + status_line(self.pipeline, self.receiver))


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
        with contextlib.suppress(KeyboardInterrupt):
            runner.run(live.wait_forever())  # returns only when Ctrl+C interrupts it
        runner.run(live.stop())
    print(f"Saved {recorder.n_samples} samples in {recorder.n_packets} packets to {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
