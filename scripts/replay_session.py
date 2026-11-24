"""Replay a recorded session through the ingest pipeline.

Usage (from the repository root, with the venv active):

    python scripts/replay_session.py data/raw/sessions/<session> [--speed 1.0]

--speed 0 (the default) replays as fast as possible; 1.0 reproduces the original packet
timing. Prints pipeline statistics and a SHA-256 digest of every resampled frame: replaying
the same session again must print the same digest.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from airhand.config import IngestConfig
from airhand.ingest.chunk import SampleChunk
from airhand.ingest.pipeline import IngestPipeline, build_pipeline
from airhand.ingest.recorder import read_session
from airhand.ingest.replayer import replay

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ingest.yaml"


async def run_replay(
    chunks: list[SampleChunk], pipeline: IngestPipeline, speed: float | None
) -> None:
    queue: asyncio.Queue[SampleChunk | None] = asyncio.Queue()
    await asyncio.gather(replay(chunks, queue, speed=speed), pipeline.run(queue))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="Session folder")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--speed", type=float, default=0.0, help="1.0 = original timing, 0 = as fast as possible"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = IngestConfig.from_yaml(args.config)
    session = read_session(args.session)
    if session.channels != config.record_channels:
        raise SystemExit("Session channels do not match configs/ingest.yaml record_channels")

    pipeline = build_pipeline(config)
    started = time.perf_counter()
    asyncio.run(run_replay(session.chunks, pipeline, args.speed or None))
    elapsed = time.perf_counter() - started

    s = pipeline.stats
    duration_s = s.frames / config.grid_hz
    print(f"session     {session.path.name}  ({session.meta.get('note', '')})")
    print(f"packets     {s.chunks}")
    print(f"samples     {s.samples} (dropped {pipeline.n_dropped_samples})")
    print(
        f"frames      {s.frames} = {duration_s:.1f} s at {config.grid_hz:g} Hz "
        f"(invalid {s.invalid_frames})"
    )
    print(f"replay time {elapsed:.2f} s")
    print(f"digest      {pipeline.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
