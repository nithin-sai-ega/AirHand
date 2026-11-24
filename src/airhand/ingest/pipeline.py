"""The ingest pipeline: chunks in -> recorded, resampled, buffered.

    source (HTTP server or replayer) -> queue -> IngestPipeline.process(chunk)
                                                  |-> recorder.append(chunk)   (raw, optional)
                                                  |-> resampler.push(chunk) -> frames
                                                  '-> ring_buffer.extend(frames)

`process` is a plain synchronous function, which keeps it deterministic and easy to test;
`run` just pulls chunks from the queue and calls it until the source sends None.

The pipeline also keeps a SHA-256 digest of every frame it produces. Two runs over the same
input must give the same digest, which turns "replay is deterministic" into a one-line check.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from airhand.config import IngestConfig
from airhand.ingest.chunk import Frames, SampleChunk
from airhand.ingest.recorder import SessionRecorder
from airhand.ingest.resampler import Resampler
from airhand.ingest.ring_buffer import RingBuffer
from airhand.latency import StageTimer

NS_PER_MS = 1_000_000


@dataclass
class PipelineStats:
    chunks: int = 0
    samples: int = 0
    frames: int = 0
    invalid_frames: int = 0
    # Laptop receive time minus the newest sample time of the latest chunk (includes the
    # clock offset between phone and laptop, see Phase 0 results).
    last_sample_age_ms: float | None = None


class IngestPipeline:
    def __init__(
        self,
        resampler: Resampler,
        ring: RingBuffer,
        recorder: SessionRecorder | None = None,
        on_frames: Callable[[Frames], None] | None = None,
        timer: StageTimer | None = None,
    ) -> None:
        """
        Args:
            on_frames: Optional callback given each batch of new frames (e.g. to feed the
                pointer's playout buffer).
            timer: Optional latency recorder: "ingest" processing time per packet and
                "sample_age" (receive time minus newest sample time) per packet.
        """
        if ring.channels != resampler.output_channels:
            raise ValueError("Ring buffer channels must match the resampler output channels")
        self.resampler = resampler
        self.ring = ring
        self.recorder = recorder
        self.on_frames = on_frames
        self.timer = timer
        self.stats = PipelineStats()
        self._digest = hashlib.sha256()

    def process(self, chunk: SampleChunk) -> Frames:
        """Record, resample and buffer one chunk; return the new frames."""
        if self.timer is None:
            return self._process(chunk)
        with self.timer.measure("ingest"):
            frames = self._process(chunk)
        if self.stats.last_sample_age_ms is not None:
            self.timer.record("sample_age", self.stats.last_sample_age_ms)
        return frames

    def _process(self, chunk: SampleChunk) -> Frames:
        if self.recorder is not None:
            self.recorder.append(chunk)
        frames = self.resampler.push(chunk)
        self.ring.extend(frames)

        self.stats.chunks += 1
        self.stats.samples += len(chunk)
        self.stats.frames += len(frames)
        self.stats.invalid_frames += int(np.count_nonzero(~frames.valid))
        if len(chunk) > 0:
            self.stats.last_sample_age_ms = (
                chunk.recv_wall_ns - int(chunk.time_ns[-1])
            ) / NS_PER_MS
        self._digest.update(frames.time_ns.tobytes())
        self._digest.update(frames.values.tobytes())
        self._digest.update(frames.valid.tobytes())
        if self.on_frames is not None and len(frames) > 0:
            self.on_frames(frames)
        return frames

    async def run(self, queue: asyncio.Queue[SampleChunk | None]) -> None:
        """Process chunks from the queue until a None end marker arrives."""
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            self.process(chunk)

    @property
    def n_dropped_samples(self) -> int:
        return self.resampler.n_dropped

    def digest(self) -> str:
        """Hex SHA-256 of all frames produced so far."""
        return self._digest.hexdigest()


def build_pipeline(
    config: IngestConfig,
    recorder: SessionRecorder | None = None,
    on_frames: Callable[[Frames], None] | None = None,
    timer: StageTimer | None = None,
) -> IngestPipeline:
    """Create resampler, ring buffer and pipeline from the ingest config."""
    resampler = Resampler(
        input_channels=config.record_channels,
        output_channels=config.resample_channels,
        quaternion_channels=config.quaternion_channels,
        grid_hz=config.grid_hz,
        max_gap_ms=config.max_interp_gap_ms,
    )
    ring = RingBuffer(config.ring_capacity_frames, config.resample_channels)
    return IngestPipeline(resampler, ring, recorder, on_frames, timer)
