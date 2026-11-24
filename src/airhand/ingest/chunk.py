"""Containers for samples moving through the ingest pipeline.

Two shapes of data flow through ingest:

- SampleChunk: raw samples of one sensor that arrived together in one packet, on the
  sensor's own (possibly irregular) timestamps. This is what sources (the HTTP server, the
  replayer) produce and what the recorder stores.
- Frames: samples on the fixed resampling grid, with a validity flag per frame. This is
  what the resampler produces and the ring buffer holds.

Both store times as int64 nanoseconds and values as a 2-D float64 array (rows = samples,
columns = channels), so whole packets are processed with numpy instead of per-sample loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class SampleChunk:
    """Raw samples from one packet.

    Attributes:
        packet_seq: Order in which the packet was received in its session.
        recv_wall_ns: Laptop wall-clock time when the packet arrived.
        recv_mono_ns: Laptop monotonic time when the packet arrived (used for replay timing).
        time_ns: Sensor timestamp of each sample, shape (N,).
        values: Sample values, shape (N, C), columns ordered as `channels`.
        channels: Channel names.
    """

    packet_seq: int
    recv_wall_ns: int
    recv_mono_ns: int
    time_ns: IntArray
    values: FloatArray
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.time_ns.ndim != 1:
            raise ValueError("time_ns must be 1-D")
        if self.values.shape != (len(self.time_ns), len(self.channels)):
            raise ValueError(
                f"values shape {self.values.shape} does not match "
                f"{len(self.time_ns)} samples x {len(self.channels)} channels"
            )

    def __len__(self) -> int:
        return len(self.time_ns)


@dataclass(frozen=True)
class Frames:
    """Samples on the fixed resampling grid.

    Attributes:
        time_ns: Grid times, shape (M,), evenly spaced.
        values: Values at those times, shape (M, C). NaN where `valid` is False.
        valid: False for grid points inside a gap that was too long to interpolate.
        channels: Channel names.
    """

    time_ns: IntArray
    values: FloatArray
    valid: BoolArray
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.time_ns)
        if self.values.shape != (n, len(self.channels)) or self.valid.shape != (n,):
            raise ValueError("Frames arrays have inconsistent shapes")

    def __len__(self) -> int:
        return len(self.time_ns)

    @classmethod
    def empty(cls, channels: tuple[str, ...]) -> Frames:
        return cls(
            time_ns=np.empty(0, dtype=np.int64),
            values=np.empty((0, len(channels)), dtype=np.float64),
            valid=np.empty(0, dtype=np.bool_),
            channels=channels,
        )

    @classmethod
    def concatenate(cls, parts: list[Frames], channels: tuple[str, ...]) -> Frames:
        """Join consecutive batches of frames into one."""
        if not parts:
            return cls.empty(channels)
        return cls(
            time_ns=np.concatenate([p.time_ns for p in parts]),
            values=np.concatenate([p.values for p in parts]),
            valid=np.concatenate([p.valid for p in parts]),
            channels=channels,
        )
