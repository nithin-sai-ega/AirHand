"""Streaming resampler: irregular sensor timestamps -> fixed-rate grid.

Downstream stages (filters, sliding windows, neural networks) assume samples are evenly
spaced. The watch delivers ~99.9 Hz on its own clock; this module produces values at exact
multiples of the grid period (10 ms for 100 Hz) of the sensor's time axis.

How each grid point g is computed:
- find the last sample at or before g (left) and the first sample after g (right);
- if a sample lands exactly on g, use it as-is;
- else if right - left <= max_gap, interpolate linearly between them;
- else the grid point is inside a gap: mark it invalid and store NaN. Long gaps are never
  filled in, because invented data would look real to every later stage.

Quaternions cannot be interpolated component-wise as plain numbers: q and -q are the same
rotation, and a straight line between two unit quaternions leaves the unit sphere. They use
normalised linear interpolation (nlerp): flip the sign of one end if needed so both are on
the same side, interpolate, then rescale to length 1. For the tiny rotations between two
samples 10 ms apart, nlerp is practically identical to the exact spherical method (slerp).

The resampler is causal and streaming: it only emits grid points up to the newest sample
it has seen (no extrapolation) and remembers the last sample between calls, so feeding
packets one at a time gives exactly the same output as feeding a whole recording at once.
"""

from __future__ import annotations

import numpy as np

from airhand.ingest.chunk import Frames, IntArray, SampleChunk

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


class Resampler:
    """Incremental resampler for one sensor stream.

    Args:
        input_channels: Channel order of incoming chunks.
        output_channels: Channels to resample, in output order (a subset of the input).
        quaternion_channels: The W, X, Y, Z channels (in that order) to interpolate as a
            unit quaternion. Empty if the stream has none.
        grid_hz: Output rate. The period is rounded to whole nanoseconds.
        max_gap_ms: Longest spacing between two samples that is still interpolated.
    """

    def __init__(
        self,
        *,
        input_channels: tuple[str, ...],
        output_channels: tuple[str, ...],
        quaternion_channels: tuple[str, ...],
        grid_hz: float,
        max_gap_ms: float,
    ) -> None:
        missing = [c for c in output_channels if c not in input_channels]
        if missing:
            raise ValueError(f"Output channels not in input: {missing}")
        if quaternion_channels and len(quaternion_channels) != 4:
            raise ValueError("quaternion_channels must be exactly W, X, Y, Z")
        not_output = [c for c in quaternion_channels if c not in output_channels]
        if not_output:
            raise ValueError(f"Quaternion channels must also be output channels: {not_output}")
        if grid_hz <= 0 or max_gap_ms <= 0:
            raise ValueError("grid_hz and max_gap_ms must be positive")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.period_ns = round(NS_PER_S / grid_hz)
        self.max_gap_ns = round(max_gap_ms * NS_PER_MS)
        self._select = [input_channels.index(c) for c in output_channels]
        self._quat_cols = [output_channels.index(c) for c in quaternion_channels]

        self._last_time: int | None = None
        self._last_values: np.ndarray | None = None
        self._next_grid_ns: int | None = None
        self.n_dropped = 0

    def push(self, chunk: SampleChunk) -> Frames:
        """Add one chunk of samples; return every grid frame that is now computable."""
        if chunk.channels != self.input_channels:
            raise ValueError("Chunk channels do not match the resampler's input channels")
        times, values = self._drop_non_increasing(chunk.time_ns, chunk.values[:, self._select])
        if len(times) == 0:
            return Frames.empty(self.output_channels)

        if self._last_time is None:
            # First grid point: the first multiple of the period at or after the first sample.
            self._next_grid_ns = -(-int(times[0]) // self.period_ns) * self.period_ns
        else:
            # Prepend the previous chunk's last sample so the first new grid points have a
            # left neighbour.
            times = np.concatenate(([self._last_time], times))
            values = np.vstack((self._last_values, values))

        assert self._next_grid_ns is not None
        grid = np.arange(self._next_grid_ns, int(times[-1]) + 1, self.period_ns, dtype=np.int64)
        frames = self._interpolate(times, values, grid)

        self._last_time = int(times[-1])
        self._last_values = values[-1].copy()
        if len(grid) > 0:
            self._next_grid_ns = int(grid[-1]) + self.period_ns
        return frames

    def _drop_non_increasing(
        self, times: IntArray, values: np.ndarray
    ) -> tuple[IntArray, np.ndarray]:
        """Keep only samples newer than every sample seen before (drops duplicates and
        out-of-order samples, which would make interpolation ambiguous)."""
        previous_max = np.maximum.accumulate(times)
        newest_before = np.concatenate(
            (
                [self._last_time if self._last_time is not None else np.iinfo(np.int64).min],
                previous_max[:-1],
            )
        )
        keep = times > newest_before
        self.n_dropped += int(np.count_nonzero(~keep))
        return times[keep], values[keep]

    def _interpolate(self, times: IntArray, values: np.ndarray, grid: IntArray) -> Frames:
        if len(grid) == 0:
            return Frames.empty(self.output_channels)

        left = np.searchsorted(times, grid, side="right") - 1  # times[left] <= grid
        right = np.minimum(left + 1, len(times) - 1)
        exact = times[left] == grid
        spacing = times[right] - times[left]

        # Fraction of the way from the left to the right sample (0 for exact hits).
        safe_spacing = np.where(spacing > 0, spacing, 1)
        alpha = ((grid - times[left]) / safe_spacing)[:, None]

        v_left, v_right = values[left], values[right]
        out = v_left + alpha * (v_right - v_left)
        if self._quat_cols:
            out[:, self._quat_cols] = _nlerp(
                v_left[:, self._quat_cols], v_right[:, self._quat_cols], alpha
            )
        out[exact] = v_left[exact]

        valid = exact | (spacing <= self.max_gap_ns)
        out[~valid] = np.nan
        return Frames(time_ns=grid, values=out, valid=valid, channels=self.output_channels)


def _nlerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Normalised linear interpolation between rows of unit quaternions."""
    same_side = np.sum(q0 * q1, axis=1, keepdims=True) >= 0
    q1 = np.where(same_side, q1, -q1)
    q = (1.0 - alpha) * q0 + alpha * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def resample_offline(chunks: list[SampleChunk], resampler: Resampler) -> Frames:
    """Resample a whole recording in one call (reference for the streaming path)."""
    if not chunks:
        return Frames.empty(resampler.output_channels)
    merged = SampleChunk(
        packet_seq=chunks[0].packet_seq,
        recv_wall_ns=chunks[-1].recv_wall_ns,
        recv_mono_ns=chunks[-1].recv_mono_ns,
        time_ns=np.concatenate([c.time_ns for c in chunks]),
        values=np.concatenate([c.values for c in chunks]),
        channels=chunks[0].channels,
    )
    return resampler.push(merged)
