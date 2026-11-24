"""Rolling history of pipeline signals for the live dashboard.

The pointer loop appends every processed batch; the dashboard (another thread) takes a
snapshot a few dozen times per second. Only the last `capacity` frames are kept.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from airhand.dsp import quaternion as quat
from airhand.dsp.processor import DspOutput
from airhand.pointer.processor import PointerOutput

FloatArray = npt.NDArray[np.float64]

# Columns of each history array, for the dashboard's legends.
SIGNALS: dict[str, tuple[str, ...]] = {
    "gyro_raw_dps": ("x", "y", "z"),
    "gyro_dps": ("x", "y", "z"),
    "knock_signal_g": ("knock",),
    "velocity_px_s": ("x", "y"),
    "position_raw_px": ("x", "y"),
    "position_smooth_px": ("x", "y"),
    "tilt_error_deg": ("tilt",),
    "heading_error_deg": ("heading",),
    "engaged": ("engaged",),
    "toggled": ("toggled",),
}


def orientation_errors(dsp: DspOutput) -> tuple[FloatArray, FloatArray]:
    """Madgwick vs watch orientation per frame: (tilt error, heading difference) in degrees."""
    tilt = np.full(len(dsp), np.nan)
    heading = np.full(len(dsp), np.nan)
    for i in range(len(dsp)):
        if dsp.valid[i] and np.isfinite(dsp.madgwick_q[i]).all():
            tilt[i] = quat.tilt_difference_deg(dsp.madgwick_q[i], dsp.device_q[i])
            heading[i] = quat.heading_difference_deg(dsp.madgwick_q[i], dsp.device_q[i])
    return tilt, heading


@dataclass
class TelemetrySnapshot:
    time_s: FloatArray
    signals: dict[str, FloatArray]
    status: dict[str, object] = field(default_factory=dict)


class TelemetryBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._time_ns = np.empty(0, dtype=np.int64)
        self._signals = {
            name: np.empty((0, len(cols))) for name, cols in SIGNALS.items()
        }
        self._status: dict[str, object] = {}
        self._lock = threading.Lock()

    def append(self, dsp: DspOutput, pointer: PointerOutput) -> None:
        tilt, heading = orientation_errors(dsp)
        new = {
            "gyro_raw_dps": np.degrees(dsp.gyro_raw),
            "gyro_dps": np.degrees(dsp.gyro),
            "knock_signal_g": pointer.knock_signal_g[:, np.newaxis],
            "velocity_px_s": pointer.velocity_px_s,
            "position_raw_px": pointer.position_raw,
            "position_smooth_px": pointer.position_smooth,
            "tilt_error_deg": tilt[:, np.newaxis],
            "heading_error_deg": heading[:, np.newaxis],
            "engaged": pointer.engaged.astype(np.float64)[:, np.newaxis],
            "toggled": pointer.toggled.astype(np.float64)[:, np.newaxis],
        }
        with self._lock:
            self._time_ns = np.concatenate([self._time_ns, dsp.time_ns])[-self.capacity :]
            for name, values in new.items():
                joined = np.concatenate([self._signals[name], values])
                self._signals[name] = joined[-self.capacity :]

    def set_status(self, **values: object) -> None:
        with self._lock:
            self._status.update(values)

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            time_ns = self._time_ns.copy()
            signals = {name: values.copy() for name, values in self._signals.items()}
            status = dict(self._status)
        # Time axis in seconds, with 0 = newest frame, so the plot scrolls.
        time_s = (time_ns - time_ns[-1]) / 1e9 if len(time_ns) else np.empty(0)
        return TelemetrySnapshot(time_s=time_s, signals=signals, status=status)
