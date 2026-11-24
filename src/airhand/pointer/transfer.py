"""Transfer function (control-display gain): wrist angular speed -> cursor speed.

A constant gain forces a bad trade-off: high gain crosses the screen quickly but makes small
targets hard to hit; low gain is precise but needs big arm movements. Pointer acceleration
(as in every desktop OS) uses a gain that grows with speed:

    gain(s) = low + (high - low) * sigmoid((s - mid) / width)      [pixels per degree]

Slow, careful motion gets ~`low` px/deg; fast flicks get ~`high` px/deg; the sigmoid makes
the switch smooth so the cursor never jumps when speed crosses a threshold.

A small dead zone first subtracts `dead_zone` deg/s from the speed (not a hard cut-off, which
would make the cursor jump into motion), so tremor and sensor noise at rest are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GainCurve:
    dead_zone_dps: float
    low_px_per_deg: float
    high_px_per_deg: float
    mid_dps: float
    width_dps: float

    def gain(self, speed_dps: FloatArray) -> FloatArray:
        """Pixels per degree at the given (dead-zone-adjusted) speed."""
        sigmoid = 1.0 / (1.0 + np.exp(-(speed_dps - self.mid_dps) / self.width_dps))
        return self.low_px_per_deg + (self.high_px_per_deg - self.low_px_per_deg) * sigmoid

    def apply(self, angular_velocity_dps: FloatArray) -> FloatArray:
        """(N, 2) angular velocity in deg/s -> (N, 2) cursor velocity in px/s."""
        speed = np.linalg.norm(angular_velocity_dps, axis=1, keepdims=True)
        effective = np.maximum(speed - self.dead_zone_dps, 0.0)
        direction = np.divide(
            angular_velocity_dps, speed, out=np.zeros_like(angular_velocity_dps), where=speed > 0
        )
        return direction * effective * self.gain(effective)
