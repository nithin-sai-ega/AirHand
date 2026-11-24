"""Wrist rotation -> screen-direction angular velocity.

The gyro measures rotation about the watch's own axes, which depend on how the forearm is
rolled. To make "turn the hand left" always mean "cursor left", the rotation is measured
about two axes defined by gravity instead:

- up:   opposite to the gravity vector (the watch reports gravity in its own frame).
- left: horizontal and perpendicular to the forearm, = up x forearm.

    horizontal speed = rotation about up    (hand sweeps left/right like a flashlight)
    vertical speed   = rotation about left  (hand tilts up/down)

Rotating the forearm about its own length (wrist roll) has almost no component along either
axis, so it does not move the cursor.

Signs, using the right-hand rule: positive rotation about up turns the hand to the LEFT, so
screen x (positive = right) is minus that. Positive rotation about left tips the hand DOWN,
and screen y grows downwards, so screen y is plus that.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class WristToScreen:
    def __init__(self, forearm_axis: tuple[float, float, float], min_horizontal_norm: float):
        axis = np.asarray(forearm_axis, dtype=np.float64)
        self.forearm = axis / np.linalg.norm(axis)
        self.min_horizontal_norm = min_horizontal_norm

    def angular_velocity_dps(self, gyro: FloatArray, gravity: FloatArray) -> FloatArray:
        """Screen-oriented angular velocity (N, 2) in deg/s: [right+, down+].

        Args:
            gyro: (N, 3) body-frame rotation rate in rad/s.
            gravity: (N, 3) body-frame gravity (pointing down), any length.
        """
        up = -gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
        left = np.cross(up, self.forearm)
        left_norm = np.linalg.norm(left, axis=1, keepdims=True)
        # When the forearm points nearly straight up/down, "left" is undefined.
        usable = left_norm[:, 0] >= self.min_horizontal_norm
        left = np.where(usable[:, np.newaxis], left / np.maximum(left_norm, 1e-12), 0.0)

        rate_about_up = np.sum(gyro * up, axis=1)
        rate_about_left = np.sum(gyro * left, axis=1)
        return np.degrees(np.column_stack([-rate_about_up, rate_about_left]))
