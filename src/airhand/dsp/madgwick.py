"""Madgwick orientation filter (IMU version: gyroscope + accelerometer, no magnetometer).

Idea in two lines:
1. Integrate the gyro: accurate over short times, but small errors add up (drift).
2. Each step, nudge the orientation so that the measured acceleration points where gravity
   should point. This removes tilt drift. Heading (rotation about the vertical) cannot be
   corrected without a magnetometer, so it still drifts slowly.

The nudge is one step of gradient descent on the error between
    "up, as predicted by the current orientation"   and   "up, as measured by the accelerometer",
with step size `beta` (rad/s). Reference: S. Madgwick, "An efficient orientation filter for
inertial and inertial/magnetic sensor arrays", 2010.

Conventions follow dsp/quaternion.py: q rotates body -> world, gyro in rad/s in the body
frame. The accelerometer input is the *specific force* a real accelerometer measures, which
points UP when the device is at rest (+1 g on z when lying flat). Core Motion reports
gravity pointing down, so the caller passes  -(gravity + user_acceleration).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from airhand.dsp import quaternion as quat

FloatArray = npt.NDArray[np.float64]


class Madgwick:
    def __init__(self, beta: float, initial: FloatArray | None = None) -> None:
        self.beta = beta
        self.q = quat.IDENTITY.copy() if initial is None else quat.normalize(initial.copy())

    def update(self, gyro: FloatArray, accel: FloatArray, dt: float) -> FloatArray:
        """Advance the orientation by one sample and return it."""
        # Rate of change of q from the gyro alone: q_dot = 0.5 * q * (0, omega).
        q_dot = 0.5 * quat.multiply(self.q, np.array([0.0, *gyro]))

        accel_norm = float(np.linalg.norm(accel))
        if accel_norm > 0.0:
            q_dot -= self.beta * self._correction_direction(accel / accel_norm)

        self.q = quat.normalize(self.q + q_dot * dt)
        return self.q

    def _correction_direction(self, a: FloatArray) -> FloatArray:
        """Unit gradient of the gravity alignment error with respect to q."""
        w, x, y, z = self.q
        # f: (world "up" expressed in the body frame) minus (measured up direction).
        f = np.array(
            [
                2.0 * (x * z - w * y) - a[0],
                2.0 * (w * x + y * z) - a[1],
                2.0 * (0.5 - x * x - y * y) - a[2],
            ]
        )
        # Jacobian of f with respect to (w, x, y, z).
        jacobian = np.array(
            [
                [-2.0 * y, 2.0 * z, -2.0 * w, 2.0 * x],
                [2.0 * x, 2.0 * w, 2.0 * z, 2.0 * y],
                [0.0, -4.0 * x, -4.0 * y, 0.0],
            ]
        )
        gradient = jacobian.T @ f
        norm = float(np.linalg.norm(gradient))
        return gradient / norm if norm > 0.0 else gradient
