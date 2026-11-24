"""Small quaternion helpers.

Conventions used everywhere in this project (they match Apple Core Motion, checked against
real watch data in Phase 2):

- A quaternion is a numpy array [w, x, y, z] (scalar first), Hamilton product.
- An orientation quaternion q rotates vectors from the watch's body frame into the
  reference (world) frame:  v_world = q * v_body * conj(q).
- Integrating a body-frame angular velocity omega over dt is  q_next = q * delta(omega*dt).

Only the handful of operations the DSP needs are implemented, each written out explicitly
so the math can be followed line by line.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def multiply(a: Vector, b: Vector) -> Vector:
    """Hamilton product a * b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def conjugate(q: Vector) -> Vector:
    """For a unit quaternion the conjugate is the inverse rotation."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(q: Vector) -> Vector:
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero quaternion")
    return q / norm


def rotate(q: Vector, v: Vector) -> Vector:
    """Rotate a 3-vector by unit quaternion q (body -> world for an orientation)."""
    p = np.array([0.0, v[0], v[1], v[2]])
    return multiply(multiply(q, p), conjugate(q))[1:]


def from_rotation_vector(rotation: Vector) -> Vector:
    """Quaternion for a rotation of |rotation| radians about the direction of `rotation`."""
    angle = float(np.linalg.norm(rotation))
    if angle == 0.0:
        return IDENTITY.copy()
    axis = rotation / angle
    half = angle / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def angle_between_deg(a: Vector, b: Vector) -> float:
    """Smallest rotation angle (degrees) taking orientation a to orientation b.

    q and -q are the same rotation, hence the absolute value of the dot product.
    """
    dot = min(1.0, abs(float(np.dot(normalize(a), normalize(b)))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def world_down_in_body(q: Vector) -> Vector:
    """Direction of gravity (world -z) expressed in the body frame: conj(q) * (0,0,-1) * q.

    For Core Motion data this equals the reported gravity vector, which makes it a direct
    way to compare a filter's tilt with the watch's own.
    """
    return rotate(conjugate(q), np.array([0.0, 0.0, -1.0]))


def tilt_difference_deg(a: Vector, b: Vector) -> float:
    """Angle (degrees) between the gravity directions that two orientations imply.

    This ignores heading, so it measures only how well "which way is down" agrees.
    """
    down_a = world_down_in_body(a)
    down_b = world_down_in_body(b)
    cos = float(np.clip(np.dot(down_a, down_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def heading_difference_deg(a: Vector, b: Vector) -> float:
    """Signed rotation (degrees) about the world vertical axis between orientations a and b.

    The world-frame difference q_err = a * conj(b) is split into a "twist" about world z and
    a remaining tilt (swing-twist decomposition); the twist angle is the heading difference.
    """
    q_err = multiply(normalize(a), conjugate(normalize(b)))
    w, z = q_err[0], q_err[3]
    angle = float(np.degrees(2.0 * np.arctan2(z, w)))
    return (angle + 180.0) % 360.0 - 180.0
