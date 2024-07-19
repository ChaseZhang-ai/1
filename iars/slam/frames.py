"""Minimal rigid-body math used across the IARS navigator.

Quaternions are stored as ``[w, x, y, z]`` and rotate body vectors into the
world frame (a point p_b satisfies p_w = R(q) @ p_b).  The world frame is a
local ENU frame: x east, y north, z up, gravity along -z.
"""

from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.80665])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that skew(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (axis * angle) -> unit quaternion."""
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        return quat_normalize(np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]]))
    axis = phi / theta
    return np.array([np.cos(0.5 * theta), *(np.sin(0.5 * theta) * axis)])


def quat_log(q: np.ndarray) -> np.ndarray:
    """Logarithmic map: unit quaternion -> rotation vector."""
    q = quat_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    if theta < 1e-9:
        return 2.0 * q[1:]
    return theta * q[1:] / s


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Stable conversion from a rotation matrix to a [w, x, y, z] quaternion."""
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3), truncated to second order."""
    S = skew(phi)
    return np.eye(3) + 0.5 * S + (1.0 / 6.0) * (S @ S)
