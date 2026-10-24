"""Rigid-body helpers: quaternions, 4x4 poses and rotation angles.

Poses are camera-to-world 4x4 matrices T_wc (TUM convention), so the camera centre is
T[:3, 3]. Quaternions are (x, y, z, w) as in TUM trajectory files.
"""

from __future__ import annotations

import numpy as np


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion (any norm) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:  # noqa: N803
    """3x3 rotation -> (x, y, z, w) unit quaternion with w >= 0 (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i] = s / 4
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    return q if q[3] >= 0 else -q


def make_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:  # noqa: N803
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def invert(T: np.ndarray) -> np.ndarray:  # noqa: N803
    """Inverse of a rigid transform without a general matrix inverse."""
    R, t = T[:3, :3], T[:3, 3]
    return make_pose(R.T, -R.T @ t)


def relative(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:  # noqa: N803
    """Motion from pose a to pose b expressed in frame a: inv(Ta) @ Tb."""
    return invert(Ta) @ Tb


def rotation_angle(R: np.ndarray) -> float:  # noqa: N803
    """Geodesic angle of a rotation matrix in radians.

    atan2(sin, cos) instead of arccos(cos): arccos loses precision for small angles, which is
    exactly the regime of frame-to-frame errors.
    """
    R = R[:3, :3]
    sin = 0.5 * np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    cos = 0.5 * (np.trace(R) - 1.0)
    return float(np.arctan2(sin, cos))


def path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
