"""Trajectory error metrics: ATE after Umeyama alignment and RPE over fixed time windows.

Definitions follow Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems"
(IROS 2012), which the TUM benchmark tools implement:

* ATE: align the estimated camera centres to ground truth with a rigid (SE3) or similarity
  (Sim3, monocular) transform, then take the RMSE of the remaining position differences.
* RPE: for pairs (i, j) with t_j - t_i = delta, compare the estimated relative motion with the
  true one, E = inv(inv(Q_i) Q_j) inv(P_i) P_j; report translation RMSE and rotation RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import invert, rotation_angle


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False
            ) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares (s, R, t) with dst ~ s * R @ src + t for (N, 3) point sets (Umeyama 1991)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError(f"need two (N>=3, 3) arrays, got {src.shape} and {dst.shape}")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, d, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # reflection -> flip the weakest axis
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


@dataclass
class AteResult:
    rmse: float
    mean: float
    median: float
    max: float
    scale: float
    errors: np.ndarray  # per-pose position error after alignment (m)
    aligned: np.ndarray  # (N, 3) aligned estimated positions


def ate(gt: np.ndarray, est: np.ndarray, with_scale: bool = False) -> AteResult:
    """Absolute trajectory error of matched (N, 4, 4) poses; Sim3 alignment if with_scale."""
    p_gt, p_est = gt[:, :3, 3], est[:, :3, 3]
    s, R, t = umeyama(p_est, p_gt, with_scale)
    aligned = (s * (R @ p_est.T)).T + t
    err = np.linalg.norm(aligned - p_gt, axis=1)
    return AteResult(float(np.sqrt((err ** 2).mean())), float(err.mean()),
                     float(np.median(err)), float(err.max()), s, err, aligned)


@dataclass
class RpeResult:
    trans_rmse: float  # m per delta
    rot_rmse_deg: float  # degrees per delta
    pairs: int
    trans_errors: np.ndarray
    rot_errors_deg: np.ndarray


def rpe_pairs(timestamps: np.ndarray, delta: float, tol: float = 0.02
              ) -> list[tuple[int, int]]:
    """Index pairs (i, j) whose time difference is within `tol` seconds of `delta`.

    A loose tolerance silently mixes in shorter windows at the end of a sequence or across
    frame drops, which biases RPE low. 20 ms is below TUM's 33 ms frame period.
    """
    t = np.asarray(timestamps, np.float64)
    j = np.searchsorted(t, t + delta)
    pairs = []
    for i, jj in enumerate(j):
        cands = [c for c in (jj - 1, jj) if i < c < len(t)]
        if not cands:
            continue
        best = min(cands, key=lambda c: abs(t[c] - t[i] - delta))
        if abs(t[best] - t[i] - delta) < tol:
            pairs.append((i, best))
    return pairs


def rpe(gt: np.ndarray, est: np.ndarray, timestamps: np.ndarray, delta: float = 1.0,
        scale: float = 1.0) -> RpeResult:
    """Relative pose error over windows of `delta` seconds; `scale` rescales est translations."""
    est = est.copy()
    est[:, :3, 3] *= scale
    te, re = [], []
    for i, j in rpe_pairs(timestamps, delta):
        rel_gt = invert(gt[i]) @ gt[j]
        rel_est = invert(est[i]) @ est[j]
        E = invert(rel_gt) @ rel_est
        te.append(np.linalg.norm(E[:3, 3]))
        re.append(np.degrees(rotation_angle(E[:3, :3])))
    if not te:
        raise ValueError(f"no frame pairs {delta} s apart")
    te, re = np.array(te), np.array(re)
    return RpeResult(float(np.sqrt((te ** 2).mean())), float(np.sqrt((re ** 2).mean())),
                     len(te), te, re)
