"""Loop closure and pose-graph optimisation on top of keyframe RGB-D odometry.

Batch (offline) version: run the odometry over the whole sequence, then

1. place recognition: a small bag of binary words, built by k-majority clustering of the
   keyframes' ORB descriptors, gives each keyframe a tf-idf vector; the most similar older
   keyframes (at least `min_gap_s` seconds earlier) are loop candidates;
2. geometric check: the candidate's depth-lifted points are matched to the current keyframe
   and PnP must find enough inliers; that relative pose becomes a loop edge;
3. pose graph: nodes are keyframe poses, edges are odometry links between consecutive
   keyframes plus the loop edges; scipy's least squares spreads the loop error over the
   loop. Loop edges that still disagree afterwards are dropped and the graph is re-solved;
4. every frame is re-attached to its corrected keyframe.

Nothing here looks at ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .config import Camera
from .estimation import pnp
from .features import match
from .geometry import invert, make_pose, rotation_angle
from .rgbd_vo import Keyframe, RgbdOdometry


@dataclass
class Edge:
    i: int
    j: int
    T_ij: np.ndarray  # measured pose of node j in node i's frame: T_j = T_i @ T_ij
    sigma_t: float  # metres
    sigma_r: float  # radians
    loop: bool = False
    inliers: int = 0


class Vocabulary:
    """Bag of binary words: k-majority centres of 256-bit ORB descriptors plus idf weights."""

    def __init__(self, descriptors: list[np.ndarray], k: int = 256, per_image: int = 300,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        sample = np.concatenate([d[rng.permutation(len(d))[:per_image]] for d in descriptors
                                 if len(d)])
        k = min(k, len(sample))
        bits = np.unpackbits(sample, axis=1).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
        cv2.setRNGSeed(seed)
        _, _, centres = cv2.kmeans(bits, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
        self.words = np.packbits(centres > 0.5, axis=1)  # majority bit of each cluster
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        df = np.zeros(len(self.words))
        for d in descriptors:
            df[np.unique(self.assign(d))] += 1
        self.idf = np.log(len(descriptors) / np.maximum(df, 1))

    def assign(self, desc: np.ndarray) -> np.ndarray:
        if len(desc) == 0:
            return np.empty(0, int)
        return np.array([m.trainIdx for m in self.matcher.match(desc, self.words)], int)

    def vector(self, desc: np.ndarray) -> np.ndarray:
        """L2-normalised tf-idf histogram of one image."""
        v = np.bincount(self.assign(desc), minlength=len(self.words)).astype(np.float64)
        v = v / max(v.sum(), 1) * self.idf
        return v / max(np.linalg.norm(v), 1e-12)


@dataclass
class SlamStats:
    keyframes: int = 0
    candidates: int = 0  # pairs that reached the geometric check
    loops_verified: int = 0  # passed PnP
    loops_kept: int = 0  # still consistent after optimisation
    loop_pairs: list[tuple[int, int]] = field(default_factory=list)  # frame indices
    loop_T: list[np.ndarray] = field(default_factory=list)  # measured T_ab of each kept loop


def odometry_edges(kfs: list[Keyframe], sigma_t: float, sigma_r: float) -> list[Edge]:
    edges = []
    for a in range(len(kfs) - 1):
        b = a + 1
        weak = kfs[b].lost  # pose came from a constant-velocity guess
        edges.append(Edge(a, b, invert(kfs[a].pose) @ kfs[b].pose,
                          1.0 if weak else sigma_t, np.radians(30) if weak else sigma_r))
    return edges


def detect_loops(kfs: list[Keyframe], stamps: np.ndarray, cam: Camera, min_gap_s: float = 10.0,
                 top_k: int = 3, min_score: float = 0.2, min_inliers: int = 60,
                 sigma_t: float = 0.02, sigma_r: float = np.radians(1.0),
                 stats: SlamStats | None = None) -> list[Edge]:
    """Appearance-based loop candidates, verified with PnP; at most one edge per keyframe."""
    stats = stats if stats is not None else SlamStats()
    if len(kfs) < 3:
        return []
    voc = Vocabulary([k.frame.desc for k in kfs])
    vecs = np.array([voc.vector(k.frame.desc) for k in kfs])
    t = stamps[[k.index for k in kfs]]
    edges = []
    for b in range(len(kfs)):
        older = np.flatnonzero(t <= t[b] - min_gap_s)
        if len(older) == 0:
            continue
        scores = vecs[older] @ vecs[b]
        best = None
        for a in older[np.argsort(-scores)[:top_k]]:
            if vecs[a] @ vecs[b] < min_score:
                break
            stats.candidates += 1
            est = verify(kfs[a], kfs[b], cam, min_inliers)
            if est is not None and (best is None or est[1] > best[2]):
                best = (a, *est)
        if best is not None:
            a, T_b_a, n = best
            edges.append(Edge(int(a), b, invert(T_b_a), sigma_t, sigma_r, loop=True, inliers=n))
    stats.loops_verified = len(edges)
    return edges


def verify(old: Keyframe, new: Keyframe, cam: Camera, min_inliers: int
           ) -> tuple[np.ndarray, int] | None:
    """T_new_old from the old keyframe's 3-D points seen in the new keyframe, or None."""
    m = match(old.frame.desc, new.frame.desc)
    if len(m) == 0:
        return None
    m = m[~np.isnan(old.pts3d[m[:, 0], 0])]
    est = pnp(old.pts3d[m[:, 0]], new.frame.pts[m[:, 1]], cam.K, 2.0, min_inliers=min_inliers)
    return None if est is None else (est.T, est.n_inliers)


def _unpack(x: np.ndarray, fixed: np.ndarray) -> tuple[Rotation, np.ndarray]:
    p = np.vstack([fixed, x.reshape(-1, 6)])
    return Rotation.from_rotvec(p[:, :3]), p[:, 3:]


def optimise(poses: np.ndarray, edges: list[Edge]) -> np.ndarray:
    """Least-squares pose graph: node 0 is held fixed; returns the corrected (N, 4, 4) poses.

    Residual per edge: rotation error log(R_ij^T R_i^T R_j) / sigma_r and translation error
    (R_i^T (t_j - t_i) - t_ij) / sigma_t, both 3-vectors.
    """
    n = len(poses)
    if n < 2 or not edges:
        return poses.copy()
    p0 = np.hstack([Rotation.from_matrix(poses[:, :3, :3]).as_rotvec(), poses[:, :3, 3]])
    ii = np.array([e.i for e in edges])
    jj = np.array([e.j for e in edges])
    Rz_inv = Rotation.from_matrix(np.array([e.T_ij[:3, :3] for e in edges])).inv()
    tz = np.array([e.T_ij[:3, 3] for e in edges])
    wr = 1 / np.array([e.sigma_r for e in edges])[:, None]
    wt = 1 / np.array([e.sigma_t for e in edges])[:, None]

    def residuals(x: np.ndarray) -> np.ndarray:
        R, t = _unpack(x, p0[:1])
        Ri_inv = R[ii].inv()
        r_rot = (Rz_inv * Ri_inv * R[jj]).as_rotvec() * wr
        r_t = (Ri_inv.apply(t[jj] - t[ii]) - tz) * wt
        return np.hstack([r_rot, r_t]).ravel()

    sparsity = lil_matrix((6 * len(edges), 6 * (n - 1)), dtype=int)
    for e, (i, j) in enumerate(zip(ii, jj, strict=True)):
        for node in (i, j):
            if node > 0:
                sparsity[6 * e:6 * e + 6, 6 * (node - 1):6 * node] = 1
    sol = least_squares(residuals, p0[1:].ravel(), jac_sparsity=sparsity, method="trf",
                        x_scale="jac", max_nfev=200)
    R, t = _unpack(sol.x, p0[:1])
    return np.array([make_pose(r, tt) for r, tt in zip(R.as_matrix(), t, strict=True)])


def edge_error(poses: np.ndarray, e: Edge) -> tuple[float, float]:
    """Translation (m) and rotation (deg) disagreement of an edge with the current poses."""
    E = invert(e.T_ij) @ invert(poses[e.i]) @ poses[e.j]
    return float(np.linalg.norm(E[:3, 3])), float(np.degrees(rotation_angle(E[:3, :3])))


def close_loops(vo: RgbdOdometry, stamps: np.ndarray, max_loop_err_m: float = 0.1,
                max_loop_err_deg: float = 5.0, rounds: int = 3, **loop_kw
                ) -> tuple[np.ndarray, SlamStats]:
    """Corrected poses for every frame tracked by `vo`, plus loop statistics."""
    kfs = vo.keyframe_list
    stats = SlamStats(keyframes=len(kfs))
    odo = odometry_edges(kfs, 0.02, np.radians(1.0))
    loops = detect_loops(kfs, stamps, vo.cam, stats=stats, **loop_kw)
    init = np.array([k.pose for k in kfs])
    for _ in range(rounds):  # re-solve without loop edges the solution can't agree with
        kf_poses = optimise(init, odo + loops)
        errs = [edge_error(kf_poses, e) for e in loops]
        keep = [et <= max_loop_err_m and er <= max_loop_err_deg for et, er in errs]
        if all(keep):
            break
        loops = [e for e, k in zip(loops, keep, strict=True) if k]
    else:
        kf_poses = optimise(init, odo + loops)
    stats.loops_kept = len(loops)
    stats.loop_pairs = [(kfs[e.i].index, kfs[e.j].index) for e in loops]
    stats.loop_T = [e.T_ij for e in loops]
    poses = np.array([kf_poses[k] @ rel for k, rel in vo.frame_refs])
    return poses, stats
