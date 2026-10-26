"""Monocular visual odometry: two-view initialisation, then PnP against triangulated points.

One camera gives no metric scale, and frame-to-frame essential matrices give only a unit
translation, so the scale would be re-guessed every frame. Instead (the 3-D-to-2-D scheme of
Scaramuzza & Fraundorfer's VO tutorial):

1. Initialise: wait until a frame has enough parallax to a reference frame, estimate the
   essential matrix, fix the scale once (median depth of the triangulated points), keep the
   points with good parallax and reprojection error as a small map.
2. Track: every frame is located by PnP against the current keyframe's map points, so all
   frames share that one scale.
3. When the inliers thin out, the current frame becomes a keyframe: inlier points carry over,
   new points are triangulated against the recent keyframe with the longest baseline. With the
   previous keyframe only (a few cm away) most points sit near the parallax threshold, the
   filter keeps those whose noise makes them look closer, and the map shrinks at every hop
   (scale fell from 0.83 to 0.01 within 175 frames on fr1_xyz). For the same reason no
   keyframe is made while the baseline is below `min_baseline` x the scene depth (the camera
   reversing direction), the old one is kept instead. Scale still drifts slowly,
   because every new point inherits the error of the poses it was made from.
4. A frame where PnP fails (often motion blur) is skipped and the next frame is tried
   against the same keyframe. After `max_lost` failures in a row tracking is lost: initialise
   again from the last tracked frame, with the scale chosen so the new map has the old map's
   median depth.

Frames seen before (re)initialisation are located by PnP afterwards; frames that never get a
pose keep the previous pose and are counted as untracked.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import Camera
from .estimation import essential, parallax_deg, pnp, reprojection_error, triangulate
from .features import Frame, OrbExtractor, match
from .geometry import invert


@dataclass
class MonoStats:
    frames: int = 0
    inits: int = 0  # successful (re)initialisations, the first included
    failed: int = 0  # frames where PnP against the keyframe failed
    losses: int = 0  # times tracking was lost and the map rebuilt
    keyframes: int = 0
    untracked: int = 0  # frames that never got their own pose


@dataclass
class _Keyframe:
    frame: Frame
    pose: np.ndarray  # T_wc
    points: np.ndarray  # (N, 3) world points per keypoint, NaN where none
    first_inliers: int = 0

    def depth_median(self) -> float:
        ok = ~np.isnan(self.points[:, 0])
        Xc = (invert(self.pose)[:3, :3] @ self.points[ok].T).T + invert(self.pose)[:3, 3]
        return float(np.median(Xc[:, 2]))


class MonoOdometry:
    """Online monocular odometry. Call `track(gray)` per frame, then `trajectory()`."""

    def __init__(self, cam: Camera, init_min_px: float = 20.0, init_min_points: int = 100,
                 min_parallax_deg: float = 1.0, max_reproj_px: float = 2.0,
                 kf_min_inliers: int = 120, kf_min_ratio: float = 0.5, max_lost: int = 5,
                 min_inliers: int = 15, window: int = 5, min_baseline: float = 0.05,
                 extractor: OrbExtractor | None = None):
        self.cam = cam
        self.init_min_px = init_min_px
        self.init_min_points = init_min_points
        self.min_parallax = min_parallax_deg
        self.max_reproj = max_reproj_px
        self.kf_min_inliers = kf_min_inliers
        self.kf_min_ratio = kf_min_ratio
        self.max_lost = max_lost
        self.min_inliers = min_inliers
        self.min_baseline = min_baseline
        self.lost_run = 0
        self.extract = extractor or OrbExtractor(cam)
        self.stats = MonoStats()
        self.poses: list[np.ndarray | None] = []
        self.kf: _Keyframe | None = None
        self.recent: deque[_Keyframe] = deque(maxlen=window)  # candidates for triangulation
        self.ref: tuple[int, Frame, np.ndarray] | None = None  # init reference: idx, frame, pose
        self.pending: list[tuple[int, Frame]] = []  # frames waiting for a map
        self.last: tuple[int, Frame] | None = None  # last tracked frame
        self.depth_hint = 1.0  # the first map gets median depth 1 (arbitrary unit)

    # ---- public -------------------------------------------------------------------------
    def track(self, gray: np.ndarray) -> np.ndarray | None:
        k = len(self.poses)
        frame = self.extract(gray)
        self.poses.append(None)
        self.stats.frames += 1
        if self.kf is None:
            self._initialise(k, frame)
        else:
            self._track(k, frame)
        return self.poses[k]

    def trajectory(self) -> np.ndarray:
        """(N, 4, 4) poses; untracked frames copy the nearest earlier (else later) pose."""
        known = [i for i, p in enumerate(self.poses) if p is not None]
        if not known:
            raise RuntimeError("monocular VO never initialised")
        out, self.stats.untracked = [], 0
        for i, p in enumerate(self.poses):
            if p is None:
                self.stats.untracked += 1
                prev = [j for j in known if j < i]
                p = self.poses[prev[-1] if prev else known[0]]
            out.append(p)
        return np.array(out)

    # ---- initialisation -----------------------------------------------------------------
    def _initialise(self, k: int, frame: Frame) -> None:
        if self.ref is None:
            pose = np.eye(4) if self.last is None else self.poses[self.last[0]]
            src = (k, frame) if self.last is None else self.last
            self.ref = (src[0], src[1], pose)
            if self.last is None:
                self.poses[k] = pose
                return
        ref_idx, ref, ref_pose = self.ref
        self.pending.append((k, frame))
        m = match(ref.desc, frame.desc)
        if len(m) < self.init_min_points:
            if len(m) < self.init_min_points // 2:  # the view has changed too much: restart
                self.ref = (k, frame, ref_pose)
            return
        pa, pb = ref.pts[m[:, 0]], frame.pts[m[:, 1]]
        if np.median(np.linalg.norm(pa - pb, axis=1)) < self.init_min_px:
            return
        est = essential(pa, pb, self.cam.K)
        if est is None:
            return
        pose = ref_pose @ invert(est.T)  # unit baseline for now
        X, good = self._triangulate(ref_pose, pose, pa[est.inliers], pb[est.inliers])
        if good.sum() < self.init_min_points:
            return
        # Fix the scale: the new map's median depth (in the reference camera) = depth_hint
        Xc = (invert(ref_pose)[:3, :3] @ X[good].T).T + invert(ref_pose)[:3, 3]
        s = self.depth_hint / np.median(Xc[:, 2])
        pose = ref_pose @ _scaled(invert(est.T), s)
        X = ref_pose[:3, 3] + s * (X - ref_pose[:3, 3])
        points = np.full((len(frame), 3), np.nan)
        points[m[est.inliers[good], 1]] = X[good]
        self.kf = _Keyframe(frame, pose, points)
        self.recent.clear()
        self.recent.append(self.kf)
        self.stats.inits += 1
        self.stats.keyframes += 1
        self.poses[k] = pose
        if self.poses[ref_idx] is None:
            self.poses[ref_idx] = ref_pose
        for j, f in self.pending:  # locate the frames seen while waiting for parallax
            if self.poses[j] is None:
                self.poses[j] = self._locate(f)
        self.pending, self.ref, self.last = [], None, (k, frame)

    # ---- tracking -----------------------------------------------------------------------
    def _track(self, k: int, frame: Frame) -> None:
        kf = self.kf
        m = match(kf.frame.desc, frame.desc)
        has_pt = ~np.isnan(kf.points[m[:, 0], 0]) if len(m) else np.zeros(0, bool)
        mp = m[has_pt]
        est = pnp(kf.points[mp[:, 0]], frame.pts[mp[:, 1]], self.cam.K, self.max_reproj,
                  self.min_inliers) if len(mp) else None
        if est is None:
            self.stats.failed += 1
            self.lost_run += 1
            if self.lost_run < self.max_lost:
                return
            self.stats.losses += 1
            self.lost_run = 0
            self.depth_hint = kf.depth_median()
            self.kf, self.ref = None, None
            self._initialise(k, frame)
            return
        pose = invert(est.T)
        self.poses[k] = pose
        self.last = (k, frame)
        self.lost_run = 0
        n = est.n_inliers
        if kf.first_inliers == 0:
            kf.first_inliers = n
        if n < self.kf_min_inliers or n < self.kf_min_ratio * kf.first_inliers:
            self._new_keyframe(frame, pose, m, mp[est.inliers])

    def _new_keyframe(self, frame: Frame, pose: np.ndarray, m: np.ndarray,
                      inl: np.ndarray) -> None:
        kf = self.kf
        points = np.full((len(frame), 3), np.nan)
        points[inl[:, 1]] = kf.points[inl[:, 0]]  # carried over: keeps the scale
        # Triangulate against the recent keyframe furthest away that still shares the view
        mo, old = None, None
        for cand in sorted(self.recent,
                           key=lambda o: -np.linalg.norm(o.pose[:3, 3] - pose[:3, 3])):
            base = np.linalg.norm(cand.pose[:3, 3] - pose[:3, 3])
            if base < self.min_baseline * cand.depth_median():
                break  # sorted by baseline: the rest are even closer
            mc = m if cand is kf else match(cand.frame.desc, frame.desc)
            if len(mc) >= self.kf_min_inliers:
                mo, old = mc, cand
                break
        if old is None:
            return  # baseline too short to triangulate without bias: keep the old keyframe
        fresh = mo[np.isnan(points[mo[:, 1], 0])]
        X, good = self._triangulate(old.pose, pose, old.frame.pts[fresh[:, 0]],
                                    frame.pts[fresh[:, 1]])
        points[fresh[good, 1]] = X[good]
        if (~np.isnan(points[:, 0])).sum() < self.kf_min_inliers:
            return  # not enough to track from; keep the old keyframe while it lasts
        self.kf = _Keyframe(frame, pose, points)
        self.recent.append(self.kf)
        self.stats.keyframes += 1

    # ---- helpers ------------------------------------------------------------------------
    def _triangulate(self, T_a: np.ndarray, T_b: np.ndarray, pa: np.ndarray,  # noqa: N803
                     pb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points and a mask: in front of both cameras, small error, enough parallax."""
        X = triangulate(T_a, T_b, pa, pb, self.cam.K)
        if len(X) == 0:
            return X, np.zeros(0, bool)
        ea = reprojection_error(X, pa, self.cam.K, invert(T_a))
        eb = reprojection_error(X, pb, self.cam.K, invert(T_b))
        good = (ea < self.max_reproj) & (eb < self.max_reproj)
        good &= parallax_deg(X, T_a, T_b) > self.min_parallax
        return X, good

    def _locate(self, frame: Frame) -> np.ndarray | None:
        kf = self.kf
        m = match(kf.frame.desc, frame.desc)
        if len(m) == 0:
            return None
        m = m[~np.isnan(kf.points[m[:, 0], 0])]
        est = pnp(kf.points[m[:, 0]], frame.pts[m[:, 1]], self.cam.K, self.max_reproj,
                  self.min_inliers) if len(m) else None
        return None if est is None else invert(est.T)


def _scaled(T: np.ndarray, s: float) -> np.ndarray:  # noqa: N803
    out = T.copy()
    out[:3, 3] *= s
    return out
