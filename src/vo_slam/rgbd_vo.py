"""RGB-D visual odometry: ORB keypoints lifted to 3-D with depth, PnP in the next frame.

Each frame is tracked against a reference frame. In frame-to-frame mode the reference is
always the previous frame; in keyframe mode it stays fixed until too few inliers remain, so
small errors are not added up 30 times a second while the camera barely moves.
Depth gives metric scale directly, so no scale drift (only rotation and translation drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Camera
from .estimation import pnp
from .features import Frame, OrbExtractor, match
from .geometry import invert


def lift(frame: Frame, depth: np.ndarray, cam: Camera, min_depth: float = 0.3,
         max_depth: float = 4.0) -> np.ndarray:
    """3-D points (N, 3) in the camera frame for each keypoint; NaN rows where depth is unusable.

    Depth is read at the raw pixel (depth is registered to the distorted image) and rejected
    near depth edges, where the 3x3 neighbourhood spans more than 5 % of the depth.
    """
    h, w = depth.shape
    u = np.clip(np.round(frame.raw[:, 0]).astype(int), 1, w - 2)
    v = np.clip(np.round(frame.raw[:, 1]).astype(int), 1, h - 2)
    patch = np.stack([depth[v + dv, u + du] for dv in (-1, 0, 1) for du in (-1, 0, 1)], 1)
    z = depth[v, u].astype(np.float64)
    ok = (z > min_depth) & (z < max_depth) & (patch.min(1) > 0)
    ok &= (patch.max(1) - patch.min(1)) < 0.05 * z
    x = (frame.pts[:, 0] - cam.cx) / cam.fx
    y = (frame.pts[:, 1] - cam.cy) / cam.fy
    pts = np.stack([x * z, y * z, z], 1)
    pts[~ok] = np.nan
    return pts


@dataclass
class Keyframe:
    """A reference frame kept for loop closure: its features, 3-D points and odometry pose."""

    index: int  # frame index in the sequence
    frame: Frame
    pts3d: np.ndarray
    pose: np.ndarray
    lost: bool = False  # created after a tracking failure: its link to the previous one is a guess


@dataclass
class TrackStats:
    frames: int = 0
    failures: int = 0  # frames where PnP found no pose (constant-velocity guess used)
    keyframes: int = 0
    inliers: list[int] = field(default_factory=list)


class RgbdOdometry:
    """Online RGB-D odometry. Call `track(gray, depth)` per frame; returns the pose T_wc."""

    def __init__(self, cam: Camera, keyframes: bool = True, kf_min_inliers: int = 150,
                 kf_min_ratio: float = 0.5, threshold_px: float = 2.0,
                 extractor: OrbExtractor | None = None):
        self.cam = cam
        self.keyframes = keyframes
        self.kf_min_inliers = kf_min_inliers
        self.kf_min_ratio = kf_min_ratio
        self.threshold_px = threshold_px
        self.extract = extractor or OrbExtractor(cam)
        self.stats = TrackStats()
        self.ref: tuple[Frame, np.ndarray, np.ndarray] | None = None  # frame, 3-D pts, pose
        self.ref_first_inliers = 0
        self.poses: list[np.ndarray] = []
        self.keyframe_list: list[Keyframe] = []
        # per frame: (keyframe id, T_kf_frame), so poses can be rebuilt after the graph moves
        self.frame_refs: list[tuple[int, np.ndarray]] = []

    def track(self, gray: np.ndarray, depth: np.ndarray) -> np.ndarray:
        frame = self.extract(gray)
        pts3d = lift(frame, depth, self.cam)
        self.stats.frames += 1
        if self.ref is None:
            pose = np.eye(4)
            self._set_ref(frame, pts3d, pose, 0)
            return self._record(pose, new_kf=True)
        ref_frame, ref_pts, ref_pose = self.ref
        est = self._solve(ref_frame, ref_pts, frame)
        if est is None:
            # Lost: assume the last frame-to-frame motion repeats, restart from this frame
            self.stats.failures += 1
            self.stats.inliers.append(0)
            pose = self._constant_velocity()
            self._set_ref(frame, pts3d, pose, 0, lost=True)
            return self._record(pose, new_kf=True)
        else:
            T_cur_ref, n = est
            self.stats.inliers.append(n)
            pose = ref_pose @ invert(T_cur_ref)
            if self.ref_first_inliers == 0:
                self.ref_first_inliers = n
            weak = n < self.kf_min_inliers or n < self.kf_min_ratio * self.ref_first_inliers
            if not self.keyframes or weak:
                self._set_ref(frame, pts3d, pose, 0)
                return self._record(pose, new_kf=True)
        return self._record(pose, new_kf=False)

    def _record(self, pose: np.ndarray, new_kf: bool) -> np.ndarray:
        kf = len(self.keyframe_list) - 1
        rel = np.eye(4) if new_kf else invert(self.keyframe_list[kf].pose) @ pose
        self.frame_refs.append((kf, rel))
        self.poses.append(pose)
        return pose

    def _solve(self, ref: Frame, ref_pts: np.ndarray, cur: Frame
               ) -> tuple[np.ndarray, int] | None:
        m = match(ref.desc, cur.desc)
        if len(m) == 0:
            return None
        m = m[~np.isnan(ref_pts[m[:, 0], 0])]
        est = pnp(ref_pts[m[:, 0]], cur.pts[m[:, 1]], self.cam.K, self.threshold_px)
        return None if est is None else (est.T, est.n_inliers)

    def _set_ref(self, frame: Frame, pts3d: np.ndarray, pose: np.ndarray, n: int,
                 lost: bool = False) -> None:
        self.ref = (frame, pts3d, pose)
        self.ref_first_inliers = n
        self.stats.keyframes += 1
        self.keyframe_list.append(Keyframe(self.stats.frames - 1, frame, pts3d, pose, lost))

    def _constant_velocity(self) -> np.ndarray:
        if len(self.poses) < 2:
            return self.poses[-1].copy()
        return self.poses[-1] @ (invert(self.poses[-2]) @ self.poses[-1])
