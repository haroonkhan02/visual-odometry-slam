"""ORB keypoints, undistortion and descriptor matching: the shared front end of both VO modes.

Keypoints are detected on the raw (distorted) image. Their undistorted pixel coordinates are
used for all geometry, while the raw coordinates are kept for depth lookup, because TUM
registered the depth images to the distorted RGB images.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Camera


@dataclass
class Frame:
    """Keypoints of one image: raw pixels (N, 2), undistorted pixels (N, 2), descriptors."""

    raw: np.ndarray
    pts: np.ndarray
    desc: np.ndarray

    def __len__(self) -> int:
        return len(self.pts)


def undistort_points(pts: np.ndarray, cam: Camera) -> np.ndarray:
    """Raw pixel coordinates -> undistorted pixel coordinates (same intrinsics K)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if not np.any(cam.D) or len(pts) == 0:
        return pts.reshape(-1, 2).copy()
    return cv2.undistortPoints(pts, cam.K, cam.D, P=cam.K).reshape(-1, 2)


class OrbExtractor:
    """ORB on a grid of cells so keypoints cover the whole image, not just the most textured spot.

    Plain ORB with nfeatures=N keeps the N strongest corners, which on a desk scene can all sit
    on one keyboard; well-spread points make PnP and the essential matrix far better conditioned.
    """

    def __init__(self, cam: Camera, n_features: int = 2000, grid: tuple[int, int] = (4, 3),
                 fast_threshold: int = 12):
        self.cam = cam
        self.n_features = n_features
        self.grid = grid
        self.orb = cv2.ORB_create(nfeatures=n_features * 2, fastThreshold=fast_threshold,
                                  scaleFactor=1.2, nlevels=8)

    def __call__(self, gray: np.ndarray) -> Frame:
        kps = self.orb.detect(gray, None)
        kps = self._bucket(kps, gray.shape)
        kps, desc = self.orb.compute(gray, kps)
        if desc is None or not kps:
            empty = np.empty((0, 2))
            return Frame(empty, empty.copy(), np.empty((0, 32), np.uint8))
        raw = np.array([k.pt for k in kps], np.float64)
        return Frame(raw, undistort_points(raw, self.cam), desc)

    def _bucket(self, kps: list, shape: tuple[int, int]) -> list:
        """Keep the strongest keypoints per grid cell, at most n_features overall."""
        gx, gy = self.grid
        h, w = shape[:2]
        per_cell = int(np.ceil(self.n_features / (gx * gy)))
        cells: dict[tuple[int, int], list] = {}
        for k in kps:
            cell = (min(int(k.pt[0] * gx / w), gx - 1), min(int(k.pt[1] * gy / h), gy - 1))
            cells.setdefault(cell, []).append(k)
        out = []
        for group in cells.values():
            group.sort(key=lambda k: -k.response)
            out += group[:per_cell]
        return out


def match(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Mutual nearest-neighbour Hamming matches that pass Lowe's ratio test; (M, 2) indices."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    fwd = bf.knnMatch(desc_a, desc_b, k=2)
    bwd = bf.match(desc_b, desc_a)
    back = {m.queryIdx: m.trainIdx for m in bwd}
    pairs = [(m[0].queryIdx, m[0].trainIdx) for m in fwd
             if len(m) == 2 and m[0].distance < ratio * m[1].distance
             and back.get(m[0].trainIdx) == m[0].queryIdx]
    return np.array(pairs, int).reshape(-1, 2)
