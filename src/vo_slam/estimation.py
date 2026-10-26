"""Geometric estimators shared by the VO front ends: PnP, essential matrix, triangulation.

All image points here are undistorted pixels, so only K is needed. Relative poses are
returned as T_ba: a 4x4 transform taking points from camera a's frame into camera b's frame
(X_b = R X_a + t), which is what OpenCV's solvers produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import invert, make_pose


@dataclass
class Estimate:
    T: np.ndarray  # 4x4 transform, see module docstring
    inliers: np.ndarray  # indices into the input correspondences

    @property
    def n_inliers(self) -> int:
        return len(self.inliers)


def pnp(points3d: np.ndarray, pts2d: np.ndarray, K: np.ndarray,  # noqa: N803
        threshold_px: float = 2.0, min_inliers: int = 15, iterations: int = 300,
        seed: int = 0) -> Estimate | None:
    """Camera pose from 3-D points (in a reference frame) and their pixels in this camera.

    RANSAC with EPnP on minimal samples, then Levenberg-Marquardt refinement on the inliers.
    Returns T_cam_ref, or None when fewer than `min_inliers` survive.
    """
    if len(points3d) < max(6, min_inliers):
        return None
    cv2.setRNGSeed(seed)
    obj = np.ascontiguousarray(points3d, np.float64).reshape(-1, 1, 3)
    img = np.ascontiguousarray(pts2d, np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, None, iterationsCount=iterations, reprojectionError=threshold_px,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    # Re-score all points with the refined pose: LM can move a few points in or out
    err = reprojection_error(points3d, pts2d, K, make_pose(R, tvec))
    inl = np.flatnonzero(err < threshold_px)
    if len(inl) < min_inliers:
        return None
    return Estimate(make_pose(R, tvec), inl)


def essential(pts_a: np.ndarray, pts_b: np.ndarray, K: np.ndarray,  # noqa: N803
              threshold_px: float = 1.0, min_inliers: int = 30) -> Estimate | None:
    """Relative motion T_ba from two views; translation has unit length (scale unknown).

    Five-point RANSAC for E, then the cheirality check in recoverPose picks the one of four
    (R, t) decompositions that puts the points in front of both cameras.
    """
    if len(pts_a) < max(8, min_inliers):
        return None
    a = np.ascontiguousarray(pts_a, np.float64)
    b = np.ascontiguousarray(pts_b, np.float64)
    E, mask = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=0.999,
                                   threshold=threshold_px)
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, mask = cv2.recoverPose(E, a, b, K, mask=mask.copy())
    inl = np.flatnonzero(mask.ravel() > 0)
    if n < min_inliers:
        return None
    return Estimate(make_pose(R, t.ravel()), inl)


def triangulate(T_a: np.ndarray, T_b: np.ndarray, pts_a: np.ndarray,  # noqa: N803
                pts_b: np.ndarray, K: np.ndarray) -> np.ndarray:  # noqa: N803
    """Linear (DLT) triangulation; T_a, T_b are camera-to-world poses, result in world (N, 3)."""
    if len(pts_a) == 0:
        return np.empty((0, 3))
    Pa = K @ invert(T_a)[:3]
    Pb = K @ invert(T_b)[:3]
    X = cv2.triangulatePoints(Pa, Pb, np.asarray(pts_a, np.float64).T,
                              np.asarray(pts_b, np.float64).T)
    return (X[:3] / X[3]).T


def reprojection_error(points: np.ndarray, pts2d: np.ndarray, K: np.ndarray,  # noqa: N803
                       T_cam: np.ndarray) -> np.ndarray:  # noqa: N803
    """Pixel error of points (N, 3) projected by a camera with transform T_cam (world->cam).

    Points behind the camera get an infinite error.
    """
    Xc = (T_cam[:3, :3] @ np.asarray(points, np.float64).T).T + T_cam[:3, 3]
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (K @ Xc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
    err = np.linalg.norm(uv - pts2d, axis=1)
    err[~(z > 1e-6)] = np.inf
    return err


def parallax_deg(points: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:  # noqa: N803
    """Angle (degrees) between the two viewing rays of each world point."""
    ra = points - T_a[:3, 3]
    rb = points - T_b[:3, 3]
    cos = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
