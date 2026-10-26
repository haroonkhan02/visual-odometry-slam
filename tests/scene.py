"""Synthetic scenes for the VO tests: point clouds seen by known cameras, and rendered images
of a textured wall so the full ORB -> PnP pipeline can run without the dataset."""

from __future__ import annotations

import cv2
import numpy as np

from vo_slam.config import Camera
from vo_slam.geometry import invert, make_pose

CAM = Camera("test", 500.0, 500.0, 320.0, 240.0)


def project(points: np.ndarray, T_wc: np.ndarray, K: np.ndarray) -> np.ndarray:  # noqa: N803
    Xc = (invert(T_wc)[:3, :3] @ points.T).T + invert(T_wc)[:3, 3]
    uv = (K @ Xc.T).T
    return uv[:, :2] / uv[:, 2:3]


def point_cloud(rng: np.random.Generator, n: int = 300) -> np.ndarray:
    """Points 2-6 m in front of a camera at the origin looking along +z."""
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n),
                            rng.uniform(2, 6, n)])


def small_motion(rng: np.random.Generator, trans: float = 0.2, rot_deg: float = 5.0
                 ) -> np.ndarray:
    axis = rng.normal(size=3)
    rvec = axis / np.linalg.norm(axis) * np.radians(rot_deg)
    return make_pose(cv2.Rodrigues(rvec)[0], rng.normal(size=3) * trans / np.sqrt(3))


def texture(size: int = 1600, seed: int = 0) -> np.ndarray:
    """Blobs and lines at several scales: plenty of ORB corners, few repeated patterns."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 128, np.uint8)
    for _ in range(900):
        c = tuple(int(v) for v in rng.integers(0, size, 2))
        cv2.circle(img, c, int(rng.integers(3, 25)), int(rng.integers(0, 256)), -1)
    for _ in range(300):
        p, q = rng.integers(0, size, (2, 2))
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q),
                 int(rng.integers(0, 256)), int(rng.integers(1, 4)))
    return cv2.GaussianBlur(img, (3, 3), 0)


def _render_plane(tex: np.ndarray, T_wc: np.ndarray, cam: Camera, z: float, x0: float,
                  y0: float, size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image, depth and coverage mask of a textured square at world z = z.

    The square spans [x0, x0 + size] x [y0, y0 + size]; texture pixel (u, v) sits at world
    (x0 + u * s, y0 + v * s, z), so the image is a homography of the texture.
    """
    s = size / tex.shape[1]
    T_cw = invert(T_wc)
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    A = np.array([[s, 0, x0], [0, s, y0], [0, 0, z]])  # world point = A @ [u, v, 1]
    H = cam.K @ (R @ A + np.outer(t, [0, 0, 1]))
    size_px = (cam.width, cam.height)
    gray = cv2.warpPerspective(tex, H, size_px, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(tex.shape, 255, np.uint8), H, size_px,
                               flags=cv2.INTER_NEAREST) > 0
    u, v = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
    rays = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u, float)], -1)
    dirs = rays @ T_wc[:3, :3].T
    lam = (z - T_wc[2, 3]) / dirs[..., 2]  # depth along the optical axis (ray has z = 1)
    return gray, lam.astype(np.float32), mask


def render_wall(tex: np.ndarray, T_wc: np.ndarray, cam: Camera = CAM, wall_z: float = 2.5,
                metres: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Grey image and depth (m) of a textured wall at world z = wall_z with a panel 1.2 m in
    front of it, left of centre.

    The panel matters: from a single plane, a small sideways move and a small pan look
    almost the same, which makes pose errors depend on numerical details of the platform.
    """
    gray, depth, _ = _render_plane(tex, T_wc, cam, wall_z, -metres / 2, -metres / 2, metres)
    near = np.ascontiguousarray(tex[::-1, ::-1])  # a different-looking texture
    g2, d2, m2 = _render_plane(near, T_wc, cam, wall_z - 1.2, -0.7, -0.5, 0.8)
    gray[m2], depth[m2] = g2[m2], d2[m2]
    return gray, depth
