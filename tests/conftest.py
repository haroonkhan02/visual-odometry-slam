"""Synthetic TUM-format sequences and trajectories: no dataset, camera or network needed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vo_slam.geometry import make_pose, quat_to_rot


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    return quat_to_rot(rng.normal(size=4))


def circle_trajectory(n: int = 200, radius: float = 2.0, dt: float = 0.1) -> tuple:
    """Camera moving on a horizontal circle while yawing; returns (stamps, poses)."""
    stamps = np.arange(n) * dt
    poses = np.empty((n, 4, 4))
    for k, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        yaw = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        poses[k] = make_pose(yaw, [radius * np.cos(a), 0.1 * np.sin(3 * a), radius * np.sin(a)])
    return stamps, poses


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def tum_folder(tmp_path):
    """A 6-frame TUM sequence: RGB at 30 Hz, depth offset by 4 ms, GT at 100 Hz."""
    root = tmp_path / "rgbd_dataset_freiburg1_fake"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    t_rgb = 1305031102.175 + np.arange(6) / 30
    rgb_lines, depth_lines = ["# color images", "# timestamp filename"], ["# depth"]
    for k, t in enumerate(t_rgb):
        img = np.full((48, 64, 3), 40 * k, np.uint8)
        cv2.imwrite(str(root / "rgb" / f"{t:.6f}.png"), img)
        depth = np.full((48, 64), 5000 * (k + 1), np.uint16)  # k+1 metres
        depth[:8, :8] = 0  # a patch with no sensor reading
        cv2.imwrite(str(root / "depth" / f"{t + 0.004:.6f}.png"), depth)
        rgb_lines.append(f"{t:.6f} rgb/{t:.6f}.png")
        depth_lines.append(f"{t + 0.004:.6f} depth/{t + 0.004:.6f}.png")
    # an extra depth frame with no RGB partner
    depth_lines.append(f"{t_rgb[-1] + 0.5:.6f} depth/missing.png")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    t_gt = np.arange(t_rgb[0] - 0.05, t_rgb[3], 0.01)  # GT stops before the last two frames
    gt = ["# ground truth trajectory"]
    gt += [f"{t:.4f} {0.1 * i:.4f} 0 0 0 0 0 1" for i, t in enumerate(t_gt)]
    (root / "groundtruth.txt").write_text("\n".join(gt) + "\n")
    return root
