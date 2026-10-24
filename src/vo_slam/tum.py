"""Reading and writing the TUM RGB-D formats.

A sequence folder holds `rgb.txt` / `depth.txt` (timestamp + image path), the images, and
`groundtruth.txt` (timestamp tx ty tz qx qy qz qw at 100 Hz, motion capture). RGB and depth are
captured at 30 Hz by separate streams, so frames are paired by nearest timestamp, as the TUM
`associate.py` tool does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import CAMERAS, DEPTH_FACTOR, SEQUENCES, TUM_DIR, Camera
from .geometry import make_pose, quat_to_rot, rot_to_quat


def read_file_list(path: Path) -> tuple[np.ndarray, list[list[str]]]:
    """Timestamps and the remaining columns of each non-comment line."""
    stamps, rows = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        stamps.append(float(parts[0]))
        rows.append(parts[1:])
    return np.array(stamps, dtype=np.float64), rows


def associate(t_a: np.ndarray, t_b: np.ndarray, max_dt: float = 0.02,
              offset: float = 0.0) -> list[tuple[int, int]]:
    """One-to-one pairs (i, j) with |t_a[i] - (t_b[j] + offset)| < max_dt, closest first.

    Same greedy rule as TUM's associate.py; the result is sorted by i.
    """
    t_a, t_b = np.asarray(t_a), np.asarray(t_b) + offset
    cands = []
    for i, t in enumerate(t_a):
        lo, hi = np.searchsorted(t_b, [t - max_dt, t + max_dt])
        cands += [(abs(t - t_b[j]), i, j) for j in range(lo, hi) if abs(t - t_b[j]) < max_dt]
    cands.sort()
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """TUM trajectory file -> timestamps (N,) and camera-to-world poses (N, 4, 4)."""
    stamps, rows = read_file_list(path)
    poses = np.empty((len(rows), 4, 4))
    for k, r in enumerate(rows):
        v = [float(x) for x in r[:7]]
        poses[k] = make_pose(quat_to_rot(v[3:7]), v[:3])
    return stamps, poses


def write_trajectory(path: Path, stamps: np.ndarray, poses: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for t, T in zip(stamps, poses, strict=True):
        q = rot_to_quat(T[:3, :3])
        lines.append(f"{t:.6f} " + " ".join(f"{v:.6f}" for v in (*T[:3, 3], *q)))
    path.write_text("\n".join(lines) + "\n")


@dataclass
class TumSequence:
    """A TUM RGB-D sequence: RGB-depth frame pairs, each with a ground-truth pose if available."""

    root: Path
    camera: Camera
    stamps: np.ndarray = field(init=False)  # RGB timestamps of the paired frames
    rgb_files: list[str] = field(init=False)
    depth_files: list[str] = field(init=False)
    gt_stamps: np.ndarray = field(init=False)
    gt_poses: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        t_rgb, rgb = read_file_list(self.root / "rgb.txt")
        t_depth, depth = read_file_list(self.root / "depth.txt")
        pairs = associate(t_rgb, t_depth)
        self.stamps = t_rgb[[i for i, _ in pairs]]
        self.rgb_files = [rgb[i][0] for i, _ in pairs]
        self.depth_files = [depth[j][0] for _, j in pairs]
        gt_path = self.root / "groundtruth.txt"
        if gt_path.exists():
            self.gt_stamps, self.gt_poses = read_trajectory(gt_path)
        else:
            self.gt_stamps, self.gt_poses = np.empty(0), np.empty((0, 4, 4))

    @classmethod
    def load(cls, short: str, data_dir: Path = TUM_DIR) -> TumSequence:
        info = SEQUENCES[short]
        return cls(data_dir / info.name, CAMERAS[info.camera])

    def __len__(self) -> int:
        return len(self.stamps)

    def gray(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.root / self.rgb_files[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.root / self.rgb_files[i])
        return img

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres (float32); 0 where the sensor has no reading."""
        raw = cv2.imread(str(self.root / self.depth_files[i]), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(self.root / self.depth_files[i])
        return raw.astype(np.float32) / DEPTH_FACTOR

    def ground_truth(self, max_dt: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
        """Frame indices that have a ground-truth pose within max_dt, and those poses."""
        pairs = associate(self.stamps, self.gt_stamps, max_dt)
        idx = np.array([i for i, _ in pairs], dtype=int)
        return idx, self.gt_poses[[j for _, j in pairs]]
