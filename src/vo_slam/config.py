"""Paths, sequences and camera calibration for the TUM RGB-D benchmark.

TUM RGB-D (https://cvg.cit.tum.de/data/datasets/rgbd-dataset) is CC BY 4.0 and downloads
without an account. KITTI odometry needs a registration, so it is not fetched automatically
(see README for the manual steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TUM_DIR = DATA_DIR / "tum"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"
CONFIGS_DIR = ROOT / "configs"

TUM_URL = "https://cvg.cit.tum.de/rgbd/dataset/{group}/{name}.tgz"
DEPTH_FACTOR = 5000.0  # 16-bit depth PNG value per metre (pre-scaled by TUM)


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics plus OpenCV (k1, k2, p1, p2, k3) distortion."""

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    width: int = 640
    height: int = 480

    @property
    def K(self) -> np.ndarray:  # noqa: N802 (standard notation)
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def D(self) -> np.ndarray:  # noqa: N802
        return np.array(self.dist, dtype=np.float64)


# From the TUM "file formats" page; TUM lists d0..d4 in OpenCV order (k1, k2, p1, p2, k3).
# Freiburg 3 images are already undistorted.
CAMERAS = {
    "fr1": Camera("fr1", 517.3, 516.5, 318.6, 255.3, (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)),
    "fr2": Camera("fr2", 520.9, 521.0, 325.1, 249.7, (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)),
    "fr3": Camera("fr3", 535.4, 539.2, 320.1, 247.6),
}


@dataclass(frozen=True)
class SequenceInfo:
    name: str  # e.g. "rgbd_dataset_freiburg1_xyz"
    camera: str
    size_bytes: int
    note: str

    @property
    def group(self) -> str:
        return {"fr1": "freiburg1", "fr2": "freiburg2", "fr3": "freiburg3"}[self.camera]

    @property
    def url(self) -> str:
        return TUM_URL.format(group=self.group, name=self.name)

    @property
    def short(self) -> str:  # "fr1_xyz"
        return self.camera + "_" + self.name.split(f"{self.group}_", 1)[1]


SEQUENCES = {
    s.short: s
    for s in (
        SequenceInfo("rgbd_dataset_freiburg1_xyz", "fr1", 448_204_271,
                     "translation along x/y/z, debugging sequence"),
        SequenceInfo("rgbd_dataset_freiburg1_desk", "fr1", 344_011_403,
                     "hand-held sweep over a desk, fast rotation"),
        SequenceInfo("rgbd_dataset_freiburg1_room", "fr1", 782_381_450,
                     "whole office room, returns to the start (loop)"),
        SequenceInfo("rgbd_dataset_freiburg3_long_office_household", "fr3", 1_483_556_251,
                     "long loop around two desks (loop closure test)"),
    )
}
DEFAULT_SEQUENCES = tuple(SEQUENCES)
