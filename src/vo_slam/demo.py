"""Demo GIF: the camera image with its ORB keypoints next to a top view of the path so far.

The top view shows ground truth (grey), keyframe odometry (blue) and the same odometry after
loop closure (green), each aligned to ground truth with one SE3 transform for the whole run,
so the drift that loop closure removes is visible as the blue line wanders off.

Example:
    vo-demo                                   # fr3_long_office_household -> assets/demo.gif
    vo-demo --seq fr1_room --step 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT, TUM_DIR
from .features import OrbExtractor
from .metrics import umeyama
from .run import TRAJ_DIR
from .tum import TumSequence, associate, read_trajectory

TRACKS = (("rgbd-kf", "odometry", (200, 110, 30)), ("rgbd-slam", "loop closure", (40, 160, 40)))
PANEL = (320, 240)  # width, height of each half


def aligned_paths(seq: TumSequence, short: str, traj_dir: Path
                  ) -> tuple[np.ndarray, dict, np.ndarray]:
    """GT positions (N, 3) per frame (NaN where missing), aligned estimates, frame mask."""
    pairs = associate(seq.stamps, seq.gt_stamps)
    has_gt = np.zeros(len(seq), bool)
    gt = np.full((len(seq), 3), np.nan)
    for i, j in pairs:
        has_gt[i] = True
        gt[i] = seq.gt_poses[j, :3, 3]
    paths = {}
    for mode, _, _ in TRACKS:
        t, poses = read_trajectory(traj_dir / f"{short}_{mode}.txt")
        if len(t) != len(seq):
            raise SystemExit(f"{mode} trajectory does not match the sequence; re-run vo-run")
        p = poses[:, :3, 3]
        _, R, tr = umeyama(p[has_gt], gt[has_gt])
        paths[mode] = (R @ p.T).T + tr
    return gt, paths, has_gt


class TopView:
    """World x/y -> pixel mapping that fits all paths into the panel with a margin."""

    def __init__(self, pts: np.ndarray, size: tuple[int, int] = PANEL, margin: int = 14):
        pts = pts[np.isfinite(pts).all(1)][:, :2]
        lo, hi = pts.min(0), pts.max(0)
        w, h = size
        self.s = min((w - 2 * margin) / max(hi[0] - lo[0], 1e-6),
                     (h - 2 * margin - 16) / max(hi[1] - lo[1], 1e-6))
        self.off = np.array([w, h + 16]) / 2 - self.s * (lo + hi) / 2
        self.size = size

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        p = self.s * xy[:, :2] + self.off
        p[:, 1] = self.size[1] - p[:, 1] + 16  # y up
        return np.round(p).astype(np.int32)


def draw_path(img: np.ndarray, pix: np.ndarray, color: tuple, thick: int = 1) -> None:
    if len(pix) > 1:
        cv2.polylines(img, [pix.reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)


def render(seq: TumSequence, short: str, traj_dir: Path, step: int, stop: int | None = None
           ) -> list[np.ndarray]:
    gt, paths, has_gt = aligned_paths(seq, short, traj_dir)
    view = TopView(np.vstack([gt, *paths.values()]))
    orb = OrbExtractor(seq.camera, n_features=500)
    frames = []
    end = len(seq) if stop is None else min(stop, len(seq))
    for i in list(range(0, end, step)) + [end - 1] * 8:  # hold the last frame
        gray = seq.gray(i)
        img = cv2.cvtColor(cv2.imread(str(seq.root / seq.rgb_files[i])), cv2.COLOR_BGR2RGB)
        for u, v in orb(gray).raw:
            cv2.circle(img, (int(u), int(v)), 3, (255, 210, 0), 1, cv2.LINE_AA)
        left = cv2.resize(img, PANEL, interpolation=cv2.INTER_AREA)
        right = np.full((PANEL[1], PANEL[0], 3), 255, np.uint8)
        g = gt[: i + 1][has_gt[: i + 1]]
        draw_path(right, view(g), (170, 170, 170), 3)
        y = 12
        for mode, label, color in TRACKS:
            draw_path(right, view(paths[mode][: i + 1]), color[::-1], 1)
            cv2.putText(right, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color[::-1], 1,
                        cv2.LINE_AA)
            y += 13
        cv2.putText(right, "ground truth", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (130, 130, 130), 1, cv2.LINE_AA)
        cv2.putText(right, f"{seq.stamps[i] - seq.stamps[0]:5.1f} s", (PANEL[0] - 58, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        frames.append(np.hstack([left, right]))
    return frames


def save_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps),
                 loop=0, optimize=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", default="fr3_long_office_household")
    parser.add_argument("--step", type=int, default=20, help="use every n-th frame")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--data-dir", type=Path, default=TUM_DIR)
    parser.add_argument("--traj-dir", type=Path, default=TRAJ_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "demo.gif")
    args = parser.parse_args(argv)
    seq = TumSequence.load(args.seq, args.data_dir)
    frames = render(seq, args.seq, args.traj_dir, args.step)
    save_gif(frames, args.out, args.fps)
    print(f"{args.out}: {len(frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
