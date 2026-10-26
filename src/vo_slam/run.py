"""Run visual odometry over TUM sequences and save the trajectories in TUM format.

Modes:
    rgbd-f2f   RGB-D, every frame tracked against the previous frame
    rgbd-kf    RGB-D, frames tracked against a keyframe until the inliers thin out
    mono       monocular (grey images only), scale fixed once at initialisation

Example:
    vo-run                                  # all downloaded sequences, all modes
    vo-run --seq fr1_xyz --mode mono        # one run
Outputs: reports/trajectories/<seq>_<mode>.txt and a .json with run statistics.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import REPORTS_DIR, SEQUENCES, TUM_DIR
from .mono_vo import MonoOdometry
from .rgbd_vo import RgbdOdometry
from .tum import TumSequence, write_trajectory

MODES = ("rgbd-f2f", "rgbd-kf", "mono")
TRAJ_DIR = REPORTS_DIR / "trajectories"


def run_sequence(seq: TumSequence, mode: str, progress: bool = False) -> tuple[np.ndarray, dict]:
    """Poses (N, 4, 4) for every frame of `seq`, plus run statistics."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    frames = tqdm(range(len(seq)), desc=f"{seq.root.name[-20:]} {mode}", disable=not progress)
    t0 = time.perf_counter()
    if mode == "mono":
        vo = MonoOdometry(seq.camera)
        for i in frames:
            vo.track(seq.gray(i))
        poses = vo.trajectory()
    else:
        vo = RgbdOdometry(seq.camera, keyframes=mode == "rgbd-kf")
        poses = np.array([vo.track(seq.gray(i), seq.depth(i)) for i in frames])
    seconds = time.perf_counter() - t0
    stats = {k: v for k, v in asdict(vo.stats).items() if k != "inliers"}
    if mode != "mono":
        stats["median_inliers"] = int(np.median(vo.stats.inliers))
    stats |= {"mode": mode, "seconds": round(seconds, 1), "fps": round(len(seq) / seconds, 1)}
    return poses, stats


def _job(args: tuple[str, str, Path, Path]) -> str:
    short, mode, data_dir, out_dir = args
    seq = TumSequence.load(short, data_dir)
    poses, stats = run_sequence(seq, mode)
    write_trajectory(out_dir / f"{short}_{mode}.txt", seq.stamps, poses)
    (out_dir / f"{short}_{mode}.json").write_text(json.dumps(stats, indent=2) + "\n")
    return f"{short:28s} {mode:9s} {stats['fps']:5.1f} fps  " + json.dumps(
        {k: v for k, v in stats.items() if k not in ("mode", "fps", "seconds")})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", nargs="*", default=None, help="sequence short names")
    parser.add_argument("--mode", nargs="*", default=None, help=f"subset of {MODES}")
    parser.add_argument("--data-dir", type=Path, default=TUM_DIR)
    parser.add_argument("--out", type=Path, default=TRAJ_DIR)
    parser.add_argument("--jobs", type=int, default=4, help="parallel processes")
    args = parser.parse_args(argv)
    seqs = args.seq or [s for s, info in SEQUENCES.items()
                        if (args.data_dir / info.name / "rgb.txt").exists()]
    modes = args.mode or list(MODES)
    bad = [s for s in seqs if s not in SEQUENCES] + [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown sequence/mode: {bad}")
    if not seqs:
        raise SystemExit(f"No sequences in {args.data_dir}; run vo-download first.")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m, args.data_dir, args.out) for s in seqs for m in modes]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
