import json

import numpy as np
import pytest
from conftest import circle_trajectory

from vo_slam import evaluate as ev
from vo_slam.config import CAMERAS, SEQUENCES
from vo_slam.run import run_sequence
from vo_slam.tum import TumSequence, write_trajectory


def test_perfect_estimate_scores_zero():
    t, poses = circle_trajectory(n=100, dt=0.05)
    r = ev.evaluate(t, poses, t, poses, monocular=False)
    assert r["frames_scored"] == 100
    assert r["ate_rmse_m"] < 1e-9 and r["rpe_trans_m_per_s"] < 1e-9 and r["drift_pct"] < 1e-6


def test_monocular_scale_is_recovered_and_removed():
    t, poses = circle_trajectory(n=100, dt=0.05)
    est = poses.copy()
    est[:, :3, 3] *= 0.25  # same path in arbitrary units
    r = ev.evaluate(t, poses, t, est, monocular=True)
    assert r["sim3_scale"] == pytest.approx(4.0)
    assert r["ate_rmse_m"] < 1e-9 and r["rpe_trans_m_per_s"] < 1e-9


def test_drift_is_error_over_distance():
    t, poses = circle_trajectory(n=200, dt=0.05)
    est = poses.copy()
    # a 2 % too long trajectory: every 1 s window is 2 % off in translation
    est[:, :3, 3] *= 1.02
    r = ev.evaluate(t, poses, t, est, monocular=False)
    assert r["drift_pct"] == pytest.approx(2.0, rel=1e-6)


def test_run_sequence_rejects_unknown_mode(tum_folder):
    seq = TumSequence(tum_folder, CAMERAS["fr1"])
    with pytest.raises(ValueError):
        run_sequence(seq, "stereo")


def test_rgbd_run_on_textureless_frames_counts_failures(tum_folder):
    seq = TumSequence(tum_folder, CAMERAS["fr1"])  # uniform grey images: nothing to track
    poses, stats = run_sequence(seq, "rgbd-kf")
    assert poses.shape == (len(seq), 4, 4)
    assert stats["failures"] == len(seq) - 1 and stats["mode"] == "rgbd-kf"


def test_vo_eval_writes_report(tmp_path):
    short = "fr1_xyz"
    data = tmp_path / "tum" / SEQUENCES[short].name
    t, poses = circle_trajectory(n=120, dt=0.05)
    write_trajectory(data / "groundtruth.txt", t, poses)
    traj = tmp_path / "traj"
    for mode in ("rgbd-kf", "mono"):
        est = poses.copy()
        if mode == "mono":
            est[:, :3, 3] *= 0.5
        write_trajectory(traj / f"{short}_{mode}.txt", t + 0.003, est)
        stats = {"mode": mode, "keyframes": 10, "fps": 50.0, "failures": 0}
        (traj / f"{short}_{mode}.json").write_text(json.dumps(stats))
    ev.main(["--traj-dir", str(traj), "--data-dir", str(tmp_path / "tum"),
             "--out", str(tmp_path)])
    res = json.loads((tmp_path / "vo_results.json").read_text())
    assert res[short]["mono"]["sim3_scale"] == pytest.approx(2.0, rel=1e-4)
    assert "| fr1_xyz | rgbd-kf |" in (tmp_path / "vo_results.md").read_text()
    assert (tmp_path / "figures" / "trajectories.png").exists()
    assert np.isfinite(res[short]["rgbd-kf"]["drift_pct"])
