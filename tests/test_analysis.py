import json

import numpy as np
import pytest
from conftest import circle_trajectory

from vo_slam import analysis
from vo_slam.config import SEQUENCES
from vo_slam.geometry import make_pose
from vo_slam.tum import write_trajectory


def test_window_errors_measure_gt_rotation_and_error():
    t, gt = circle_trajectory(n=100, dt=0.05)  # 360 deg in 5 s -> 72 deg per 1 s window
    est = gt.copy()
    est[:, :3, 3] *= 1.1
    w = analysis.window_errors(t, gt, est)
    assert len(w) > 0
    np.testing.assert_allclose(w[:, 0], 72.0, atol=0.5)
    assert np.all(w[:, 1] > 0) and np.all(w[:, 2] < 1e-6)  # translation-only error


def test_segment_drift_is_percent_of_length():
    gt = np.array([make_pose(np.eye(3), [0.01 * k, 0, 0]) for k in range(500)])  # 5 m line
    est = gt.copy()
    est[:, :3, 3] *= 1.03
    np.testing.assert_allclose(analysis.segment_drift(gt, est, 1.0), 3.0, rtol=1e-6)
    assert len(analysis.segment_drift(gt, est, 10.0)) == 0  # longer than the path


def test_vo_analyse_writes_report(tmp_path):
    short = "fr1_xyz"
    data = tmp_path / "tum" / SEQUENCES[short].name
    t, gt = circle_trajectory(n=200, dt=0.05)
    write_trajectory(data / "groundtruth.txt", t, gt)
    for mode, scale in (("rgbd-kf", 1.0), ("mono", 0.5)):
        est = gt.copy()
        est[:, :3, 3] *= scale
        write_trajectory(tmp_path / "traj" / f"{short}_{mode}.txt", t, est)
    analysis.main(["--traj-dir", str(tmp_path / "traj"), "--data-dir", str(tmp_path / "tum"),
                   "--out", str(tmp_path)])
    res = json.loads((tmp_path / "error_analysis.json").read_text())
    # mono is rescaled by its Sim3 scale before scoring, so both are perfect
    for mode in ("rgbd-kf", "mono"):
        assert res["by_rotation_speed"][mode]["20–40"]["trans_cm"] == pytest.approx(0, abs=0.01)
    assert (tmp_path / "figures" / "error_analysis.png").exists()
