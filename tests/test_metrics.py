import numpy as np
import pytest
from conftest import circle_trajectory, random_rotation

from vo_slam.geometry import make_pose
from vo_slam.metrics import ate, rpe, rpe_pairs, umeyama


def transform(poses, s, R, t):
    """Apply a similarity transform to a trajectory (positions scaled, rotations composed)."""
    out = poses.copy()
    out[:, :3, :3] = R @ poses[:, :3, :3]
    out[:, :3, 3] = (s * (R @ poses[:, :3, 3].T)).T + t
    return out


def test_umeyama_recovers_similarity(rng):
    src = rng.normal(size=(100, 3))
    R, t, s = random_rotation(rng), rng.normal(size=3), 2.5
    dst = (s * (R @ src.T)).T + t
    s_hat, R_hat, t_hat = umeyama(src, dst, with_scale=True)
    assert np.isclose(s_hat, s)
    np.testing.assert_allclose(R_hat, R, atol=1e-10)
    np.testing.assert_allclose(t_hat, t, atol=1e-10)


def test_umeyama_never_returns_a_reflection(rng):
    src = rng.normal(size=(50, 3))
    dst = src * [1, 1, -1]  # mirror image: best proper rotation is not the mirror
    _, R, _ = umeyama(src, dst)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_umeyama_rejects_bad_shapes():
    with pytest.raises(ValueError):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))


def test_ate_is_zero_after_rigid_or_similarity_transform(rng):
    _, gt = circle_trajectory()
    R, t = random_rotation(rng), rng.normal(size=3)
    assert ate(gt, transform(gt, 1.0, R, t)).rmse < 1e-9
    # A monocular estimate at the wrong scale needs Sim3 alignment
    est = transform(gt, 0.3, R, t)
    assert ate(gt, est).rmse > 0.5
    res = ate(gt, est, with_scale=True)
    assert res.rmse < 1e-9 and np.isclose(res.scale, 1 / 0.3)


def test_ate_measures_known_noise(rng):
    _, gt = circle_trajectory(n=2000)
    est = gt.copy()
    est[:, :3, 3] += rng.normal(scale=0.01, size=(2000, 3))
    # 3-D isotropic noise with sigma 1 cm -> RMSE ~ sqrt(3) cm (alignment removes a little)
    assert ate(gt, est).rmse == pytest.approx(0.01 * np.sqrt(3), rel=0.05)


def test_rpe_pairs_match_the_requested_window():
    stamps = np.arange(0, 10, 0.1)
    pairs = rpe_pairs(stamps, 1.0)
    assert len(pairs) == 90
    assert all(np.isclose(stamps[j] - stamps[i], 1.0) for i, j in pairs)


def test_rpe_detects_constant_drift():
    stamps, gt = circle_trajectory(n=200, dt=0.1)
    # Estimated motion overshoots by 2 cm along the camera x-axis every 0.1 s
    est = gt.copy()
    drift = make_pose(np.eye(3), [0.02, 0, 0])
    for k in range(1, len(gt)):
        est[k] = est[k - 1] @ (np.linalg.inv(gt[k - 1]) @ gt[k]) @ drift
    res = rpe(gt, est, stamps, delta=0.1)
    assert res.trans_rmse == pytest.approx(0.02, rel=1e-6)
    assert res.rot_rmse_deg == pytest.approx(0.0, abs=1e-6)
    assert rpe(gt, gt, stamps, delta=1.0).trans_rmse < 1e-12


def test_rpe_scale_argument_undoes_monocular_scale():
    stamps, gt = circle_trajectory()
    est = gt.copy()
    est[:, :3, 3] *= 0.5
    assert rpe(gt, est, stamps, scale=2.0).trans_rmse < 1e-12
