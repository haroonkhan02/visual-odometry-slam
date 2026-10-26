import numpy as np
from scene import CAM, point_cloud, project, small_motion

from vo_slam.estimation import essential, parallax_deg, pnp, reprojection_error, triangulate
from vo_slam.geometry import invert, rotation_angle

K = CAM.K


def test_pnp_recovers_pose_despite_outliers(rng):
    X = point_cloud(rng)
    T_wc = small_motion(rng)
    uv = project(X, T_wc, K) + rng.normal(scale=0.3, size=(len(X), 2))
    bad = rng.choice(len(X), 60, replace=False)
    uv[bad] += rng.uniform(20, 80, (60, 2))  # 20 % gross outliers
    est = pnp(X, uv, K)
    assert est is not None
    np.testing.assert_allclose(invert(est.T), T_wc, atol=5e-3)
    assert not set(bad) & set(est.inliers)


def test_pnp_gives_up_on_too_few_points(rng):
    X = point_cloud(rng, 5)
    assert pnp(X, project(X, np.eye(4), K), K) is None


def test_essential_recovers_rotation_and_translation_direction(rng):
    X = point_cloud(rng)
    T_wb = small_motion(rng, trans=0.8)
    a, b = project(X, np.eye(4), K), project(X, T_wb, K)
    est = essential(a + rng.normal(scale=0.2, size=a.shape),
                    b + rng.normal(scale=0.2, size=b.shape), K)
    assert est is not None
    T_ba = invert(T_wb)
    assert np.degrees(rotation_angle(est.T[:3, :3].T @ T_ba[:3, :3])) < 0.3
    t_true = T_ba[:3, 3] / np.linalg.norm(T_ba[:3, 3])
    assert np.isclose(np.linalg.norm(est.T[:3, 3]), 1.0)
    assert np.dot(est.T[:3, 3], t_true) > 0.998  # same direction (< 3.6 deg), not flipped


def test_triangulation_and_reprojection(rng):
    X = point_cloud(rng, 50)
    T_b = small_motion(rng, trans=0.5)
    a, b = project(X, np.eye(4), K), project(X, T_b, K)
    Xh = triangulate(np.eye(4), T_b, a, b, K)
    np.testing.assert_allclose(Xh, X, atol=1e-6)
    assert reprojection_error(Xh, b, K, invert(T_b)).max() < 1e-6
    # A point behind the camera is never an inlier
    assert np.isinf(reprojection_error(np.array([[0, 0, -2.0]]), np.array([[320, 240.0]]), K,
                                       np.eye(4)))[0]


def test_parallax_is_the_angle_between_rays():
    T_b = np.eye(4)
    T_b[0, 3] = 2.0
    ang = parallax_deg(np.array([[1.0, 0.0, 1.0]]), np.eye(4), T_b)
    np.testing.assert_allclose(ang, 90.0)
