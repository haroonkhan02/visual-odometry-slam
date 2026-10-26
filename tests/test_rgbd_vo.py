import numpy as np
from scene import CAM, render_wall, texture

from vo_slam.features import Frame
from vo_slam.geometry import make_pose, rotation_angle
from vo_slam.rgbd_vo import RgbdOdometry, lift


def wall_path(n: int = 15) -> list[np.ndarray]:
    """Camera sliding sideways and forward while panning slightly: ~1 cm, 0.3 deg per frame."""
    poses = []
    for k in range(n):
        a = np.radians(0.3 * k)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        poses.append(make_pose(R, [0.01 * k, 0.003 * k, 0.005 * k]))
    return poses


def test_lift_backprojects_and_rejects_bad_depth():
    depth = np.full((480, 640), 2.0, np.float32)
    depth[100:110, 100:110] = 0.0  # hole
    depth[300:, :] = 3.0  # depth edge at row 300
    pts = np.array([[CAM.cx, CAM.cy], [105.0, 105.0], [400.0, 300.0], [400.0, 400.0]])
    frame = Frame(pts, pts.copy(), np.zeros((4, 32), np.uint8))
    X = lift(frame, depth, CAM)
    np.testing.assert_allclose(X[0], [0, 0, 2.0])
    assert np.isnan(X[1]).all() and np.isnan(X[2]).all()
    np.testing.assert_allclose(X[3], [(400 - CAM.cx) / CAM.fx * 3, (400 - CAM.cy) / CAM.fy * 3, 3])


def test_rgbd_odometry_follows_a_rendered_camera_path():
    tex = texture()
    truth = wall_path()
    for keyframes in (False, True):
        vo = RgbdOdometry(CAM, keyframes=keyframes)
        est = [vo.track(*render_wall(tex, T)) for T in truth]
        assert vo.stats.failures == 0
        # Frame-to-frame chains 14 small errors (up to ~5 mm here, keyframes ~1 mm); a sign or
        # frame bug gives centimetres.
        for T, E in zip(truth, est, strict=True):
            assert np.linalg.norm(T[:3, 3] - E[:3, 3]) < 0.01
            assert np.degrees(rotation_angle(T[:3, :3].T @ E[:3, :3])) < 0.25
        if keyframes:
            assert vo.stats.keyframes < len(truth)


def test_lost_tracking_falls_back_to_constant_velocity():
    tex = texture()
    truth = wall_path(4)
    vo = RgbdOdometry(CAM)
    for T in truth[:3]:
        vo.track(*render_wall(tex, T))
    blank = np.zeros((480, 640), np.uint8)  # no texture -> no keypoints
    pose = vo.track(blank, np.zeros((480, 640), np.float32))
    assert vo.stats.failures == 1
    step = truth[2][:3, 3] - truth[1][:3, 3]
    np.testing.assert_allclose(pose[:3, 3], vo.poses[2][:3, 3] + step, atol=0.005)
