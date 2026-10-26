import numpy as np
import pytest
from scene import CAM, render_wall, texture

from vo_slam.geometry import make_pose
from vo_slam.metrics import ate
from vo_slam.mono_vo import MonoOdometry


def slide(n: int) -> list[np.ndarray]:
    """Mostly sideways motion, 2.3 cm per frame: enough parallax to initialise in a few frames."""
    return [make_pose(np.eye(3), [0.02 * k, 0.005 * k, 0.01 * k]) for k in range(n)]


@pytest.fixture(scope="module")
def tex():
    return texture()


def test_mono_odometry_tracks_up_to_scale(tex):
    truth = slide(30)
    vo = MonoOdometry(CAM)
    for T in truth:
        vo.track(render_wall(tex, T)[0])
    est = vo.trajectory()
    assert vo.stats.inits == 1 and vo.stats.losses == 0
    assert vo.stats.untracked == 0  # frames before initialisation were located afterwards
    res = ate(np.array(truth), est, with_scale=True)
    assert res.rmse < 0.03  # on a 0.67 m path
    assert res.scale > 0
    # The unit is arbitrary, but it is one unit: the second half of the constant-speed path is
    # about as long as the first (single steps jitter, so compare windows)
    first = np.linalg.norm(est[16, :3, 3] - est[3, :3, 3])
    second = np.linalg.norm(est[29, :3, 3] - est[16, :3, 3])
    assert 0.75 < second / first < 1.33


def test_frames_without_a_pose_copy_the_previous_one(tex):
    truth = slide(12)
    vo = MonoOdometry(CAM, max_lost=50)
    for T in truth[:10]:
        vo.track(render_wall(tex, T)[0])
    blank = np.zeros((CAM.height, CAM.width), np.uint8)
    vo.track(blank)
    vo.track(blank)
    est = vo.trajectory()
    assert vo.stats.failed == 2 and vo.stats.untracked == 2
    np.testing.assert_array_equal(est[10], est[9])
    np.testing.assert_array_equal(est[11], est[9])


def test_tracking_loss_reinitialises_with_the_old_depth(tex):
    truth = slide(40)
    vo = MonoOdometry(CAM, max_lost=2)
    for T in truth[:15]:
        vo.track(render_wall(tex, T)[0])
    depth_before = vo.kf.depth_median()
    blank = np.zeros((CAM.height, CAM.width), np.uint8)
    vo.track(blank)
    vo.track(blank)
    assert vo.stats.losses == 1 and vo.kf is None
    for T in truth[15:]:
        vo.track(render_wall(tex, T)[0])
    assert vo.stats.inits == 2
    assert vo.kf.depth_median() == pytest.approx(depth_before, rel=0.3)


def test_trajectory_needs_an_initialisation():
    vo = MonoOdometry(CAM)
    vo.poses = [None, None]
    with pytest.raises(RuntimeError):
        vo.trajectory()
