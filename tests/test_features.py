import cv2
import numpy as np
from scene import CAM, render_wall, texture

from vo_slam.config import CAMERAS
from vo_slam.features import OrbExtractor, match, undistort_points


def test_undistort_inverts_opencv_distortion():
    cam = CAMERAS["fr1"]
    ideal = np.array([[50.0, 40.0], [320.0, 240.0], [600.0, 450.0]])
    norm = np.column_stack([(ideal[:, 0] - cam.cx) / cam.fx, (ideal[:, 1] - cam.cy) / cam.fy,
                            np.ones(3)])
    raw, _ = cv2.projectPoints(norm, np.zeros(3), np.zeros(3), cam.K, cam.D)
    np.testing.assert_allclose(undistort_points(raw.reshape(-1, 2), cam), ideal, atol=0.05)


def test_undistort_is_identity_without_distortion():
    pts = np.array([[10.0, 20.0], [300.0, 200.0]])
    np.testing.assert_array_equal(undistort_points(pts, CAMERAS["fr3"]), pts)


def test_keypoints_spread_over_the_grid():
    gray, _ = render_wall(texture(), np.eye(4))
    frame = OrbExtractor(CAM, n_features=600)(gray)
    assert 300 < len(frame) <= 600
    cells = {(int(x * 4 / CAM.width), int(y * 3 / CAM.height)) for x, y in frame.raw}
    assert len(cells) == 12


def test_matches_are_mutual_and_mostly_correct():
    tex = texture()
    a, _ = render_wall(tex, np.eye(4))
    shifted = np.roll(a, 7, axis=1)  # every keypoint moves 7 px to the right
    ex = OrbExtractor(CAM, n_features=800)
    fa, fb = ex(a), ex(shifted)
    m = match(fa.desc, fb.desc)
    assert len(m) > 200
    assert len(set(m[:, 0])) == len(m) and len(set(m[:, 1])) == len(m)
    shift = fb.raw[m[:, 1]] - fa.raw[m[:, 0]]
    good = np.all(np.abs(shift - [7, 0]) < 1.5, axis=1)
    assert good.mean() > 0.9


def test_match_handles_empty_descriptors():
    assert match(np.empty((0, 32), np.uint8), np.empty((0, 32), np.uint8)).shape == (0, 2)
