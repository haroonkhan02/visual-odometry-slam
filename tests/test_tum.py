import numpy as np
import pytest

from vo_slam.config import CAMERAS, SEQUENCES
from vo_slam.geometry import make_pose, quat_to_rot
from vo_slam.tum import (
    TumSequence,
    associate,
    read_file_list,
    read_trajectory,
    write_trajectory,
)


def test_read_file_list_skips_comments(tmp_path):
    f = tmp_path / "rgb.txt"
    f.write_text("# a comment\n\n1.5 rgb/1.png\n2.0,rgb/2.png\n")
    stamps, rows = read_file_list(f)
    np.testing.assert_allclose(stamps, [1.5, 2.0])
    assert rows == [["rgb/1.png"], ["rgb/2.png"]]


def test_associate_is_one_to_one_and_closest_first():
    t_a = np.array([0.00, 0.033, 0.066, 0.5])
    t_b = np.array([0.001, 0.030, 0.040, 0.9])
    # 0.033 is closest to 0.030; 0.066 -> 0.040 is 26 ms, outside 20 ms; 0.5 has no partner
    assert associate(t_a, t_b) == [(0, 0), (1, 1)]
    # offset shifts the second stream before matching
    assert associate(np.array([1.0]), np.array([0.0]), offset=1.0) == [(0, 0)]


def test_trajectory_round_trip(tmp_path, rng):
    stamps = np.array([1.0, 2.0, 3.0])
    poses = np.stack([make_pose(quat_to_rot(rng.normal(size=4)), rng.normal(size=3))
                      for _ in stamps])
    write_trajectory(tmp_path / "traj.txt", stamps, poses)
    t2, p2 = read_trajectory(tmp_path / "traj.txt")
    np.testing.assert_allclose(t2, stamps)
    np.testing.assert_allclose(p2, poses, atol=1e-5)


def test_sequence_pairs_frames_and_loads_images(tum_folder):
    seq = TumSequence(tum_folder, CAMERAS["fr1"])
    assert len(seq) == 6  # the unmatched depth line is ignored
    assert seq.gray(2).shape == (48, 64) and seq.gray(2).dtype == np.uint8
    depth = seq.depth(2)
    assert depth.dtype == np.float32 and depth[20, 20] == pytest.approx(3.0) and depth[0, 0] == 0
    idx, poses = seq.ground_truth()
    assert list(idx) == [0, 1, 2, 3]  # GT ends before the last two frames
    assert poses.shape == (4, 4, 4)
    assert np.all(np.diff(poses[:, 0, 3]) > 0)  # moving along +x


def test_sequence_without_ground_truth(tum_folder):
    (tum_folder / "groundtruth.txt").unlink()
    idx, poses = TumSequence(tum_folder, CAMERAS["fr1"]).ground_truth()
    assert len(idx) == 0 and poses.shape == (0, 4, 4)


def test_sequence_registry_and_cameras():
    info = SEQUENCES["fr1_xyz"]
    assert info.url.endswith("/freiburg1/rgbd_dataset_freiburg1_xyz.tgz")
    assert SEQUENCES["fr3_long_office_household"].camera == "fr3"
    K = CAMERAS["fr1"].K
    assert K[0, 0] == 517.3 and K[1, 2] == 255.3 and K[2, 2] == 1
    assert not CAMERAS["fr3"].D.any()  # fr3 images are pre-undistorted
