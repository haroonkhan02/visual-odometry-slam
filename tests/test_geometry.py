import numpy as np
from conftest import random_rotation

from vo_slam.geometry import (
    invert,
    make_pose,
    path_length,
    quat_to_rot,
    relative,
    rot_to_quat,
    rotation_angle,
)


def test_quaternion_round_trip(rng):
    for _ in range(50):
        R = random_rotation(rng)
        np.testing.assert_allclose(quat_to_rot(rot_to_quat(R)), R, atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_quaternion_known_values():
    # 90 degrees about z, TUM order (x, y, z, w)
    q = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    np.testing.assert_allclose(quat_to_rot(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(quat_to_rot([0, 0, 0, 2.0]), np.eye(3))  # normalised


def test_rot_to_quat_handles_180_degree_rotation():
    R = np.diag([1.0, -1.0, -1.0])  # trace = -1, the branch without a positive w
    np.testing.assert_allclose(quat_to_rot(rot_to_quat(R)), R, atol=1e-12)


def test_invert_and_relative(rng):
    Ta = make_pose(random_rotation(rng), rng.normal(size=3))
    Tb = make_pose(random_rotation(rng), rng.normal(size=3))
    np.testing.assert_allclose(invert(Ta) @ Ta, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(Ta @ relative(Ta, Tb), Tb, atol=1e-12)


def test_rotation_angle_and_path_length():
    a = 0.3
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    assert np.isclose(rotation_angle(Rz), a)
    assert rotation_angle(np.eye(3) * (1 + 1e-12)) == 0.0  # no NaN just above identity
    assert np.isclose(rotation_angle(np.diag([1.0, -1.0, -1.0])), np.pi)
    assert np.isclose(path_length(np.array([[0, 0, 0], [3, 4, 0], [3, 4, 2]])), 7.0)
