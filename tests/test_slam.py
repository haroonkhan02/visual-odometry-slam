import cv2
import numpy as np
from conftest import circle_trajectory
from scene import CAM, render_wall, texture

from vo_slam.evaluate import loop_accuracy
from vo_slam.geometry import invert, make_pose, rotation_angle
from vo_slam.rgbd_vo import RgbdOdometry
from vo_slam.slam import Edge, Vocabulary, close_loops, edge_error, optimise


def drifting_chain(poses: np.ndarray, yaw_bias_deg: float) -> list[Edge]:
    """Odometry edges whose rotation is off by a constant yaw per step (heading drift)."""
    bias = make_pose(cv2.Rodrigues(np.array([0.0, np.radians(yaw_bias_deg), 0.0]))[0], [0, 0, 0])
    return [Edge(k, k + 1, invert(poses[k]) @ poses[k + 1] @ bias, 0.02, np.radians(1.0))
            for k in range(len(poses) - 1)]


def chain(edges: list[Edge]) -> np.ndarray:
    poses = [np.eye(4)]
    for e in edges:
        poses.append(poses[-1] @ e.T_ij)
    return np.array(poses)


def test_consistent_graph_is_left_alone():
    _, gt = circle_trajectory(n=30)
    gt = np.array([invert(gt[0]) @ T for T in gt])  # node 0 at the origin
    edges = drifting_chain(gt, 0.0)
    edges.append(Edge(0, 29, invert(gt[0]) @ gt[29], 0.02, np.radians(1.0), loop=True))
    out = optimise(gt, edges)
    np.testing.assert_allclose(out, gt, atol=1e-6)


def test_loop_edge_removes_accumulated_drift():
    _, gt = circle_trajectory(n=40)
    gt = np.array([invert(gt[0]) @ T for T in gt])
    odo_edges = drifting_chain(gt, 0.5)  # 0.5 deg per step -> ~20 deg by the end
    odo = chain(odo_edges)
    loop = Edge(0, 39, invert(gt[0]) @ gt[39], 0.02, np.radians(1.0), loop=True)
    out = optimise(odo, odo_edges + [loop])
    err_odo = np.linalg.norm(odo[:, :3, 3] - gt[:, :3, 3], axis=1).max()
    err_opt = np.linalg.norm(out[:, :3, 3] - gt[:, :3, 3], axis=1).max()
    assert err_odo > 0.5 and err_opt < 0.25 * err_odo
    t_err, r_err = edge_error(out, loop)
    assert t_err < 0.05 and r_err < 1.0  # the loop is (nearly) closed


def test_vocabulary_scores_same_place_above_other_places(rng):
    images = [rng.integers(0, 256, (300, 32), dtype=np.uint8) for _ in range(6)]
    voc = Vocabulary(images, k=64)
    v = np.array([voc.vector(d) for d in images])
    noisy = images[2].copy()
    flip = rng.random(noisy.shape) < 0.02  # a few bits differ, like a re-observed corner
    noisy[flip] ^= np.uint8(1)
    scores = v @ voc.vector(noisy)
    assert int(np.argmax(scores)) == 2
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)


def out_and_back(n: int = 13, step: float = 0.06) -> list[np.ndarray]:
    """Slide right with a slight pan, then come back along the same path."""
    go = []
    for k in range(n):
        a = np.radians(0.5 * k)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        go.append(make_pose(R, [step * k, 0.0, 0.01 * k]))
    return go + go[-2::-1]


def test_close_loops_on_a_rendered_revisit():
    tex = texture()
    truth = out_and_back()
    vo = RgbdOdometry(CAM)
    for T in truth:
        vo.track(*render_wall(tex, T))
    stamps = np.arange(len(truth), dtype=float)  # 1 s per frame
    poses, stats = close_loops(vo, stamps, min_gap_s=6.0)
    assert stats.keyframes >= 3 and stats.loops_kept >= 1
    for T, E in zip(truth, poses, strict=True):
        assert np.linalg.norm(T[:3, 3] - E[:3, 3]) < 0.01
        assert np.degrees(rotation_angle(T[:3, :3].T @ E[:3, :3])) < 0.3
    # loop edges are close to the true relative pose
    for (a, b), T in zip(stats.loop_pairs, stats.loop_T, strict=True):
        E = invert(invert(truth[a]) @ truth[b]) @ T
        assert np.linalg.norm(E[:3, 3]) < 0.01


def test_loop_accuracy_flags_a_wrong_loop():
    t, gt = circle_trajectory(n=50)
    good = invert(gt[3]) @ gt[45]
    bad = invert(gt[10]) @ gt[49] @ make_pose(np.eye(3), [0.4, 0, 0])  # 40 cm off
    rows = [[3, 45, *good[:3].ravel()], [5, 40, *(invert(gt[5]) @ gt[40])[:3].ravel()],
            [10, 49, *bad[:3].ravel()]]
    r = loop_accuracy(t, gt, t, rows)
    assert r["loops_scored"] == 3
    assert r["loop_err_median_cm"] < 1e-6 and abs(r["loop_err_max_cm"] - 40) < 1e-6
