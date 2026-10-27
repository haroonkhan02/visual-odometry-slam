import numpy as np

from vo_slam import demo
from vo_slam.config import CAMERAS
from vo_slam.geometry import make_pose
from vo_slam.tum import TumSequence, write_trajectory


def test_top_view_fits_points_inside_the_panel():
    pts = np.array([[-3.0, -2.0, 0.0], [4.0, 1.0, 0.0], [np.nan, np.nan, np.nan]])
    view = demo.TopView(pts)
    pix = view(pts[:2])
    w, h = demo.PANEL
    assert (pix[:, 0] >= 0).all() and (pix[:, 0] < w).all()
    assert (pix[:, 1] >= 0).all() and (pix[:, 1] < h).all()
    assert pix[0, 1] > pix[1, 1]  # larger world y is higher up in the image


def test_render_and_save_gif(tum_folder, tmp_path):
    seq = TumSequence(tum_folder, CAMERAS["fr1"])
    poses = np.array([make_pose(np.eye(3), [0.1 * k, 0.02 * k * k, 0]) for k in range(len(seq))])
    for mode, _, _ in demo.TRACKS:
        write_trajectory(tmp_path / f"fake_{mode}.txt", seq.stamps, poses)
    frames = demo.render(seq, "fake", tmp_path, step=2)
    assert frames[0].shape == (demo.PANEL[1], 2 * demo.PANEL[0], 3)
    out = tmp_path / "demo.gif"
    demo.save_gif(frames, out, fps=10)
    assert out.stat().st_size > 0
