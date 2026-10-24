from vo_slam.config import CAMERAS
from vo_slam.dataset_stats import plot_trajectories, sequence_stats, to_markdown
from vo_slam.tum import TumSequence


def test_stats_on_synthetic_sequence(tum_folder, tmp_path):
    seq = TumSequence(tum_folder, CAMERAS["fr1"])
    s = sequence_stats(seq)
    assert s["frames"] == 6 and s["frames_with_gt"] == 4
    assert s["fps"] == 30.0
    assert s["total_rotation_deg"] == 0
    assert s["depth_valid_fraction"] == round(1 - 64 / (48 * 64), 3)
    md = to_markdown({"fake": s})
    assert "| fake | 6 | 4 |" in md
    plot_trajectories({"fake": seq}, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0
