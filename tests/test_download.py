import io
import tarfile

from vo_slam.download import extract, safe_member


def test_safe_member_rejects_escapes():
    root = "rgbd_dataset_freiburg1_xyz"
    assert safe_member(f"{root}/rgb/1.png", root)
    assert not safe_member(f"{root}/../evil.txt", root)
    assert not safe_member("/etc/passwd", root)
    assert not safe_member("other_folder/file.txt", root)


def _add(tf, name, data=b"x"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def test_extract_only_writes_inside_the_sequence_folder(tmp_path):
    root = "rgbd_dataset_freiburg1_fake"
    archive = tmp_path / f"{root}.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt", "rgb/1.png"):
            _add(tf, f"{root}/{name}")
        _add(tf, f"{root}/../escaped.txt")
        _add(tf, "stray.txt")
    out = tmp_path / "out"
    folder = extract(archive, out)
    assert folder == out / root and (folder / "rgb" / "1.png").exists()
    assert not (tmp_path / "escaped.txt").exists() and not (out / "stray.txt").exists()
    # second call is a no-op once the folder is complete
    (folder / "rgb" / "1.png").unlink()
    extract(archive, out)
    assert not (folder / "rgb" / "1.png").exists()
