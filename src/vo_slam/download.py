"""Download TUM RGB-D sequences (resumable) and extract them into data/tum/.

Example:
    vo-download                    # all default sequences (~3.1 GB of archives)
    vo-download fr1_xyz fr1_desk   # a subset
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from .config import DEFAULT_SEQUENCES, RAW_DIR, SEQUENCES, TUM_DIR


def download_file(url: str, dest: Path, expected_bytes: int | None = None) -> Path:
    """Download with HTTP Range resume; safe to re-run after an interrupted transfer."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expected_bytes and have == expected_bytes:
        return dest
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request, timeout=60) as resp:
        resumed = resp.status == 206
        total = int(resp.headers.get("Content-Length", 0)) + (have if resumed else 0)
        with (
            dest.open("ab" if resumed else "wb") as f,
            tqdm(total=total, initial=have if resumed else 0, unit="B", unit_scale=True,
                 desc=dest.name) as bar,
        ):
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if expected_bytes and dest.stat().st_size != expected_bytes:
        raise OSError(f"{dest.name}: got {dest.stat().st_size} bytes, expected {expected_bytes}")
    return dest


def safe_member(name: str, root: str) -> bool:
    """Only plain relative paths inside the sequence folder (no '..', no absolute paths)."""
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts
            and len(path.parts) > 0 and path.parts[0] == root)


def extract(archive: Path, out_dir: Path = TUM_DIR) -> Path:
    """Extract a sequence tgz; skips work when the folder is already complete."""
    root = archive.name.removesuffix(".tgz")
    target = out_dir / root
    if (target / "groundtruth.txt").exists() and (target / "rgb.txt").exists():
        return target
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if (m.isfile() or m.isdir()) and safe_member(m.name, root)]
        for m in tqdm(members, desc=f"extract {root}", unit="file"):
            tf.extract(m, out_dir, filter="data")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequences", nargs="*", metavar="SEQ",
                        help=f"any of {', '.join(SEQUENCES)} (default: all)")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the .tgz files after extraction")
    args = parser.parse_args(argv)
    unknown = set(args.sequences) - set(SEQUENCES)
    if unknown:
        parser.error(f"unknown sequence(s): {', '.join(sorted(unknown))}")
    for short in args.sequences or DEFAULT_SEQUENCES:
        info = SEQUENCES[short]
        archive = download_file(info.url, RAW_DIR / f"{info.name}.tgz", info.size_bytes)
        folder = extract(archive)
        if not args.keep_archives:
            archive.unlink()
        print(f"{short}: {folder}")


if __name__ == "__main__":
    main()
