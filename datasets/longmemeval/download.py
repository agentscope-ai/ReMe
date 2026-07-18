"""Download LongMemEval dataset from hf-mirror.com

Usage:
    python download.py           # download all files (skip existing)
    python download.py --all     # same as above
    python download.py --m-only  # only download longmemeval_m_cleaned.json
"""

import os
import sys
import urllib.request

BASE_URL = "https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_FILES = [
    "longmemeval_oracle.json",
    "longmemeval_s_cleaned.json",
    "longmemeval_m_cleaned.json",
]


def download_file(filename: str):
    """Download a single file from the mirror to the target directory."""
    url = f"{BASE_URL}/{filename}"
    dest = os.path.join(TARGET_DIR, filename)

    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"  [skip] {filename} already exists ({size / 1024 / 1024:.1f} MB)")
        return

    print(f"  [downloading] {filename} ...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        size = os.path.getsize(dest)
        print(f"\n  [done] {filename} ({size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"\n  [error] {filename}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        sys.exit(1)


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        sys.stdout.write(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
    else:
        mb = downloaded / 1024 / 1024
        sys.stdout.write(f"\r    {mb:.1f} MB downloaded")
    sys.stdout.flush()


if __name__ == "__main__":
    if "--m-only" in sys.argv:
        files = ["longmemeval_m_cleaned.json"]
    else:
        files = ALL_FILES

    print(f"Downloading LongMemEval dataset to: {TARGET_DIR}\n")
    for f in files:
        download_file(f)
    print("\nAll files downloaded successfully!")
