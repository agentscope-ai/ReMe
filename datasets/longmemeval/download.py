"""Download the LongMemEval cleaned-S dataset used by ReMe.

Source: https://huggingface.co/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2
(downloaded via the hf-mirror.com mirror for reliability).

The file ``longmemeval_s_reme_cleaned.json`` is saved under this directory using the same
name as on the remote (``benchmark/longmemeval/config.yaml`` points to it).

Usage:
    python download.py           # download cleaned-S (skip if it already exists)
"""

import os
import sys
import urllib.request

BASE_URL = "https://hf-mirror.com/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2/resolve/main"
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to download (saved with the same name as on the remote).
FILES = [
    "longmemeval_s_reme_cleaned.json",
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
    print(f"Downloading LongMemEval cleaned-S dataset to: {TARGET_DIR}\n")
    for filename in FILES:
        download_file(filename)
    print("\nAll files downloaded successfully!")
