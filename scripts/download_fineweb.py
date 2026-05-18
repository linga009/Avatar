#!/usr/bin/env python3
"""Download 3 FineWeb-Edu Parquet shards to data/fineweb/.

Run on the HOST (not in Docker):
    python scripts/download_fineweb.py
"""
from __future__ import annotations
import os
import sys

TARGET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "fineweb"))
REPO_ID = "HuggingFaceFW/fineweb-edu"
ALLOW_PATTERNS = [
    "sample-10BT/data/train-00000-of-00096.parquet",
    "sample-10BT/data/train-00001-of-00096.parquet",
    "sample-10BT/data/train-00002-of-00096.parquet",
]


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub first:  pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Downloading 3 FineWeb-Edu shards to {TARGET_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
    )

    import glob
    files = sorted(glob.glob(os.path.join(TARGET_DIR, "**/*.parquet"), recursive=True))
    print(f"\nDone. {len(files)} shard(s):")
    for f in files:
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {os.path.relpath(f, TARGET_DIR)}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
