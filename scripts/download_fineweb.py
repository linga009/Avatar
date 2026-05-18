#!/usr/bin/env python3
"""Download 1 FineWeb-Edu Parquet shard (~2GB) to data/fineweb/.

Actual repo path: sample/10BT/000_00000.parquet
Each shard has ~700K rows of educational web text (int_score 0-5).

Run on the HOST (not in Docker):
    python scripts/download_fineweb.py
"""
from __future__ import annotations
import os
import sys

TARGET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "fineweb"))
REPO_ID = "HuggingFaceFW/fineweb-edu"
# 1 shard (~2GB, ~700K rows) — enough for ~140K ticks at 5 rows/tick
ALLOW_PATTERNS = ["sample/10BT/000_00000.parquet"]


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub first:  pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Downloading FineWeb-Edu shard to {TARGET_DIR} (~2GB, may take 10-30 min)...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=TARGET_DIR,
    )

    import glob
    files = sorted(glob.glob(os.path.join(TARGET_DIR, "**/*.parquet"), recursive=True))
    print(f"\nDone. {len(files)} shard(s) in {TARGET_DIR}:")
    for f in files:
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {os.path.relpath(f, TARGET_DIR)}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
