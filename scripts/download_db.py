#!/usr/bin/env python3
"""Download the latest benchmark.db from Backblaze B2.

No credentials required — the bucket is public.
"""

import sys
import urllib.request
from pathlib import Path

BUCKET_NAME = "chutes-bench"
URL = f"https://f005.backblazeb2.com/file/{BUCKET_NAME}/benchmark.db"
DEST = Path("results/benchmark.db")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists():
        print(f"{DEST} already exists. Overwrite? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"Downloading from {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    size_mb = DEST.stat().st_size / 1024 / 1024
    print(f"Saved to {DEST} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
