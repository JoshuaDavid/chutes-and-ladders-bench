#!/usr/bin/env python3
"""Export benchmark.db to JSON and upload to Backblaze B2.

Reads results/benchmark.db (downloading it first if missing), generates
games.json + per-game event files, and uploads them to B2.

Requires env vars:
    BACKBLAZE_KEY_ID
    BACKBLAZE_APPLICATION_KEY
    BACKBLAZE_CHUTES_BUCKET_NAME
    BACKBLAZE_CHUTES_HOST          (S3-compatible endpoint hostname)
"""

import os
import sys
import tempfile
from pathlib import Path

DB_PATH = Path("results/benchmark.db")

REQUIRED_VARS = (
    "BACKBLAZE_KEY_ID",
    "BACKBLAZE_APPLICATION_KEY",
    "BACKBLAZE_CHUTES_BUCKET_NAME",
    "BACKBLAZE_CHUTES_HOST",
)


def download_db() -> None:
    """Download benchmark.db from the public B2 bucket."""
    import urllib.request

    bucket = os.environ.get("BACKBLAZE_CHUTES_BUCKET_NAME", "chutes-bench")
    url = f"https://f005.backblazeb2.com/file/{bucket}/benchmark.db"
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, DB_PATH)
    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"Saved to {DB_PATH} ({size_mb:.1f} MB)")


def main() -> None:
    for var in REQUIRED_VARS:
        if not os.environ.get(var):
            print(f"Missing env var: {var}", file=sys.stderr)
            sys.exit(1)

    if not DB_PATH.exists():
        download_db()

    from chutes_bench.export import generate_all, upload_to_b2

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        generated = generate_all(DB_PATH, out_dir)
        print(f"Generated {len(generated)} JSON files")

        # Build upload mapping: B2 object key -> file bytes
        files: dict[str, bytes] = {}
        for path in generated:
            key = "data/" + str(path.relative_to(out_dir))
            files[key] = path.read_bytes()

        host = os.environ["BACKBLAZE_CHUTES_HOST"]
        upload_to_b2(
            files=files,
            bucket_name=os.environ["BACKBLAZE_CHUTES_BUCKET_NAME"],
            endpoint_url=f"https://{host}",
            key_id=os.environ["BACKBLAZE_KEY_ID"],
            app_key=os.environ["BACKBLAZE_APPLICATION_KEY"],
        )
        print(f"Uploaded {len(files)} files to B2")


if __name__ == "__main__":
    main()
