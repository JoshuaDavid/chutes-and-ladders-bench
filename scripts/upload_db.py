#!/usr/bin/env python3
"""Upload results/benchmark.db to Backblaze B2.

Requires env vars:
    BACKBLAZE_KEY_ID
    BACKBLAZE_APPLICATION_KEY
    BACKBLAZE_CHUTES_BUCKET_NAME
"""

import os
import sys
from pathlib import Path

DB_PATH = Path("results/benchmark.db")


def main() -> None:
    if not DB_PATH.exists():
        print(f"No database at {DB_PATH}. Run some games first.", file=sys.stderr)
        sys.exit(1)

    for var in ("BACKBLAZE_KEY_ID", "BACKBLAZE_APPLICATION_KEY", "BACKBLAZE_CHUTES_BUCKET_NAME"):
        if not os.environ.get(var):
            print(f"Missing env var: {var}", file=sys.stderr)
            sys.exit(1)

    from b2sdk.v2 import InMemoryAccountInfo, B2Api

    info = InMemoryAccountInfo()
    b2 = B2Api(info)
    b2.authorize_account(
        "production",
        os.environ["BACKBLAZE_KEY_ID"],
        os.environ["BACKBLAZE_APPLICATION_KEY"],
    )

    bucket = b2.get_bucket_by_name(os.environ["BACKBLAZE_CHUTES_BUCKET_NAME"])
    result = bucket.upload_local_file(
        local_file=str(DB_PATH),
        file_name="benchmark.db",
    )
    size_mb = result.size / 1024 / 1024
    print(f"Uploaded benchmark.db ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
