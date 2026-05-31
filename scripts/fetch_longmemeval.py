#!/usr/bin/env python3
"""Fetch the LongMemEval oracle dataset.

The full LongMemEval oracle (~14 MB) and a 5-case sample are not committed to
the Reify repository because they are external benchmark artifacts. This
script downloads the files into the repo root so you can run benchmark scripts
that expect them.

Usage:
    python scripts/fetch_longmemeval.py            # fetches both files
    python scripts/fetch_longmemeval.py --sample   # fetches only the sample

The Reify automated test suite does NOT depend on these files; it ships with
its own bundled fixtures via ``reify.eval.longmemeval.default_longmemeval_fixture_cases``.
The oracle files are only needed when reproducing externally-published
benchmark numbers.

LongMemEval is released by xiaowu0162/LongMemEval under the terms in their
upstream repository: https://github.com/xiaowu0162/LongMemEval

This script does not redistribute the dataset; it fetches it on demand.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Upstream raw URLs. Override via --base-url if upstream layout changes.
DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/data"
)
FILES = {
    "longmemeval_oracle.json": "longmemeval_oracle.json",
    "longmemeval_oracle.sample5.json": "longmemeval_oracle.sample5.json",
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def _download(url: str, dest: Path) -> None:
    print(f"  -> {url}")
    print(f"     to {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as fh:
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    print(f"     ok ({dest.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch LongMemEval oracle dataset")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Fetch only the 5-case sample (~174 KB); skip the full ~14 MB oracle.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Override upstream base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT,
        help="Destination directory (default: repo root).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = (
        ["longmemeval_oracle.sample5.json"]
        if args.sample
        else list(FILES.keys())
    )

    for filename in targets:
        url = f"{args.base_url}/{FILES[filename]}"
        dest = args.out_dir / filename
        if dest.exists():
            print(f"[skip] {dest} already exists; delete it to re-fetch.")
            continue
        try:
            _download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] failed to fetch {filename}: {exc}", file=sys.stderr)
            return 1

    print("\nDone. Files are gitignored and will not be committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
