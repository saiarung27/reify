#!/usr/bin/env python3
"""Fetch the STATE-Bench benchmark dataset.

STATE-Bench is a benchmark for AI agent memory released by Microsoft.
It evaluates agents on realistic, multi-step enterprise workflows across
three domains: travel, customer support, and shopping assistant.

Usage:
    python scripts/fetch_statebench.py

The dataset is downloaded to ~/.reify/datasets/statebench/.

STATE-Bench repository: https://github.com/microsoft/STATE-Bench
Blog: https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/

Note: STATE-Bench is an agent framework benchmark with a "bring your own memory"
interface. Integration requires adapting Reify to the STATE-Bench memory API.
This script only downloads the dataset; the integration work is separate.

This script does not redistribute the dataset; it fetches it on demand.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# STATE-Bench dataset files from GitHub
DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/microsoft/STATE-Bench/main/datasets"
)
FILES = [
    "train_task_trajectories",
]

# Default to ~/.reify/datasets for consistency with other Reify data
DEFAULT_OUT_DIR = Path.home() / ".reify" / "datasets" / "statebench"


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
    parser = argparse.ArgumentParser(description="Fetch STATE-Bench benchmark dataset")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Override upstream base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        url = f"{args.base_url}/{filename}"
        dest = args.out_dir / filename

        if dest.exists():
            print(f"[skip] {dest} already exists; delete it to re-fetch.")
            continue

        try:
            _download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] failed to fetch {filename}: {exc}", file=sys.stderr)
            return 1

    print("\nDone. STATE-Bench integration requires adapter work.")
    print("See marketing/benchmark-implementation-plan.md for details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
