#!/usr/bin/env python3
"""Fetch the LoCoMo benchmark dataset.

LoCoMo is a high-quality evaluation benchmark for very long-term conversational
memory, released by snap-research under ACL 2024.

Usage:
    python scripts/fetch_locomo.py

The dataset is downloaded to ~/.reify/datasets/locomo10.json.

LoCoMo repository: https://github.com/snap-research/locomo
Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024)

This script does not redistribute the dataset; it fetches it on demand.
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

# Upstream raw URL
DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data"
)
FILE = "locomo10.json"

# Default to ~/.reify/datasets for consistency with other Reify data
DEFAULT_OUT_DIR = Path.home() / ".reify" / "datasets"


def _download(url: str, dest: Path) -> None:
    print(f"  -> {url}")
    print(f"     to {dest}")
    # Create SSL context that doesn't verify certificates (for macOS compatibility)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    with urllib.request.urlopen(url, context=ssl_context) as response, dest.open("wb") as fh:
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    print(f"     ok ({dest.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch LoCoMo benchmark dataset")
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

    url = f"{args.base_url}/{FILE}"
    dest = args.out_dir / FILE

    if dest.exists():
        print(f"[skip] {dest} already exists; delete it to re-fetch.")
        return 0

    try:
        _download(url, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] failed to fetch {FILE}: {exc}", file=sys.stderr)
        return 1

    print("\nDone. Dataset can be used with:")
    print(f"  reify eval locomo --dataset {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
