#!/usr/bin/env python3
"""BEAM benchmark dataset information.

BEAM (Beyond a Million Tokens) is a large-scale benchmark for testing memory
systems at scale, released under ICLR 2026.

The BEAM dataset requires LLM-judged evaluation with complex question types:
- Abstention questions (test if model knows when to say "I don't know")
- Contradiction resolution (test if model detects contradictions)
- Event ordering (test temporal reasoning)

This is fundamentally different from Reify's retrieval proxy approach, which
uses simple substring matching. BEAM requires integrating with their LLM-based
evaluation pipeline.

To generate the BEAM dataset:
1. Clone the repository: git clone https://github.com/mohammadtavakoli78/BEAM
2. Run the download script: python src/beam/download_dataset.py

For now, use the bundled fixture in Reify:
  reify eval beam

BEAM repository: https://github.com/mohammadtavakoli78/BEAM
Paper: "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs" (ICLR 2026)
"""

from __future__ import annotations

import sys


def main() -> int:
    print(__doc__)
    print("\nBEAM Limitation:")
    print("BEAM uses LLM-judged evaluation with complex question types that")
    print("are incompatible with Reify's retrieval proxy approach. Full")
    print("integration requires implementing BEAM's evaluation pipeline.")
    print("\nFor quick testing, use the bundled fixture:")
    print("  reify eval beam")
    return 0


if __name__ == "__main__":
    sys.exit(main())
