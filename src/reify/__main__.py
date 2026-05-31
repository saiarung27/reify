"""Module entry point so ``python -m reify ...`` mirrors the ``reify`` CLI.

Useful in CI environments where the console script may not be on PATH yet,
and for tooling that prefers the explicit module-execution form.
"""

from __future__ import annotations

from reify.cli import main


if __name__ == "__main__":
    main()
