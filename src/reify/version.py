"""Single source of truth for the package version.

Kept separate from ``__init__`` so build tooling can read it without importing
the rest of the package.
"""

__version__ = "0.1.0.dev0"
