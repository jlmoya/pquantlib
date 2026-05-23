"""Load JSON reference values produced by the C++ probe harness.

# C++ parity: none — this is harness, not a port.

Probe outputs live at::

    <repo>/migration-harness/references/<topic>/<class>.json

Resolution: search upward from the caller's location to find the
``migration-harness/references/`` directory. This makes the loader
work both in the main worktree and in cluster worktrees, without
hard-coding an absolute path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HARNESS_REL: Path = Path("migration-harness") / "references"


def _find_references_root(start: Path) -> Path:
    cur = start.resolve()
    for ancestor in (cur, *cur.parents):
        candidate = ancestor / _HARNESS_REL
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find {_HARNESS_REL} starting from {start}")


def load(key: str, *, start: Path | None = None) -> dict[str, Any]:
    """Load ``migration-harness/references/<key>.json`` as a dict.

    ``key`` is the topic/class path without extension, e.g. ``"math/beta"``.
    ``start`` defaults to this file's directory; tests can override to
    pin the search root for isolation.
    """
    root = _find_references_root(start if start is not None else Path(__file__).parent)
    path = root / f"{key}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]
