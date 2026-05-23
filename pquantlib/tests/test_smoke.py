"""Smoke test — proves the workspace + pytest + import path are wired correctly.

This is the only test in Phase 0. Phase 1 will start replacing it with real
C++-cross-validated math primitive tests.
"""

from __future__ import annotations

import pquantlib


def test_version_is_present() -> None:
    assert pquantlib.__version__ == "1.0.0"


def test_import_does_not_raise() -> None:
    # If we got here, the import succeeded — smoke test passes.
    assert pquantlib is not None
