"""Smoke test — proves the workspace + pytest + import path are wired correctly.

This is the only test in Phase 0. Phase 1 will start replacing it with real
C++-cross-validated math primitive tests.
"""

from __future__ import annotations

from importlib.metadata import version

import pquantlib


def test_version_matches_package_metadata() -> None:
    # __version__ is sourced dynamically from the installed package metadata,
    # so it always tracks pyproject's version with no manual bump to drift.
    assert pquantlib.__version__ == version("pquantlib")
    assert pquantlib.__version__ not in ("", "0.0.0+unknown")


def test_import_does_not_raise() -> None:
    # If we got here, the import succeeded — smoke test passes.
    assert pquantlib is not None
