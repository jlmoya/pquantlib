"""Tests for pquantlib.testing.reference_reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pquantlib.testing import reference_reader, tolerance

# --- happy path: loads the Stage 0 sentinel emitted by the C++ harness ------


def test_load_returns_dict() -> None:
    data = reference_reader.load("harness/sentinel")
    assert isinstance(data, dict)


def test_load_carries_quantlib_version() -> None:
    data = reference_reader.load("harness/sentinel")
    assert data["quantlib_version"] == "1.42.1"


def test_load_carries_sqrt_two_at_full_precision() -> None:
    data = reference_reader.load("harness/sentinel")
    tolerance.exact(data["sqrt_two"], 1.4142135623730951)


def test_load_carries_pi_at_full_precision() -> None:
    data = reference_reader.load("harness/sentinel")
    tolerance.tight(data["pi"], 3.141592653589793)


# --- error paths ------------------------------------------------------------


def test_load_missing_key_raises_filenotfound() -> None:
    with pytest.raises(FileNotFoundError):
        reference_reader.load("does/not/exist")


def test_load_with_explicit_start_walks_up_to_find_references_root(tmp_path: Path) -> None:
    # Create a fake nested directory that lives under the real repo. The loader
    # must walk up until it finds migration-harness/references.
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        reference_reader.load("harness/sentinel", start=nested)


def test_load_raw_json_matches_disk(tmp_path: Path) -> None:
    """Loader must return the same dict json.load would, no munging."""
    data = reference_reader.load("harness/sentinel")
    # Locate the file via the same upward-walk logic and compare.
    cur = Path(__file__).resolve()
    for ancestor in (cur, *cur.parents):
        candidate = ancestor / "migration-harness" / "references" / "harness" / "sentinel.json"
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            assert data == raw
            return
    pytest.fail("could not locate sentinel.json from test file location")
