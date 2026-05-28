"""Tests for Uniform1dMesher (1-D equispaced grid).

# C++ parity: ql/methods/finitedifferences/meshers/uniform1dmesher.hpp
# @ v1.42.1.

Cross-validates the 1-D uniform mesh against ``uniform_1d_mesher``
section of ``migration-harness/references/cluster/l5d.json``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l5d")


def test_size_matches(reference_data: dict[str, Any]) -> None:
    m = Uniform1dMesher(-2.0, 2.0, 11)
    assert m.size() == reference_data["uniform_1d_mesher"]["size"]


def test_locations_match_reference(reference_data: dict[str, Any]) -> None:
    m = Uniform1dMesher(-2.0, 2.0, 11)
    locs = m.locations()
    expected = reference_data["uniform_1d_mesher"]["locations"]
    assert len(locs) == len(expected)
    # TIGHT (not EXACT): C++ at -O3 fuses ``start + i*dx`` into an
    # FMA (one rounding); Python does two roundings. That produces a
    # 1-ULP difference at some indices (e.g. idx=3 -> bfe...998 vs
    # bfe...999). All other arithmetic is identical; TIGHT tier is
    # sufficient and matches the C++/Python parity for this stub.
    for actual_v, expected_v in zip(locs, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_dplus_at_5(reference_data: dict[str, Any]) -> None:
    m = Uniform1dMesher(-2.0, 2.0, 11)
    # TIGHT: dx = (end-start)/(size-1) = 0.4; reuses the same IEEE-754
    # division in both ports, so the ULP-level pattern matches.
    tight(m.dplus(5), float(reference_data["uniform_1d_mesher"]["dplus_at_5"]))


def test_dminus_at_5(reference_data: dict[str, Any]) -> None:
    m = Uniform1dMesher(-2.0, 2.0, 11)
    tight(m.dminus(5), float(reference_data["uniform_1d_mesher"]["dminus_at_5"]))


def test_dplus_at_last_is_nan() -> None:
    m = Uniform1dMesher(-2.0, 2.0, 11)
    assert math.isnan(m.dplus(10))


def test_dminus_at_first_is_nan() -> None:
    m = Uniform1dMesher(-2.0, 2.0, 11)
    assert math.isnan(m.dminus(0))


def test_invalid_range_raises() -> None:
    with pytest.raises(LibraryException):
        Uniform1dMesher(2.0, -2.0, 11)


def test_last_location_is_exactly_end() -> None:
    """The C++ Uniform1dMesher assigns ``locations_.back() = end`` after
    the loop — TIGHT-tier confirms the exact endpoint.
    """
    m = Uniform1dMesher(-2.0, 2.0, 11)
    tight(m.location(10), 2.0)
