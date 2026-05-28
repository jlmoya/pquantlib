"""Tests for the inflation Region IntEnum.

Cross-validates region name + code against
``migration-harness/references/l7a/foundations.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.indexes.inflation.region import Region
from pquantlib.testing.reference_reader import load as load_reference


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return load_reference("l7a/foundations")


def test_region_members_match_cpp_names_and_codes(reference: dict[str, Any]) -> None:
    """Every Region enum member round-trips the C++ Region::name / code strings."""
    # Probe carries one (name, code) pair per index; pivot by region:
    by_region: dict[Region, tuple[str, str]] = {}
    for key, payload in reference["zero_indexes"].items():
        rname = payload["region_name"]
        rcode = payload["region_code"]
        if rname == "EU":
            r = Region.Europe
        elif rname == "France":
            r = Region.France
        elif rname == "UK":
            r = Region.UnitedKingdom
        elif rname == "USA":
            r = Region.UnitedStates
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected region name in probe: {rname} for {key}")
        by_region.setdefault(r, (rname, rcode))

    assert by_region[Region.Europe] == ("EU", "EU")
    assert by_region[Region.France] == ("France", "FR")
    assert by_region[Region.UnitedKingdom] == ("UK", "UK")
    assert by_region[Region.UnitedStates] == ("USA", "US")

    # Round-trip via the enum's accessors.
    for r, (expected_name, expected_code) in by_region.items():
        assert r.region_name() == expected_name
        assert r.region_code() == expected_code


def test_region_is_ordered_intenum() -> None:
    """Region is an IntEnum, so members are ordered and hashable."""
    assert int(Region.Europe) == 1
    assert int(Region.UnitedStates) == 4
    assert sorted(Region) == [
        Region.Europe,
        Region.France,
        Region.UnitedKingdom,
        Region.UnitedStates,
    ]
    assert {Region.Europe, Region.Europe} == {Region.Europe}
