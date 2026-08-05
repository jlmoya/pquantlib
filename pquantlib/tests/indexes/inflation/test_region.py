"""Tests for the inflation ``Region`` hierarchy.

The four regions the L7-A probe exercises are cross-validated against
``migration-harness/references/l7a/foundations.json``; the three the probe
does not reach (Australia, South Africa, Generic) are cross-validated by the
v1.43 region probe in ``test_v143_indexes_region.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.indexes.inflation.region import (
    AustraliaRegion,
    CustomRegion,
    EURegion,
    FranceRegion,
    GenericRegion,
    Region,
    UKRegion,
    USRegion,
    ZARegion,
)
from pquantlib.testing.reference_reader import load as load_reference

_BY_NAME: dict[str, Region] = {
    "EU": EURegion(),
    "France": FranceRegion(),
    "UK": UKRegion(),
    "USA": USRegion(),
}


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return load_reference("l7a/foundations")


def test_region_concretes_match_cpp_names_and_codes(reference: dict[str, Any]) -> None:
    """Every region the probe reaches round-trips the C++ name / code strings."""
    seen: set[str] = set()
    for key, payload in reference["zero_indexes"].items():
        rname = payload["region_name"]
        rcode = payload["region_code"]
        assert rname in _BY_NAME, f"unexpected region name in probe: {rname} for {key}"
        region = _BY_NAME[rname]
        assert region.name() == rname
        assert region.code() == rcode
        seen.add(rname)
    assert seen == set(_BY_NAME)


def test_equality_is_by_name_only() -> None:
    """C++ ``operator==`` compares ``name()`` and ignores ``code()``."""
    assert EURegion() == EURegion()
    assert EURegion() != USRegion()
    # A CustomRegion with a matching name compares equal even with a different
    # code — that is exactly what the C++ comparison does.
    assert CustomRegion("EU", "XX") == EURegion()
    assert CustomRegion("EU", "XX").code() == "XX"
    assert hash(CustomRegion("EU", "XX")) == hash(EURegion())


def test_regions_are_hashable_and_usable_as_keys() -> None:
    registry = {EURegion(): "euro", USRegion(): "dollar"}
    assert registry[EURegion()] == "euro"
    assert len({EURegion(), EURegion(), USRegion()}) == 2


def test_custom_region_carries_arbitrary_payload() -> None:
    r = CustomRegion("Atlantis", "AT")
    assert r.name() == "Atlantis"
    assert r.code() == "AT"
    assert isinstance(r, Region)


@pytest.mark.parametrize(
    ("region", "expected_name", "expected_code"),
    [
        (AustraliaRegion(), "Australia", "AU"),
        (EURegion(), "EU", "EU"),
        (FranceRegion(), "France", "FR"),
        (UKRegion(), "UK", "UK"),
        (USRegion(), "USA", "US"),
        (ZARegion(), "South Africa", "ZA"),
        (GenericRegion(), "Generic", "GENERIC"),
    ],
)
def test_region_payloads(region: Region, expected_name: str, expected_code: str) -> None:
    assert region.name() == expected_name
    assert region.code() == expected_code
