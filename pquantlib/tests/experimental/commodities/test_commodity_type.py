"""Cross-validate CommodityType flyweight + inspectors.

Probe source: migration-harness/cpp/probes/cluster_w7b/probe.cpp
Reference:    migration-harness/references/cluster/w7b.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.commodities.commodity_type import (
    CommodityType,
    NullCommodityType,
)
from pquantlib.testing import reference_reader


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w7b")


def test_commodity_type_inspectors(cpp_ref: dict[str, Any]) -> None:
    # C++ parity quirk: first positional becomes `name`, second becomes `code`
    # (see commodity_type module docstring). The probe confirms this:
    # CommodityType("HO", "Heating Oil") -> name=="HO", code=="Heating Oil".
    ho = CommodityType("HO", "Heating Oil")
    assert ho.name == cpp_ref["commodity_ho_name"]
    assert ho.code == cpp_ref["commodity_ho_code"]


def test_null_commodity_type(cpp_ref: dict[str, Any]) -> None:
    nct = NullCommodityType()
    assert nct.code == cpp_ref["null_commodity_code"]
    assert nct.name == "<NULL>"
    assert not nct.empty()


def test_default_is_empty() -> None:
    c = CommodityType()
    assert c.empty()
    assert str(c) == "null commodity type"


def test_flyweight_identity() -> None:
    # Same code -> same shared _Data payload (parity with the C++ static
    # commodityTypes_ map keyed on code). EXACT-tier identity check.
    a = CommodityType("HO", "Heating Oil")
    b = CommodityType("HO", "Heating Oil")
    # White-box probe of the shared flyweight payload.
    assert a._data is b._data  # pyright: ignore[reportPrivateUsage]


def test_equality_is_code_based() -> None:
    a = CommodityType("WTI", "WTI Crude")
    b = CommodityType("WTI", "WTI Crude")
    assert a == b
    assert a != NullCommodityType()
    assert CommodityType() == CommodityType()
