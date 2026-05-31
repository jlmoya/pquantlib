"""Cross-validate UnitOfMeasure + UOM conversions + the conversion manager.

Probe source: migration-harness/cpp/probes/cluster_w7b/probe.cpp
Reference:    migration-harness/references/cluster/w7b.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.commodities.commodity_type import NullCommodityType
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
    GallonUnitOfMeasure,
    KilolitreUnitOfMeasure,
    LitreUnitOfMeasure,
    MBUnitOfMeasure,
    MTUnitOfMeasure,
)
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.experimental.commodities.unit_of_measure import (
    UnitOfMeasure,
    UnitType,
)
from pquantlib.experimental.commodities.unit_of_measure_conversion import (
    UnitOfMeasureConversion,
)
from pquantlib.experimental.commodities.unit_of_measure_conversion_manager import (
    UnitOfMeasureConversionManager,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w7b")


# ---- UnitOfMeasure basics ----


def test_uom_inspectors(cpp_ref: dict[str, Any]) -> None:
    bbl = BarrelUnitOfMeasure()
    assert bbl.code == cpp_ref["uom_bbl_code"]
    assert bbl.name == cpp_ref["uom_bbl_name"]
    assert bbl.unit_type == UnitType.VOLUME
    assert not bbl.empty()


def test_uom_default_is_empty() -> None:
    u = UnitOfMeasure()
    assert u.empty()
    assert str(u) == "null unit of measure"


def test_uom_equality_is_code_based() -> None:
    assert BarrelUnitOfMeasure() == BarrelUnitOfMeasure()
    assert BarrelUnitOfMeasure() != LitreUnitOfMeasure()


def test_uom_flyweight_shares_data() -> None:
    # Two instances built from the same name share the same _Data payload
    # (parity with the C++ static unitsOfMeasure_ map).
    a = BarrelUnitOfMeasure()
    b = BarrelUnitOfMeasure()
    # White-box probe of the shared flyweight payload.
    assert a._data is b._data  # pyright: ignore[reportPrivateUsage]


def test_uom_triangulation(cpp_ref: dict[str, Any]) -> None:
    litre = LitreUnitOfMeasure()
    assert litre.triangulation_unit_of_measure.code == (
        cpp_ref["uom_litre_triangulation_code"]
    )


# ---- known conversion factors via the manager ----


def test_known_conversion_factors(cpp_ref: dict[str, Any]) -> None:
    mgr = UnitOfMeasureConversionManager.instance()
    nct = NullCommodityType()

    c_bl = mgr.lookup(
        nct, BarrelUnitOfMeasure(), LitreUnitOfMeasure(),
        UnitOfMeasureConversion.Type.DIRECT,
    )
    tolerance.tight(c_bl.conversion_factor, cpp_ref["conv_bbl_litre_factor"])

    c_bg = mgr.lookup(
        nct, BarrelUnitOfMeasure(), GallonUnitOfMeasure(),
        UnitOfMeasureConversion.Type.DIRECT,
    )
    tolerance.tight(c_bg.conversion_factor, cpp_ref["conv_bbl_gallon_factor"])

    c_mb = mgr.lookup(
        nct, MBUnitOfMeasure(), BarrelUnitOfMeasure(),
        UnitOfMeasureConversion.Type.DIRECT,
    )
    tolerance.tight(c_mb.conversion_factor, cpp_ref["conv_mb_bbl_factor"])

    c_kl = mgr.lookup(
        nct, KilolitreUnitOfMeasure(), BarrelUnitOfMeasure(),
        UnitOfMeasureConversion.Type.DIRECT,
    )
    tolerance.tight(c_kl.conversion_factor, cpp_ref["conv_kilolitre_bbl_factor"])


def test_conversion_code_string(cpp_ref: dict[str, Any]) -> None:
    mgr = UnitOfMeasureConversionManager.instance()
    c = mgr.lookup(
        NullCommodityType(), BarrelUnitOfMeasure(), LitreUnitOfMeasure(),
        UnitOfMeasureConversion.Type.DIRECT,
    )
    assert c.code == cpp_ref["conv_bbl_litre_code"]


# ---- UnitOfMeasureConversion.convert: forward + inverse ----


def test_convert_forward_and_inverse(cpp_ref: dict[str, Any]) -> None:
    mgr = UnitOfMeasureConversionManager.instance()
    nct = NullCommodityType()
    c = mgr.lookup(
        nct, BarrelUnitOfMeasure(), LitreUnitOfMeasure(),
        UnitOfMeasureConversion.Type.DIRECT,
    )

    one_barrel = Quantity(nct, BarrelUnitOfMeasure(), 1.0)
    in_litres = c.convert(one_barrel)
    tolerance.tight(in_litres.amount, cpp_ref["convert_1bbl_to_litre_amount"])
    assert in_litres.unit_of_measure.code == cpp_ref["convert_1bbl_to_litre_uom"]

    some_litres = Quantity(nct, LitreUnitOfMeasure(), 158.987)
    in_barrels = c.convert(some_litres)
    tolerance.tight(in_barrels.amount, cpp_ref["convert_litre_to_bbl_amount"])
    assert in_barrels.unit_of_measure.code == cpp_ref["convert_litre_to_bbl_uom"]


def test_convert_not_applicable_raises() -> None:
    mgr = UnitOfMeasureConversionManager.instance()
    c = mgr.lookup(
        NullCommodityType(), BarrelUnitOfMeasure(), LitreUnitOfMeasure(),
        UnitOfMeasureConversion.Type.DIRECT,
    )
    # A gallon quantity doesn't match this barrel<->litre direct conversion.
    q = Quantity(NullCommodityType(), GallonUnitOfMeasure(), 1.0)
    with pytest.raises(LibraryException):
        c.convert(q)


# ---- manager triangulated lookup (Derived) round-trips through Barrel ----


def test_triangulated_lookup_litre_to_gallon() -> None:
    # Neither litre nor gallon is the other's direct pair, but both
    # triangulate through Barrel, so the default (Derived) lookup chains.
    mgr = UnitOfMeasureConversionManager.instance()
    nct = NullCommodityType()
    conv = mgr.lookup(nct, LitreUnitOfMeasure(), GallonUnitOfMeasure())
    one_litre = Quantity(nct, LitreUnitOfMeasure(), 1.0)
    in_gallons = conv.convert(one_litre)
    # 1 litre -> barrels -> gallons: (1/158.987) * 42 gallons.
    expected = (1.0 / 158.987) * 42.0
    tolerance.loose(in_gallons.amount, expected)


def test_direct_lookup_missing_raises() -> None:
    mgr = UnitOfMeasureConversionManager.instance()
    # Metric tonnes has no registered conversion -> direct lookup fails.
    with pytest.raises(LibraryException):
        mgr.lookup(
            NullCommodityType(), MTUnitOfMeasure(), BarrelUnitOfMeasure(),
            UnitOfMeasureConversion.Type.DIRECT,
        )
