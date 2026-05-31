"""Cross-validate Quantity arithmetic (same-UOM + cross-UOM conversion).

Probe source: migration-harness/cpp/probes/cluster_w7b/probe.cpp
Reference:    migration-harness/references/cluster/w7b.json
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.commodities.commodity_type import NullCommodityType
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
    LitreUnitOfMeasure,
)
from pquantlib.experimental.commodities.quantity import (
    Quantity,
    close,
    close_enough,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w7b")


@pytest.fixture(autouse=True)
def _reset_conversion_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Restore the process-global Quantity conversion settings after each test."""
    saved_type = Quantity.conversion_type
    saved_base = Quantity.base_unit_of_measure
    yield
    Quantity.conversion_type = saved_type
    Quantity.base_unit_of_measure = saved_base


def _bbl(amount: float) -> Quantity:
    return Quantity(NullCommodityType(), BarrelUnitOfMeasure(), amount)


# ---- same-UOM arithmetic ----


def test_same_uom_add(cpp_ref: dict[str, Any]) -> None:
    tolerance.tight((_bbl(1.0) + _bbl(1.0)).amount, cpp_ref["qty_1bbl_plus_1bbl"])


def test_same_uom_sub(cpp_ref: dict[str, Any]) -> None:
    tolerance.tight((_bbl(5.0) - _bbl(2.0)).amount, cpp_ref["qty_5bbl_minus_2bbl"])


def test_scalar_mul(cpp_ref: dict[str, Any]) -> None:
    tolerance.tight((_bbl(3.0) * 4.0).amount, cpp_ref["qty_3bbl_times_4"])
    # rmul commutes
    tolerance.tight((4.0 * _bbl(3.0)).amount, cpp_ref["qty_3bbl_times_4"])


def test_quantity_ratio(cpp_ref: dict[str, Any]) -> None:
    ratio = _bbl(6.0) / _bbl(2.0)
    assert isinstance(ratio, float)
    tolerance.tight(ratio, cpp_ref["qty_6bbl_over_2bbl"])


def test_scalar_div(cpp_ref: dict[str, Any]) -> None:
    divided = _bbl(9.0) / 3.0
    assert isinstance(divided, Quantity)
    tolerance.tight(divided.amount, cpp_ref["qty_9bbl_over_3"])


def test_negate(cpp_ref: dict[str, Any]) -> None:
    tolerance.tight((-_bbl(7.0)).amount, cpp_ref["qty_negate_7bbl"])
    # unary plus is a copy
    assert (+_bbl(7.0)).amount == 7.0


def test_inspectors() -> None:
    q = _bbl(2.5)
    assert q.commodity_type == NullCommodityType()
    assert q.unit_of_measure == BarrelUnitOfMeasure()
    assert q.amount == 2.5


# ---- cross-UOM arithmetic under AutomatedConversion ----


def test_cross_uom_add_automated(cpp_ref: dict[str, Any]) -> None:
    Quantity.conversion_type = Quantity.ConversionType.AUTOMATED_CONVERSION
    nct = NullCommodityType()
    total = Quantity(nct, BarrelUnitOfMeasure(), 1.0) + Quantity(
        nct, LitreUnitOfMeasure(), 158.987
    )
    tolerance.tight(total.amount, cpp_ref["qty_1bbl_plus_158_987litre_amount"])
    assert total.unit_of_measure.code == cpp_ref["qty_1bbl_plus_158_987litre_uom"]


def test_cross_uom_no_conversion_raises() -> None:
    # Default conversion_type is NO_CONVERSION -> mismatched UOM add fails.
    assert Quantity.conversion_type == Quantity.ConversionType.NO_CONVERSION
    nct = NullCommodityType()
    with pytest.raises(LibraryException):
        _ = Quantity(nct, BarrelUnitOfMeasure(), 1.0) + Quantity(
            nct, LitreUnitOfMeasure(), 1.0
        )


def test_cross_uom_base_conversion() -> None:
    Quantity.conversion_type = Quantity.ConversionType.BASE_UNIT_OF_MEASURE_CONVERSION
    Quantity.base_unit_of_measure = BarrelUnitOfMeasure()
    nct = NullCommodityType()
    # 1 barrel + 158.987 litres, both routed to base (barrels) -> 2 barrels.
    total = Quantity(nct, LitreUnitOfMeasure(), 158.987) + Quantity(
        nct, BarrelUnitOfMeasure(), 1.0
    )
    assert total.unit_of_measure == BarrelUnitOfMeasure()
    tolerance.loose(total.amount, 2.0)


def test_base_conversion_without_base_raises() -> None:
    Quantity.conversion_type = Quantity.ConversionType.BASE_UNIT_OF_MEASURE_CONVERSION
    Quantity.base_unit_of_measure = Quantity.base_unit_of_measure.__class__()
    nct = NullCommodityType()
    with pytest.raises(LibraryException):
        _ = Quantity(nct, BarrelUnitOfMeasure(), 1.0) + Quantity(
            nct, LitreUnitOfMeasure(), 1.0
        )


# ---- comparisons + closeness ----


def test_comparisons_same_uom() -> None:
    assert _bbl(1.0) < _bbl(2.0)
    assert _bbl(2.0) > _bbl(1.0)
    assert _bbl(1.0) <= _bbl(1.0)
    assert _bbl(2.0) >= _bbl(2.0)
    assert _bbl(1.0) == _bbl(1.0)
    assert _bbl(1.0) != _bbl(2.0)


def test_close_helpers() -> None:
    assert close(_bbl(1.0), _bbl(1.0))
    assert close_enough(_bbl(1.0), _bbl(1.0))
    assert not close(_bbl(1.0), _bbl(2.0))


def test_rounded_is_noop_by_default() -> None:
    # Barrel's default rounding is the no-op Rounding(), so amount is unchanged.
    q = _bbl(1.23456789)
    tolerance.exact(q.rounded().amount, 1.23456789)
