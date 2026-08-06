"""Cross-validation of the inner-value calculator family against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdminnervaluecalculator.{hpp,cpp}
# @ v1.43 (6b57206e0).

Covers ``FdmInnerValueCalculator`` (through its concretes),
``FdmCellAveragingInnerValue``, ``FdmLogInnerValue``,
``FdmLogBasketInnerValue`` and ``FdmZeroInnerValue``.
"""

from __future__ import annotations

import math
from typing import Any

from pquantlib.instruments.basket_option import MaxBasketPayoff, MinBasketPayoff
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmCellAveragingInnerValue,
    FdmLogBasketInnerValue,
    FdmLogInnerValue,
    FdmZeroInnerValue,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.testing.tolerance import tight

from .conftest import as_floats


def _uniform_spot_mesher() -> FdmMesherComposite:
    """Linear (non-log) 11-node spot mesh on ``[50, 150]``."""
    return FdmMesherComposite(Uniform1dMesher(50.0, 150.0, 11))


def test_log_inner_value_put(
    reference_data: dict[str, Any], log_spot_mesher: FdmMesherComposite
) -> None:
    calc = FdmLogInnerValue(PlainVanillaPayoff(OptionType.Put, 100.0), log_spot_mesher, 0)
    actual = [calc.inner_value(it, 0.0) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["log_inner_value_put"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_log_avg_inner_value_put(
    reference_data: dict[str, Any], log_spot_mesher: FdmMesherComposite
) -> None:
    """Cell average via ``SimpsonIntegral(acc, 8)`` on each interior cell.

    TIGHT holds (worst measured relative deviation 2.0e-14) — the iterative
    Romberg refinement runs the identical sequence of operations in both
    ports and stops on the same iteration.
    """
    calc = FdmLogInnerValue(PlainVanillaPayoff(OptionType.Put, 100.0), log_spot_mesher, 0)
    actual = [calc.avg_inner_value(it, 0.0) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["log_avg_inner_value_put"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_log_avg_inner_value_call(
    reference_data: dict[str, Any], log_spot_mesher: FdmMesherComposite
) -> None:
    calc = FdmLogInnerValue(PlainVanillaPayoff(OptionType.Call, 100.0), log_spot_mesher, 0)
    actual = [calc.avg_inner_value(it, 0.0) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["log_avg_inner_value_call"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_cell_averaging_identity_mapping_inner(reference_data: dict[str, Any]) -> None:
    """Default ``gridMapping`` is the identity, so the mesh holds spots directly."""
    mesher = _uniform_spot_mesher()
    calc = FdmCellAveragingInnerValue(PlainVanillaPayoff(OptionType.Call, 100.0), mesher, 0)
    actual = [calc.inner_value(it, 0.0) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["cell_avg_inner_value_identity_inner"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_cell_averaging_identity_mapping_avg(reference_data: dict[str, Any]) -> None:
    mesher = _uniform_spot_mesher()
    calc = FdmCellAveragingInnerValue(PlainVanillaPayoff(OptionType.Call, 100.0), mesher, 0)
    actual = [calc.avg_inner_value(it, 0.0) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["cell_avg_inner_value_identity_avg"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_cell_averaging_2d_direction_1(reference_data: dict[str, Any]) -> None:
    """The per-coordinate cache must key on ``direction``, not on the flat index."""
    mesher = FdmMesherComposite(
        Uniform1dMesher(0.0, 1.0, 4), Uniform1dMesher(50.0, 150.0, 11)
    )
    calc = FdmCellAveragingInnerValue(PlainVanillaPayoff(OptionType.Call, 100.0), mesher, 1)
    actual = [calc.avg_inner_value(it, 0.0) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["cell_avg_inner_value_2d_dir1_avg"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def _basket_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(150.0), 5),
        Uniform1dMesher(math.log(60.0), math.log(140.0), 4),
    )


def test_log_basket_inner_value_max(reference_data: dict[str, Any]) -> None:
    mesher = _basket_mesher()
    calc = FdmLogBasketInnerValue(
        MaxBasketPayoff(PlainVanillaPayoff(OptionType.Call, 100.0)), mesher
    )
    actual = [calc.inner_value(it, 0.0) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["log_basket_inner_value_max"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_log_basket_avg_equals_inner(reference_data: dict[str, Any]) -> None:
    """# C++ parity: ``avgInnerValue`` forwards to ``innerValue``."""
    mesher = _basket_mesher()
    calc = FdmLogBasketInnerValue(
        MaxBasketPayoff(PlainVanillaPayoff(OptionType.Call, 100.0)), mesher
    )
    actual = [calc.avg_inner_value(it, 0.0) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["log_basket_avg_inner_value_max"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_log_basket_inner_value_min(reference_data: dict[str, Any]) -> None:
    mesher = _basket_mesher()
    calc = FdmLogBasketInnerValue(
        MinBasketPayoff(PlainVanillaPayoff(OptionType.Call, 100.0)), mesher
    )
    actual = [calc.inner_value(it, 0.0) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["log_basket_inner_value_min"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_zero_inner_value(
    reference_data: dict[str, Any], log_spot_mesher: FdmMesherComposite
) -> None:
    calc = FdmZeroInnerValue()
    it = next(i for i in log_spot_mesher.layout().iter() if i.index == 3)
    tight(calc.inner_value(it, 0.75), float(reference_data["zero_inner_value"]))
    tight(calc.avg_inner_value(it, 0.75), float(reference_data["zero_avg_inner_value"]))
