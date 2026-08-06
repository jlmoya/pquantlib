"""Cross-validation of ``FdmEscrowedLogInnerValueCalculator`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmescrowedloginnervaluecalculator.{hpp,cpp}
# @ v1.43 (6b57206e0).
"""

from __future__ import annotations

from typing import Any

from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.utilities.escrowed_dividend_adjustment import (
    EscrowedDividendAdjustment,
)
from pquantlib.methods.finitedifferences.utilities.fdm_escrowed_log_inner_value_calculator import (
    FdmEscrowedLogInnerValueCalculator,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.testing.tolerance import tight

from .conftest import as_floats


def test_call_inner_value_t025(
    reference_data: dict[str, Any],
    escrowed_adjustment: EscrowedDividendAdjustment,
    log_spot_mesher: FdmMesherComposite,
) -> None:
    calc = FdmEscrowedLogInnerValueCalculator(
        escrowed_adjustment,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        log_spot_mesher,
        0,
    )
    actual = [calc.inner_value(it, 0.25) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["escrowed_log_inner_value_t025"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_call_inner_value_t050(
    reference_data: dict[str, Any],
    escrowed_adjustment: EscrowedDividendAdjustment,
    log_spot_mesher: FdmMesherComposite,
) -> None:
    calc = FdmEscrowedLogInnerValueCalculator(
        escrowed_adjustment,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        log_spot_mesher,
        0,
    )
    actual = [calc.inner_value(it, 0.50) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["escrowed_log_inner_value_t050"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_avg_equals_inner(
    reference_data: dict[str, Any],
    escrowed_adjustment: EscrowedDividendAdjustment,
    log_spot_mesher: FdmMesherComposite,
) -> None:
    """# C++ parity: ``avgInnerValue`` forwards to ``innerValue``."""
    calc = FdmEscrowedLogInnerValueCalculator(
        escrowed_adjustment,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        log_spot_mesher,
        0,
    )
    actual = [calc.avg_inner_value(it, 0.25) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["escrowed_log_avg_inner_value_t025"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_put_inner_value_t025(
    reference_data: dict[str, Any],
    escrowed_adjustment: EscrowedDividendAdjustment,
    log_spot_mesher: FdmMesherComposite,
) -> None:
    """Put branch — this is the most cancellation-prone case in the cluster.

    ``strike - (exp(x) - divAdj)`` subtracts two same-magnitude quantities,
    so the relative deviation from C++ climbs to 2.2e-13 at the node where
    the difference is smallest. Still inside TIGHT (1e-12), so no exception
    is taken.
    """
    calc = FdmEscrowedLogInnerValueCalculator(
        escrowed_adjustment,
        PlainVanillaPayoff(OptionType.Put, 100.0),
        log_spot_mesher,
        0,
    )
    actual = [calc.inner_value(it, 0.25) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["escrowed_log_inner_value_put_t025"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)
