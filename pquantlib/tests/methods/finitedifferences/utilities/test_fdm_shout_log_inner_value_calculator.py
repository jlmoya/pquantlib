"""Cross-validation of ``FdmShoutLogInnerValueCalculator`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmshoutloginnervaluecalculator.{hpp,cpp}
# @ v1.43 (6b57206e0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.utilities.escrowed_dividend_adjustment import (
    EscrowedDividendAdjustment,
)
from pquantlib.methods.finitedifferences.utilities.fdm_shout_log_inner_value_calculator import (
    FdmShoutLogInnerValueCalculator,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

from .conftest import as_floats


@pytest.fixture
def black_vol(
    today: Date, calendar: Calendar, day_counter: DayCounter
) -> BlackVolTermStructure:
    """Probe setup: flat 25% Black vol."""
    return BlackConstantVol(
        reference_date=today, calendar=calendar, volatility=0.25, day_counter=day_counter
    )


@pytest.mark.parametrize(
    ("t", "key"),
    [
        (0.25, "shout_log_inner_value_call_t025"),
        (0.75, "shout_log_inner_value_call_t075"),
    ],
)
def test_call_inner_value(
    reference_data: dict[str, Any],
    black_vol: BlackVolTermStructure,
    escrowed_adjustment: EscrowedDividendAdjustment,
    log_spot_mesher: FdmMesherComposite,
    t: float,
    key: str,
) -> None:
    """``max(0, blackFormula(type, s_t, fwd, stdDev, df) + intrinsic * df)``.

    Note the Black *strike* is ``s_t`` itself (the shout locks the strike at
    the current spot), not the payoff strike.
    """
    calc = FdmShoutLogInnerValueCalculator(
        black_vol,
        escrowed_adjustment,
        1.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        log_spot_mesher,
        0,
    )
    actual = [calc.inner_value(it, t) for it in log_spot_mesher.layout().iter()]
    for a, e in zip(actual, as_floats(reference_data[key]), strict=True):
        tight(a, e)


def test_avg_equals_inner(
    reference_data: dict[str, Any],
    black_vol: BlackVolTermStructure,
    escrowed_adjustment: EscrowedDividendAdjustment,
    log_spot_mesher: FdmMesherComposite,
) -> None:
    """# C++ parity: ``avgInnerValue`` forwards to ``innerValue``."""
    calc = FdmShoutLogInnerValueCalculator(
        black_vol,
        escrowed_adjustment,
        1.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        log_spot_mesher,
        0,
    )
    actual = [calc.avg_inner_value(it, 0.25) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["shout_log_avg_inner_value_call_t025"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_put_inner_value(
    reference_data: dict[str, Any],
    black_vol: BlackVolTermStructure,
    escrowed_adjustment: EscrowedDividendAdjustment,
    log_spot_mesher: FdmMesherComposite,
) -> None:
    calc = FdmShoutLogInnerValueCalculator(
        black_vol,
        escrowed_adjustment,
        1.0,
        PlainVanillaPayoff(OptionType.Put, 100.0),
        log_spot_mesher,
        0,
    )
    actual = [calc.inner_value(it, 0.25) for it in log_spot_mesher.layout().iter()]
    expected = as_floats(reference_data["shout_log_inner_value_put_t025"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)
