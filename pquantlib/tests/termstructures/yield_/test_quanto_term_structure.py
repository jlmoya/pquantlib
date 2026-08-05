"""Cross-validate ``QuantoTermStructure`` against C++ QuantLib v1.43.

Probe: ``v143/cf/equitycashflow`` (sections ``quanto_term_structure``,
``quanto_term_structure_delegation`` and
``quanto_term_structure_strike_sensitivity``).

Three things are checked:

1. the zero yield ``z_div + z_rf - z_foreign_rf + rho * sigma_eq * sigma_fx``
   and the discount factors it produces;
2. that ``day_counter`` / ``reference_date`` really delegate to the *dividend*
   curve — pinned with a dividend curve whose day counter and reference date
   differ from every other curve in the structure;
3. that the ``strike`` and ``exch_rate_atm_level`` arguments are threaded into
   the two vol surfaces. Against a constant vol those two arguments are
   invisible, so this part uses strike-dependent ``BlackVarianceSurface``s and
   varies only those arguments.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.volatility.equity_fx.black_variance_surface import (
    BlackVarianceSurface,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.quanto_term_structure import QuantoTermStructure
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

_TODAY = Date.from_ymd(27, Month.January, 2023)
_INTEREST_RATE = 0.0375
_DIVIDEND_RATE = 0.005
_QUANTO_RATE = 0.001
_EQUITY_VOL = 0.4
_FX_VOL = 0.2
_CORRELATION = 0.4

_VOL_DATES = [
    Date.from_ymd(27, Month.July, 2023),
    Date.from_ymd(27, Month.January, 2024),
    Date.from_ymd(27, Month.January, 2025),
]
_EQUITY_STRIKES = [7000.0, 8500.0, 10000.0]
_FX_STRIKES = [0.5, 1.0, 1.5]
# rows = strikes, columns = the three pillar dates.
_EQUITY_VOLS = np.asarray([[0.50, 0.46, 0.42], [0.40, 0.38, 0.36], [0.34, 0.33, 0.32]], dtype=np.float64)
_FX_VOLS = np.asarray([[0.28, 0.26, 0.24], [0.20, 0.19, 0.18], [0.16, 0.155, 0.15]], dtype=np.float64)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/equitycashflow")


@pytest.fixture(autouse=True)
def _evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    previous = ObservableSettings().evaluation_date
    ObservableSettings().evaluation_date = _TODAY
    yield
    ObservableSettings().evaluation_date = previous


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(_TODAY, rate, Actual365Fixed())


def _constant_vol(vol: float) -> BlackConstantVol:
    return BlackConstantVol(
        reference_date=_TODAY,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        volatility=vol,
    )


def _surface(strikes: list[float], vols: np.ndarray[Any, np.dtype[np.float64]]) -> BlackVarianceSurface:
    return BlackVarianceSurface(
        reference_date=_TODAY,
        calendar=TARGET(),
        dates=_VOL_DATES,
        strikes=strikes,
        black_vol_matrix=vols,
        day_counter=Actual365Fixed(),
    )


def _zero_rate(curve: QuantoTermStructure, t: float) -> float:
    return curve.zero_rate(t, Compounding.Continuous, Frequency.NoFrequency, True).rate()


# --- zero yields and discounts ------------------------------------------------


def test_zero_rates_and_discounts(cpp: dict[str, Any]) -> None:
    ref = cpp["quanto_term_structure"]
    strike = ref["strike"]
    curve = QuantoTermStructure(
        _flat(_DIVIDEND_RATE),
        _flat(_QUANTO_RATE),
        _flat(_INTEREST_RATE),
        _constant_vol(_EQUITY_VOL),
        strike,
        _constant_vol(_FX_VOL),
        1.0,
        _CORRELATION,
    )
    assert curve.day_counter().name() == ref["day_counter"]
    assert curve.reference_date().serial_number() == ref["reference_date_serial"]
    assert curve.max_date().serial_number() == ref["max_date_serial"]
    for i in (1, 2, 3):
        t = ref[f"time_{i}"]
        tolerance.tight(_zero_rate(curve, t), ref[f"zero_rate_{i}"])
        tolerance.tight(curve.discount(t, True), ref[f"discount_{i}"])
    d = Date(int(ref["discount_date_serial"]))
    tolerance.tight(curve.discount(d, True), ref["discount_at_date"])


def test_correlation_argument_reaches_the_zero_yield(cpp: dict[str, Any]) -> None:
    """rho enters as ``+ rho * sigma_eq * sigma_fx``: flipping its sign must move
    the zero rate by exactly ``2 * rho * sigma_eq * sigma_fx``."""
    ref = cpp["quanto_term_structure"]

    def build(rho: float) -> QuantoTermStructure:
        return QuantoTermStructure(
            _flat(_DIVIDEND_RATE),
            _flat(_QUANTO_RATE),
            _flat(_INTEREST_RATE),
            _constant_vol(_EQUITY_VOL),
            ref["strike"],
            _constant_vol(_FX_VOL),
            1.0,
            rho,
        )

    positive = _zero_rate(build(_CORRELATION), 1.0)
    negative = _zero_rate(build(-_CORRELATION), 1.0)
    # The vols are constant, so the only t-dependence cancels exactly.
    tolerance.tight(positive - negative, 2.0 * _CORRELATION * _EQUITY_VOL * _FX_VOL)


# --- delegation to the dividend curve -----------------------------------------


def test_delegates_day_counter_and_reference_date(cpp: dict[str, Any]) -> None:
    ref = cpp["quanto_term_structure_delegation"]
    dividend_reference = Date(int(ref["dividend_reference_date_serial"]))
    # Actual/360 and a reference date five business days back: neither matches
    # any other curve in the structure.
    dividend = FlatForward.from_rate(dividend_reference, _DIVIDEND_RATE, Actual360())
    curve = QuantoTermStructure(
        dividend,
        _flat(_QUANTO_RATE),
        _flat(_INTEREST_RATE),
        _constant_vol(_EQUITY_VOL),
        8000.0,
        _constant_vol(_FX_VOL),
        1.0,
        _CORRELATION,
    )
    assert curve.day_counter().name() == ref["day_counter"]
    assert curve.reference_date().serial_number() == ref["reference_date_serial"]
    assert curve.reference_date() != _TODAY
    tolerance.tight(_zero_rate(curve, 1.0), ref["zero_rate_1y"])


# --- strike / atm-level threading ---------------------------------------------


@pytest.mark.parametrize(
    ("key", "strike", "atm_level"),
    [
        ("k1_atm1", 8000.0, 1.0),
        ("k1_atm2", 8000.0, 1.25),
        ("k2_atm1", 9000.0, 1.0),
        ("k2_atm2", 9000.0, 1.25),
    ],
)
def test_strike_and_atm_level_reach_the_vol_surfaces(
    cpp: dict[str, Any], key: str, strike: float, atm_level: float
) -> None:
    ref = cpp["quanto_term_structure_strike_sensitivity"]
    for i, d in enumerate(_VOL_DATES, start=1):
        assert d.serial_number() == ref[f"vol_date_serial_{i}"]

    equity_surface = _surface(_EQUITY_STRIKES, _EQUITY_VOLS)
    fx_surface = _surface(_FX_STRIKES, _FX_VOLS)
    t = ref["time"]
    block = ref[key]

    tolerance.tight(equity_surface.black_vol_at_time(t, strike, True), block["equity_vol"])
    tolerance.tight(fx_surface.black_vol_at_time(t, atm_level, True), block["fx_vol"])

    curve = QuantoTermStructure(
        _flat(_DIVIDEND_RATE),
        _flat(_QUANTO_RATE),
        _flat(_INTEREST_RATE),
        equity_surface,
        strike,
        fx_surface,
        atm_level,
        _CORRELATION,
    )
    tolerance.tight(_zero_rate(curve, t), block["zero_rate"])


def test_strike_sensitivity_grid_is_discriminating(cpp: dict[str, Any]) -> None:
    """Guard the guard: the four pinned zero rates must all differ, otherwise
    the parametrised test above would pass for a port that drops an argument."""
    ref = cpp["quanto_term_structure_strike_sensitivity"]
    rates = {ref[k]["zero_rate"] for k in ("k1_atm1", "k1_atm2", "k2_atm1", "k2_atm2")}
    assert len(rates) == 4
