"""Tests for OptionletStripper1.

Cross-validated against L8-C C++ probe (cluster/l8c.json).

The stripper inverts cap NPVs to recover caplet-by-caplet implied
vols. On a flat 18% lognormal CapFloorTermVolSurface (single column
of strikes / per-strike vol = 18%), each stripped caplet vol should
equal 18% modulo the Brent-residual at machine precision —
asserted at LOOSE tier (1e-8).
"""

from __future__ import annotations

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_surface import (
    CapFloorTermVolSurface,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_1 import (
    OptionletStripper1,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("cluster/l8c")
_S1 = _REF["optionlet_stripper_1"]


def _eval_date() -> Date:
    return Date(_REF["setup"]["eval_date_serial"])


def _setup_stripper() -> OptionletStripper1:
    eval_date = _eval_date()
    # Flat 3% Act/365F discount curve.
    curve = FlatForward(
        eval_date,
        SimpleQuote(0.03),
        Actual365Fixed(),
    )
    euribor = Euribor(Period(3, TimeUnit.Months), curve)
    # Flat 18% CapFloorTermVolSurface across 3 strikes.
    vols = np.full((4, 3), 0.18, dtype=np.float64)
    surface = CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(3, TimeUnit.Years),
            Period(5, TimeUnit.Years),
        ],
        strikes=[0.02, 0.04, 0.06],
        volatilities=vols,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=eval_date,
    )
    return OptionletStripper1(
        surface,
        euribor,
        volatility_type=VolatilityType.ShiftedLognormal,
    )


def test_optionlet_maturities_match_probe() -> None:
    stripper = _setup_stripper()
    expected = _S1["n_fixings"]
    assert stripper.optionlet_maturities() == expected
    times = stripper.optionlet_fixing_times()
    assert len(times) == expected


def test_caplet_vols_recover_input_flat_18pct() -> None:
    stripper = _setup_stripper()
    n = stripper.optionlet_maturities()
    for i in range(n):
        v = stripper.optionlet_volatilities(i)[1]  # column 1 = 0.04
        tolerance.loose(v, 0.18)


def test_volatility_type_and_displacement_round_trip() -> None:
    stripper = _setup_stripper()
    assert stripper.volatility_type() == VolatilityType.ShiftedLognormal
    assert stripper.displacement() == 0.0


def test_switch_strike_floats_to_atm_average() -> None:
    stripper = _setup_stripper()
    # No explicit switch_strike → switches to average ATM.
    ss = stripper.switch_strike()
    assert 0.01 < ss < 0.1


def test_atm_optionlet_rates_populated_after_calc() -> None:
    stripper = _setup_stripper()
    # Force a calc + read.
    _ = stripper.optionlet_volatilities(0)
    rates = stripper.atm_optionlet_rates()
    assert len(rates) == _S1["n_fixings"]
    for r in rates:
        # Each cap's last-coupon ATM should be close to the 3% curve
        # forward — within 30 bps for the various maturities.
        assert 0.02 < r < 0.05


def test_strikes_round_trip() -> None:
    stripper = _setup_stripper()
    assert stripper.optionlet_strikes(0) == [0.02, 0.04, 0.06]


def test_caplet_vols_align_with_cpp_probe() -> None:
    # Both PQuantLib and C++ should recover ~0.18 modulo their Brent
    # residuals. We assert each stripped vol agrees with the
    # corresponding C++ probe value at LOOSE (1e-8), reflecting the
    # fact that both implementations Brent-solve the same Black-76
    # implied-vol problem from the same NPV difference.
    stripper = _setup_stripper()
    expected = _S1["caplet_vols_04"]
    for i, exp in enumerate(expected):
        tolerance.loose(stripper.optionlet_volatilities(i)[1], exp)
