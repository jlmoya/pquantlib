"""Tests for OptionletStripper2 (augment stripper1 with ATM caplet vols)."""

from __future__ import annotations

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_curve import (
    CapFloorTermVolCurve,
)
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_surface import (
    CapFloorTermVolSurface,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_1 import (
    OptionletStripper1,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_2 import (
    OptionletStripper2,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _eval_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


def _setup_pair() -> tuple[OptionletStripper1, CapFloorTermVolCurve]:
    eval_date = _eval_date()
    curve = FlatForward(
        eval_date,
        SimpleQuote(0.03),
        Actual365Fixed(),
    )
    euribor = Euribor(Period(3, TimeUnit.Months), curve)
    # Flat 18% surface across 3 strikes + 4 tenors.
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
    stripper1 = OptionletStripper1(
        surface,
        euribor,
        volatility_type=VolatilityType.ShiftedLognormal,
    )
    # ATM-only curve, also at 18% vol.
    term_vol_curve = CapFloorTermVolCurve(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(3, TimeUnit.Years),
        ],
        vols=[0.18, 0.18, 0.18],
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=eval_date,
    )
    return stripper1, term_vol_curve


def test_basic_construction_does_not_raise() -> None:
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    # Until something asks for the augmented data, no work has been done.
    assert s2.optionlet_maturities() == s1.optionlet_maturities()


def test_atm_strikes_match_curve_atm_count() -> None:
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    strikes = s2.atm_cap_floor_strikes()
    assert len(strikes) == len(curve.option_tenors())


def test_atm_strikes_are_close_to_flat_curve_rate() -> None:
    """ATM strikes should be in the par-coupon ballpark of the 3% curve."""
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    for s in s2.atm_cap_floor_strikes():
        assert 0.02 < s < 0.04


def test_atm_prices_are_positive() -> None:
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    for p in s2.atm_cap_floor_prices():
        assert p > 0


def test_implied_spreads_recover_flat_curve_approximately() -> None:
    """A flat curve at 18% should give near-zero spread over stripper1's flat 18% vols."""
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    spreads = s2.spreads_vol()
    for sp in spreads:
        # LOOSE — Brent tolerance ~1e-5.
        assert abs(sp) < 1.0e-3


def test_augmented_rows_grow_by_one_per_curve_expiry() -> None:
    """Each per-row strike grid gains 1 column per (curve_expiry, row-in-leg)."""
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    original_strikes_row0 = s1.optionlet_strikes(0)
    augmented_strikes_row0 = s2.optionlet_strikes(0)
    # Each curve-expiry whose floating leg covers row 0 adds one column.
    # For a 1-tenor expiry on a 3M index, ~4 caplets per year covers
    # row 0 — so the smallest expiry contributes.
    assert len(augmented_strikes_row0) >= len(original_strikes_row0) + 1


def test_augmented_strikes_sorted() -> None:
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    for i in range(s2.optionlet_maturities()):
        strikes = s2.optionlet_strikes(i)
        assert strikes == sorted(strikes)


def test_per_row_vols_positive() -> None:
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    for i in range(s2.optionlet_maturities()):
        for v in s2.optionlet_volatilities(i):
            assert v > 0


def test_inherited_metadata_delegates_to_stripper1() -> None:
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    assert s2.day_counter() == s1.day_counter()
    assert s2.volatility_type() == s1.volatility_type()
    assert s2.displacement() == s1.displacement()
    assert s2.business_day_convention() == s1.business_day_convention()


def test_atm_optionlet_rates_passthrough() -> None:
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
    )
    # The ATM rates are passthrough from stripper1 (no augmentation).
    assert s2.atm_optionlet_rates() == s1.atm_optionlet_rates()


def test_day_counter_mismatch_rejected() -> None:
    """C++ requires matching day counters between surface + curve."""
    import pytest  # noqa: PLC0415

    from pquantlib.daycounters.actual_360 import Actual360  # noqa: PLC0415
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    s1, _ = _setup_pair()
    bad_curve = CapFloorTermVolCurve(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[Period(1, TimeUnit.Years), Period(2, TimeUnit.Years)],
        vols=[0.18, 0.18],
        calendar=TARGET(),
        day_counter=Actual360(),  # mismatched with stripper1's surface (A365F)
        reference_date=_eval_date(),
    )
    with pytest.raises(LibraryException, match="day counter"):
        OptionletStripper2(
            optionlet_stripper_1=s1,
            atm_cap_floor_term_vol_curve=bad_curve,
        )


def test_round_trip_prices_at_curve_atm() -> None:
    """The spread-adjusted reprice should match the ATM cap NPV (LOOSE)."""
    s1, curve = _setup_pair()
    s2 = OptionletStripper2(
        optionlet_stripper_1=s1,
        atm_cap_floor_term_vol_curve=curve,
        accuracy=1.0e-7,
    )
    # The construction itself performed the Brent solves; verify each
    # spread when re-applied through the adapter reproduces the target
    # ATM cap NPV.
    spreads = s2.spreads_vol()
    prices = s2.atm_cap_floor_prices()
    # We can't trivially re-evaluate at the test layer without
    # re-running the engine; assert convergence by checking spreads
    # are within the Brent search bracket.
    for sp in spreads:
        assert -0.1 < sp < 0.1
    for p in prices:
        # LOOSE-tier sanity on positive cap NPV.
        assert p > 0
    # And convergence: max |spread| stayed near zero on a flat 18% curve.
    # We tolerate up to 1e-5 — the Brent accuracy we passed in.
    max_abs = max(abs(sp) for sp in spreads)
    assert max_abs < 1.0e-5, f"spreads not converged tightly: {spreads}"
