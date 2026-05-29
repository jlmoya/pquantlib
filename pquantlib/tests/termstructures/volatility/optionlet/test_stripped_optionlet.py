"""Tests for StrippedOptionlet container + StrippedOptionletAdapter."""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet import (
    StrippedOptionlet,
)
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_adapter import (
    StrippedOptionletAdapter,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit


def _new_container() -> StrippedOptionlet:
    ref = Date.from_ymd(15, Month.January, 2024)
    dates = [
        TARGET().advance(ref, 1, TimeUnit.Years),
        TARGET().advance(ref, 2, TimeUnit.Years),
        TARGET().advance(ref, 3, TimeUnit.Years),
    ]
    vols = [
        [0.22, 0.20, 0.18],
        [0.20, 0.18, 0.16],
        [0.18, 0.16, 0.14],
    ]
    return StrippedOptionlet(
        settlement_days=2,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        ibor_index=object(),  # sentinel; not used by the container
        optionlet_dates=dates,
        strikes=[0.02, 0.04, 0.06],
        optionlet_volatilities=vols,
        day_counter=Actual365Fixed(),
    )


def test_container_inspectors_round_trip() -> None:
    c = _new_container()
    assert c.optionlet_maturities() == 3
    assert c.optionlet_strikes(0) == [0.02, 0.04, 0.06]
    assert c.optionlet_volatilities(0) == [0.22, 0.20, 0.18]
    assert c.optionlet_volatilities(2) == [0.18, 0.16, 0.14]


def test_container_mismatched_row_raises() -> None:
    ref = Date.from_ymd(15, Month.January, 2024)
    with pytest.raises(LibraryException):
        StrippedOptionlet(
            settlement_days=2,
            calendar=TARGET(),
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            ibor_index=object(),
            optionlet_dates=[TARGET().advance(ref, 1, TimeUnit.Years)],
            strikes=[0.02, 0.04],
            optionlet_volatilities=[[0.20]],  # row too short
            day_counter=Actual365Fixed(),
        )


def test_container_settlement_days() -> None:
    c = _new_container()
    assert c.settlement_days() == 2


def test_adapter_at_node_returns_pillar_vol() -> None:
    c = _new_container()
    adapter = StrippedOptionletAdapter(c)
    # Adapter linearly interpolates strike then time. At node
    # (date[0]=1y, strike=0.04) we should get the (0, 1) pillar
    # vol = 0.20 modulo the LinearInterpolation pass-through.
    t0 = c.optionlet_fixing_times()[0]
    v = adapter._volatility_impl(t0, 0.04)  # pyright: ignore[reportPrivateUsage]
    tolerance.tight(v, 0.20)


def test_adapter_volatility_type_and_displacement_forward() -> None:
    c = _new_container()
    adapter = StrippedOptionletAdapter(c)
    assert adapter.volatility_type() == VolatilityType.ShiftedLognormal
    assert adapter.displacement() == 0.0


def test_adapter_min_max_strike_from_container() -> None:
    c = _new_container()
    adapter = StrippedOptionletAdapter(c)
    assert adapter.min_strike() == 0.02
    assert adapter.max_strike() == 0.06


def test_adapter_strike_interp_intermediate() -> None:
    c = _new_container()
    adapter = StrippedOptionletAdapter(c)
    # At date 0, vols are (0.22, 0.20, 0.18) on strikes
    # (0.02, 0.04, 0.06). Midpoint strike 0.03 should give 0.21.
    t0 = c.optionlet_fixing_times()[0]
    v = adapter._volatility_impl(t0, 0.03)  # pyright: ignore[reportPrivateUsage]
    tolerance.tight(v, 0.21)


def test_adapter_time_interp_intermediate() -> None:
    c = _new_container()
    adapter = StrippedOptionletAdapter(c)
    # Times at indices 0, 1, 2 with strike 0.04 — vols
    # (0.20, 0.18, 0.16). Linear interp between t0 and t1 at strike
    # 0.04 evaluated at midpoint should be 0.19.
    times = c.optionlet_fixing_times()
    t_mid = 0.5 * (times[0] + times[1])
    v = adapter._volatility_impl(t_mid, 0.04)  # pyright: ignore[reportPrivateUsage]
    tolerance.tight(v, 0.19)


def test_adapter_max_date_from_container() -> None:
    c = _new_container()
    adapter = StrippedOptionletAdapter(c)
    assert adapter.max_date() == c.optionlet_fixing_dates()[-1]
