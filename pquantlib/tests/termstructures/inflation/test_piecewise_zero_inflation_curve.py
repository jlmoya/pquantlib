"""Tests for PiecewiseZeroInflationCurve + ZeroCouponInflationSwapHelper.

# C++ parity: ql/termstructures/inflation/piecewisezeroinflationcurve.hpp +
   ql/termstructures/inflation/inflationhelpers.{hpp,cpp} (v1.42.1).

Roundtrip: build a 3-instrument piecewise curve from
``ZeroCouponInflationSwapHelper``s + verify the bootstrap pins each
helper's implied quote within LOOSE tolerance (solver convergence).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.inflation.eu_hicp import EUHICP
from pquantlib.termstructures.inflation.inflation_helpers import (
    ZeroCouponInflationSwapHelper,
)
from pquantlib.termstructures.inflation.piecewise_zero_inflation_curve import (
    PiecewiseZeroInflationCurve,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_HARNESS_REF = (
    Path(__file__).parents[4]
    / "migration-harness"
    / "references"
    / "cluster"
    / "l7b.json"
)


def _load_piecewise_ref() -> dict[str, object]:
    blob = cast(dict[str, object], json.loads(_HARNESS_REF.read_text()))
    obj = blob["piecewise_zero_roundtrip"]
    assert isinstance(obj, dict)
    return cast(dict[str, object], obj)


def _build_index_with_history(today: Date) -> EUHICP:
    """Build an EUHICP index with two stored fixings (matches L7-B probe).

    The probe stores `100.0` at 1-Oct-2019 and `100.5` at 1-Nov-2019 —
    these are the base fixings the helper needs to compute I_0.
    """
    zii = EUHICP()
    # Clear historic state and add the two known fixings.
    base_fix = Date.from_ymd(1, Month.October, 2019)
    next_fix = Date.from_ymd(1, Month.November, 2019)
    zii.add_fixing(base_fix, 100.0, force_overwrite=True)
    zii.add_fixing(next_fix, 100.5, force_overwrite=True)
    # Help the bootstrap: the curve will need an I_0 at its base date.
    # The curve base date is `inflation_period(today - 3M, Monthly).first`;
    # for today = 15-Jan-2020 and lag = 3M, that's Oct-2019 → start = 1-Oct-2019.
    del today
    return zii


def _build_helpers(
    today: Date,
    zii: EUHICP,
    quotes: list[float],
    maturity_serials: list[int],
) -> list[ZeroCouponInflationSwapHelper]:
    """Build 3 ZCIIS helpers matching the L7-B probe topology."""
    calendar = TARGET()
    bdc = BusinessDayConvention.ModifiedFollowing
    swap_dc = Thirty360(Thirty360Convention.BondBasis)
    swap_obs_lag = Period(3, TimeUnit.Months)
    helpers: list[ZeroCouponInflationSwapHelper] = []
    for q, mat_serial in zip(quotes, maturity_serials, strict=True):
        helpers.append(
            ZeroCouponInflationSwapHelper(
                quote=q,
                observation_lag=swap_obs_lag,
                maturity=Date(mat_serial),
                calendar=calendar,
                payment_convention=bdc,
                day_counter=swap_dc,
                index=zii,
            )
        )
    del today
    return helpers


def _build_nominal_yts(today: Date) -> FlatForward:
    return FlatForward.from_rate(today, 0.04, Actual360(), Compounding.Continuous)


def test_piecewise_zero_roundtrip_matches_input_quotes() -> None:
    """Bootstrap roundtrip: each helper.implied_quote() ≈ input quote (LOOSE).

    Tolerance LOOSE: bootstrap convergence is set at 1e-12, but
    finite-precision arithmetic in the chained interpolation /
    fair-rate / discount path accumulates ~1e-10 noise.
    """
    today = Date.from_ymd(15, Month.January, 2020)
    ref = _load_piecewise_ref()
    quotes = cast(list[float], list(ref["quotes"]))  # type: ignore[call-overload]
    maturity_serials = cast(list[int], list(ref["maturity_serials"]))  # type: ignore[call-overload]

    zii = _build_index_with_history(today)
    helpers = _build_helpers(today, zii, quotes, maturity_serials)
    nominal_yts = _build_nominal_yts(today)

    curve = PiecewiseZeroInflationCurve(
        reference_date=today,
        calendar=TARGET(),
        day_counter=Actual360(),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        instruments=helpers,
        nominal_yts=nominal_yts,
    )
    assert curve is not None
    # The roundtrip: every helper's implied quote matches its input quote.
    for h, q in zip(helpers, quotes, strict=True):
        loose(
            h.implied_quote(),
            q,
            reason="bootstrap solver tolerance + finite-precision drift in "
            "interpolation/forecast chain (target accuracy = 1e-12).",
        )


def test_piecewise_zero_curve_base_date_matches_cpp() -> None:
    """C++ parity: base_date = inflation_period(today - lag, freq).first."""
    today = Date.from_ymd(15, Month.January, 2020)
    ref = _load_piecewise_ref()
    quotes = cast(list[float], list(ref["quotes"]))  # type: ignore[call-overload]
    maturity_serials = cast(list[int], list(ref["maturity_serials"]))  # type: ignore[call-overload]

    zii = _build_index_with_history(today)
    helpers = _build_helpers(today, zii, quotes, maturity_serials)
    curve = PiecewiseZeroInflationCurve(
        reference_date=today,
        calendar=TARGET(),
        day_counter=Actual360(),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        instruments=helpers,
    )
    expected_base = Date(int(ref["curve_base_serial"]))  # type: ignore[arg-type]
    assert curve.base_date() == expected_base


def test_piecewise_zero_requires_at_least_one_helper() -> None:
    """C++ parity: empty helper list is rejected."""
    today = Date.from_ymd(15, Month.January, 2020)
    with pytest.raises(Exception, match="no helpers"):
        PiecewiseZeroInflationCurve(
            reference_date=today,
            calendar=TARGET(),
            day_counter=Actual360(),
            observation_lag=Period(3, TimeUnit.Months),
            frequency=Frequency.Monthly,
            instruments=[],
        )


def test_piecewise_zero_instruments_inspector_returns_copy() -> None:
    """instruments() returns a defensive copy."""
    today = Date.from_ymd(15, Month.January, 2020)
    ref = _load_piecewise_ref()
    quotes = cast(list[float], list(ref["quotes"]))  # type: ignore[call-overload]
    maturity_serials = cast(list[int], list(ref["maturity_serials"]))  # type: ignore[call-overload]
    zii = _build_index_with_history(today)
    helpers = _build_helpers(today, zii, quotes, maturity_serials)
    curve = PiecewiseZeroInflationCurve(
        reference_date=today,
        calendar=TARGET(),
        day_counter=Actual360(),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        instruments=helpers,
    )
    i1 = curve.instruments()
    i2 = curve.instruments()
    assert i1 is not i2
    assert len(i1) == len(helpers)


def test_piecewise_zero_accuracy_inspector() -> None:
    today = Date.from_ymd(15, Month.January, 2020)
    ref = _load_piecewise_ref()
    quotes = cast(list[float], list(ref["quotes"]))  # type: ignore[call-overload]
    maturity_serials = cast(list[int], list(ref["maturity_serials"]))  # type: ignore[call-overload]
    zii = _build_index_with_history(today)
    helpers = _build_helpers(today, zii, quotes, maturity_serials)
    curve = PiecewiseZeroInflationCurve(
        reference_date=today,
        calendar=TARGET(),
        day_counter=Actual360(),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        instruments=helpers,
        accuracy=1.0e-10,
    )
    assert curve.accuracy() == 1.0e-10
