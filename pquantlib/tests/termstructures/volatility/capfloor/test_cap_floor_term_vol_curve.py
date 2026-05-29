"""Tests for CapFloorTermVolCurve.

Cross-validated against L8-C C++ probe (cluster/l8c.json).
"""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_curve import (
    CapFloorTermVolCurve,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("cluster/l8c")
_CURVE = _REF["capfloor_term_vol_curve"]


def _eval_date() -> Date:
    return Date(_REF["setup"]["eval_date_serial"])


def _new_curve() -> CapFloorTermVolCurve:
    return CapFloorTermVolCurve(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(3, TimeUnit.Years),
            Period(5, TimeUnit.Years),
        ],
        vols=_CURVE["vols"],
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
    )


def test_volatility_at_node_matches_probe() -> None:
    curve = _new_curve()
    d2y = Date(_CURVE["d2y_serial"])
    tolerance.tight(curve.volatility(d2y, 0.05, True), _CURVE["v_at_2y"])


def test_volatility_at_intermediate_uses_linear_interp() -> None:
    # C++ ``CapFloorTermVolCurve`` hard-wires natural cubic-spline
    # interpolation (``CubicInterpolation::Spline,
    # SecondDerivative=0``). PQuantLib uses ``LinearInterpolation``
    # because cubic-spline is in the L1 carve-out (see
    # ``phase1-completion.md``). Therefore the *intermediate* value
    # diverges from the C++ probe value by ~0.07e-2 — we assert
    # internal coherence (linear interp through the two surrounding
    # pillars) and document the divergence.
    curve = _new_curve()
    d2_5y = Date(_CURVE["d2_5y_serial"])
    # Linear interp on (t_2y, 0.18) and (t_3y, 0.16) at t = 2.5y
    # serial.
    dc = Actual365Fixed()
    t_2y = dc.year_fraction(_eval_date(), Date(_CURVE["d2y_serial"]))
    # 3-year pillar — derived from advance per TARGET; mirror the
    # ctor's option_date_from_tenor advance.
    t_3y = dc.year_fraction(
        _eval_date(), TARGET().advance(_eval_date(), 3, TimeUnit.Years)
    )
    t_2_5y = dc.year_fraction(_eval_date(), d2_5y)
    u = (t_2_5y - t_2y) / (t_3y - t_2y)
    expected_linear = (1 - u) * 0.18 + u * 0.16
    tolerance.tight(curve.volatility(d2_5y, 0.05, True), expected_linear)


def test_max_date_returns_last_pillar() -> None:
    curve = _new_curve()
    # Last tenor is 5Y.
    target = TARGET().advance(_eval_date(), 5, TimeUnit.Years)
    assert curve.max_date() == target


def test_option_tenors_inspector() -> None:
    curve = _new_curve()
    tenors = curve.option_tenors()
    assert len(tenors) == 4
    assert tenors[0] == Period(1, TimeUnit.Years)
    assert tenors[-1] == Period(5, TimeUnit.Years)


def test_option_dates_inspector() -> None:
    curve = _new_curve()
    dates = curve.option_dates()
    assert len(dates) == 4
    assert dates[1].serial_number() == _CURVE["d2y_serial"]


def test_strike_independence() -> None:
    curve = _new_curve()
    d3y = TARGET().advance(_eval_date(), 3, TimeUnit.Years)
    v_low = curve.volatility(d3y, 0.001, True)
    v_high = curve.volatility(d3y, 100.0, True)
    tolerance.tight(v_low, v_high)


def test_min_max_strike_infinite() -> None:
    curve = _new_curve()
    assert curve.min_strike() == -math.inf
    assert curve.max_strike() == math.inf


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(LibraryException):
        CapFloorTermVolCurve(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            option_tenors=[Period(1, TimeUnit.Years)],
            vols=[0.2, 0.18],
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            reference_date=_eval_date(),
        )


def test_empty_tenors_raise() -> None:
    with pytest.raises(LibraryException):
        CapFloorTermVolCurve(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            option_tenors=[],
            vols=[],
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            reference_date=_eval_date(),
        )
