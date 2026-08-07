"""Tests for CapFloorTermVolSurface.

Cross-validated against L8-C C++ probe (cluster/l8c.json).

C++ ``CapFloorTermVolSurface`` hard-wires bicubic-spline 2-D
interpolation; PQuantLib uses bilinear (the bicubic spline is in
the L1 carve-out — see ``phase1-completion.md``). At pillar nodes
the two agree exactly; intermediate-strike values on locally linear
smiles also agree. Off-node time values diverge in the cubic
corrections — we assert TIGHT at nodes and use a custom-derived
expected for the one intermediate point.
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_surface import (
    CapFloorTermVolSurface,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("cluster/l8c")
_SURF = _REF["capfloor_term_vol_surface"]


def _eval_date() -> Date:
    return Date(_REF["setup"]["eval_date_serial"])


def _new_surface() -> CapFloorTermVolSurface:
    # Matrix layout: rows = tenors (1y, 2y, 3y, 5y), columns = strikes
    # (0.02, 0.04, 0.06).
    vols = np.asarray(
        [
            [0.22, 0.20, 0.18],
            [0.20, 0.18, 0.16],
            [0.18, 0.16, 0.14],
            [0.16, 0.15, 0.13],
        ],
        dtype=np.float64,
    )
    return CapFloorTermVolSurface(
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
        reference_date=_eval_date(),
    )


def test_volatility_at_node_2y_4pct_matches_probe() -> None:
    surf = _new_surface()
    d2y = Date(_SURF["d2y_serial"])
    tolerance.tight(surf.volatility(d2y, 0.04, True), _SURF["v_2y_4pct"])


def test_volatility_at_node_3y_2pct_matches_probe() -> None:
    surf = _new_surface()
    d3y = Date(_SURF["d3y_serial"])
    tolerance.tight(surf.volatility(d3y, 0.02, True), _SURF["v_3y_2pct"])


def test_volatility_intermediate_strike_at_node_tenor_matches_probe() -> None:
    surf = _new_surface()
    d2y = Date(_SURF["d2y_serial"])
    # The 2y row is (0.20, 0.18, 0.16) on strikes (0.02, 0.04, 0.06).
    # At strike=0.03 (midpoint between 0.02 and 0.04), linear is 0.19;
    # bicubic also gives 0.19 because the smile is locally linear
    # there (equispaced points on a straight segment). The probe
    # value is 0.19, both interpolators agree.
    tolerance.tight(surf.volatility(d2y, 0.03, True), _SURF["v_2y_3pct"])


def test_volatility_at_an_off_node_time_matches_cpp() -> None:
    """The off-node time value the bilinear default used to miss.

    C++ ``CapFloorTermVolSurface::interpolate`` (capfloortermvolsurface.cpp:
    186-193) hard-codes ``BicubicSpline`` and offers no interpolator parameter.
    This port defaulted to bilinear, so at t = 2.5y it returned the linear
    blend 0.17 where C++ returns 0.16935225825198075 -- and the probe had
    recorded that number all along, under a test that asserted the bilinear
    value instead and explained the gap in a comment.

    With the default corrected the two agree to TIGHT. The bilinear blend is
    recomputed here and asserted to be a DIFFERENT number, so this cannot pass
    again by accident if the default regresses.
    """
    surf = _new_surface()
    d2_5y = TARGET().advance_period(
        _eval_date(),
        Period(30, TimeUnit.Months),
        BusinessDayConvention.ModifiedFollowing,
    )
    tolerance.tight(surf.volatility(d2_5y, 0.04, True), _SURF["v_2_5y_4pct"])

    dc = Actual365Fixed()
    t_2y = dc.year_fraction(_eval_date(), Date(_SURF["d2y_serial"]))
    t_3y = dc.year_fraction(_eval_date(), Date(_SURF["d3y_serial"]))
    t_2_5y = dc.year_fraction(_eval_date(), d2_5y)
    u = (t_2_5y - t_2y) / (t_3y - t_2y)
    bilinear = (1 - u) * 0.18 + u * 0.16
    assert abs(bilinear - _SURF["v_2_5y_4pct"]) > 1e-4


def test_max_date_returns_last_tenor() -> None:
    surf = _new_surface()
    expected = TARGET().advance(_eval_date(), 5, TimeUnit.Years)
    assert surf.max_date() == expected


def test_min_max_strike_match_input() -> None:
    surf = _new_surface()
    assert surf.min_strike() == 0.02
    assert surf.max_strike() == 0.06


def test_option_inspectors_return_pillars() -> None:
    surf = _new_surface()
    assert len(surf.option_tenors()) == 4
    assert len(surf.option_dates()) == 4
    assert len(surf.option_times()) == 4
    assert surf.strikes() == [0.02, 0.04, 0.06]


def test_wrong_matrix_shape_raises() -> None:
    with pytest.raises(LibraryException):
        CapFloorTermVolSurface(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            option_tenors=[Period(1, TimeUnit.Years), Period(2, TimeUnit.Years)],
            strikes=[0.02, 0.04, 0.06],
            volatilities=np.zeros((3, 2)),  # transposed
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            reference_date=_eval_date(),
        )


def test_non_increasing_strikes_raise() -> None:
    with pytest.raises(LibraryException):
        CapFloorTermVolSurface(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            option_tenors=[Period(1, TimeUnit.Years), Period(2, TimeUnit.Years)],
            strikes=[0.04, 0.02],
            volatilities=np.zeros((2, 2)),
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            reference_date=_eval_date(),
        )


def test_non_increasing_tenors_raise() -> None:
    with pytest.raises(LibraryException):
        CapFloorTermVolSurface(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            option_tenors=[Period(2, TimeUnit.Years), Period(1, TimeUnit.Years)],
            strikes=[0.02, 0.04],
            volatilities=np.zeros((2, 2)),
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            reference_date=_eval_date(),
        )
