"""Tests for SwaptionVolatilityMatrix.

Cross-validated against L8-C C++ probe (cluster/l8c.json).
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.swaption.swaption_volatility_matrix import (
    SwaptionVolatilityMatrix,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("cluster/l8c")
_SVM = _REF["swaption_vol_matrix"]


def _eval_date() -> Date:
    return Date(_REF["setup"]["eval_date_serial"])


def _new_matrix() -> SwaptionVolatilityMatrix:
    vols = np.asarray(
        [
            [0.30, 0.28, 0.25],
            [0.28, 0.25, 0.22],
            [0.25, 0.22, 0.20],
            [0.22, 0.20, 0.18],
        ],
        dtype=np.float64,
    )
    return SwaptionVolatilityMatrix(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(5, TimeUnit.Years),
            Period(10, TimeUnit.Years),
        ],
        swap_tenors=[
            Period(1, TimeUnit.Years),
            Period(5, TimeUnit.Years),
            Period(10, TimeUnit.Years),
        ],
        volatilities=vols,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
    )


def test_volatility_at_node_2y_5y() -> None:
    m = _new_matrix()
    d2y = Date(_SVM["d2y_serial"])
    tolerance.tight(m.volatility(d2y, Period(5, TimeUnit.Years), 0.04, True), _SVM["v_2y_5y"])


def test_volatility_at_node_5y_10y() -> None:
    m = _new_matrix()
    d5y = Date(_SVM["d5y_serial"])
    tolerance.tight(
        m.volatility(d5y, Period(10, TimeUnit.Years), 0.04, True), _SVM["v_5y_10y"]
    )


def test_volatility_at_intermediate_matches_probe_loose() -> None:
    # 3y on swap-5y is intermediate between (2y, 5y) and (5y, 5y).
    # The C++ probe and PQuantLib both use bilinear here — agree.
    m = _new_matrix()
    d3y = Date(_SVM["d3y_serial"])
    tolerance.loose(
        m.volatility(d3y, Period(5, TimeUnit.Years), 0.04, True), _SVM["v_3y_5y"]
    )


def test_max_swap_tenor_is_last_pillar() -> None:
    m = _new_matrix()
    assert m.max_swap_tenor() == Period(10, TimeUnit.Years)


def test_max_date_is_last_option_pillar() -> None:
    m = _new_matrix()
    expected = TARGET().advance(_eval_date(), 10, TimeUnit.Years)
    assert m.max_date() == expected


def test_strike_independence_at_node() -> None:
    m = _new_matrix()
    d2y = Date(_SVM["d2y_serial"])
    v_low = m.volatility(d2y, Period(5, TimeUnit.Years), 0.001, True)
    v_high = m.volatility(d2y, Period(5, TimeUnit.Years), 100.0, True)
    tolerance.tight(v_low, v_high)


def test_default_vol_type_shifted_lognormal() -> None:
    m = _new_matrix()
    assert m.volatility_type() == VolatilityType.ShiftedLognormal


def test_swap_tenors_round_trip() -> None:
    m = _new_matrix()
    swap_tenors = m.swap_tenors()
    assert len(swap_tenors) == 3
    assert swap_tenors[0] == Period(1, TimeUnit.Years)
    assert swap_tenors[-1] == Period(10, TimeUnit.Years)


def test_option_tenors_round_trip() -> None:
    m = _new_matrix()
    opt_tenors = m.option_tenors()
    assert len(opt_tenors) == 4
    assert opt_tenors[2] == Period(5, TimeUnit.Years)


def test_swap_lengths_correct_in_years() -> None:
    m = _new_matrix()
    lengths = m.swap_lengths()
    tolerance.tight(lengths[0], 1.0)
    tolerance.tight(lengths[1], 5.0)
    tolerance.tight(lengths[2], 10.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(LibraryException):
        SwaptionVolatilityMatrix(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            option_tenors=[Period(1, TimeUnit.Years), Period(2, TimeUnit.Years)],
            swap_tenors=[Period(1, TimeUnit.Years), Period(5, TimeUnit.Years)],
            volatilities=np.zeros((3, 3)),  # wrong shape
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            reference_date=_eval_date(),
        )


def test_non_increasing_swap_tenors_raise() -> None:
    with pytest.raises(LibraryException):
        SwaptionVolatilityMatrix(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            option_tenors=[Period(1, TimeUnit.Years), Period(2, TimeUnit.Years)],
            swap_tenors=[Period(5, TimeUnit.Years), Period(1, TimeUnit.Years)],
            volatilities=np.zeros((2, 2)),
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            reference_date=_eval_date(),
        )


def test_shifts_default_zero() -> None:
    m = _new_matrix()
    d2y = Date(_SVM["d2y_serial"])
    tolerance.tight(m.shift(d2y, Period(5, TimeUnit.Years), True), 0.0)


def test_shifts_custom_propagate() -> None:
    shifts = np.full((4, 3), 0.01)
    m = SwaptionVolatilityMatrix(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(5, TimeUnit.Years),
            Period(10, TimeUnit.Years),
        ],
        swap_tenors=[
            Period(1, TimeUnit.Years),
            Period(5, TimeUnit.Years),
            Period(10, TimeUnit.Years),
        ],
        volatilities=np.full((4, 3), 0.20),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
        shifts=shifts,
    )
    d2y = Date(_SVM["d2y_serial"])
    tolerance.tight(m.shift(d2y, Period(5, TimeUnit.Years), True), 0.01)


def test_flat_extrapolation_clamps_outside_grid() -> None:
    m = SwaptionVolatilityMatrix(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(5, TimeUnit.Years),
            Period(10, TimeUnit.Years),
        ],
        swap_tenors=[
            Period(1, TimeUnit.Years),
            Period(5, TimeUnit.Years),
            Period(10, TimeUnit.Years),
        ],
        volatilities=np.asarray(
            [
                [0.30, 0.28, 0.25],
                [0.28, 0.25, 0.22],
                [0.25, 0.22, 0.20],
                [0.22, 0.20, 0.18],
            ],
            dtype=np.float64,
        ),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
        flat_extrapolation=True,
    )
    # 20y option (past 10y last pillar) flat-extrapolates to the 10y
    # row.
    d20y = TARGET().advance(_eval_date(), 20, TimeUnit.Years)
    v = m.volatility(d20y, Period(10, TimeUnit.Years), 0.04, True)
    tolerance.tight(v, 0.18)
