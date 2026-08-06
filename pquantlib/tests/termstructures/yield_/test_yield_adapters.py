"""Layered yield term structures, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/yieldadapters.json
Probe:     migration-harness/cpp/probes/v143_ts_yieldadapters/probe.cpp

Covers ``UltimateForwardTermStructure``, ``CompositeZeroYieldStructure``,
``QuantoTermStructure``, ``InterpolatedPiecewiseZeroSpreadedTermStructure``,
``InterpolatedPiecewiseForwardSpreadedTermStructure`` and
``InterpolatedSimpleZeroCurve``. The probe reads each at times straddling
every branch boundary in its implementation, so the tolerance is checked
against the discriminating cases rather than the easy ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.interpolations.backward_flat import BackwardFlatInterpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.composite_zero_yield_structure import (
    CompositeZeroYieldStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.interpolated_simple_zero_curve import (
    InterpolatedSimpleZeroCurve,
)
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.termstructures.yield_.piecewise_forward_spreaded_term_structure import (
    InterpolatedPiecewiseForwardSpreadedTermStructure,
)
from pquantlib.termstructures.yield_.piecewise_zero_spreaded_term_structure import (
    InterpolatedPiecewiseZeroSpreadedTermStructure,
)
from pquantlib.termstructures.yield_.quanto_term_structure import QuantoTermStructure
from pquantlib.termstructures.yield_.ultimate_forward_term_structure import (
    UltimateForwardTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing import tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration-harness/references/v143/ts/yieldadapters.json"
)

_REF_DATE = Date.from_ymd(17, Month.January, 2024)
_Y = TimeUnit.Years


def _years(n: int) -> Period:
    return Period(n, _Y)


def _base_curve() -> YieldTermStructure:
    dates = [
        _REF_DATE,
        _REF_DATE + _years(1),
        _REF_DATE + _years(3),
        _REF_DATE + _years(5),
        _REF_DATE + _years(10),
        _REF_DATE + _years(30),
    ]
    zeros = [0.0250, 0.0280, 0.0315, 0.0330, 0.0345, 0.0360]
    curve = InterpolatedZeroCurve(dates, zeros, Actual365Fixed(), calendar=TARGET())
    curve.enable_extrapolation()
    return curve


def _flat(r: float) -> YieldTermStructure:
    curve = FlatForward.from_rate(
        _REF_DATE, r, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    curve.enable_extrapolation()
    return curve


def _spread_dates() -> list[Date]:
    return [_REF_DATE + _years(2), _REF_DATE + _years(5), _REF_DATE + _years(10)]


def _spread_quotes() -> list[Quote]:
    return [SimpleQuote(0.0010), SimpleQuote(0.0035), SimpleQuote(0.0020)]


def _llfr() -> Quote:
    return SimpleQuote(0.0426)


def _ufr() -> Quote:
    return SimpleQuote(0.0210)


def _enabled(curve: YieldTermStructure) -> YieldTermStructure:
    curve.enable_extrapolation()
    return curve


def _builders() -> dict[str, Callable[[], YieldTermStructure]]:
    return {
        "ufr_alpha_010": lambda: _enabled(
            UltimateForwardTermStructure(_base_curve(), _llfr(), _ufr(), _years(5), 0.1)
        ),
        "ufr_alpha_002": lambda: _enabled(
            UltimateForwardTermStructure(_base_curve(), _llfr(), _ufr(), _years(5), 0.02)
        ),
        "ufr_fsp_20y": lambda: _enabled(
            UltimateForwardTermStructure(_base_curve(), _llfr(), _ufr(), _years(20), 0.1)
        ),
        "ufr_rounded_4dp_annual": lambda: _enabled(
            UltimateForwardTermStructure(
                _base_curve(), _llfr(), _ufr(), _years(5), 0.1, 4,
                Compounding.Compounded, Frequency.Annual,
            )
        ),
        "ufr_rounded_5dp_continuous": lambda: _enabled(
            UltimateForwardTermStructure(
                _base_curve(), _llfr(), _ufr(), _years(5), 0.1, 5,
                Compounding.Continuous, Frequency.NoFrequency,
            )
        ),
        "composite_sum_continuous": lambda: _enabled(
            CompositeZeroYieldStructure(
                _base_curve(), _flat(0.005), lambda a, b: a + b
            )
        ),
        "composite_sum_annual": lambda: _enabled(
            CompositeZeroYieldStructure(
                _base_curve(), _flat(0.005), lambda a, b: a + b,
                Compounding.Compounded, Frequency.Annual,
            )
        ),
        "composite_diff_semiannual": lambda: _enabled(
            CompositeZeroYieldStructure(
                _base_curve(), _flat(0.005), lambda a, b: a - b,
                Compounding.Compounded, Frequency.Semiannual,
            )
        ),
        "quanto": lambda: _enabled(
            QuantoTermStructure(
                _flat(0.015),
                _flat(0.030),
                _flat(0.022),
                BlackConstantVol(
                    reference_date=_REF_DATE,
                    calendar=TARGET(),
                    day_counter=Actual365Fixed(),
                    volatility=0.25,
                ),
                100.0,
                BlackConstantVol(
                    reference_date=_REF_DATE,
                    calendar=TARGET(),
                    day_counter=Actual365Fixed(),
                    volatility=0.12,
                ),
                1.25,
                -0.4,
            )
        ),
        "zero_spreaded_linear": lambda: _enabled(
            InterpolatedPiecewiseZeroSpreadedTermStructure(
                _base_curve(), _spread_quotes(), _spread_dates()
            )
        ),
        "zero_spreaded_linear_annual": lambda: _enabled(
            InterpolatedPiecewiseZeroSpreadedTermStructure(
                _base_curve(), _spread_quotes(), _spread_dates(),
                Compounding.Compounded, Frequency.Annual,
            )
        ),
        "zero_spreaded_backward_flat": lambda: _enabled(
            InterpolatedPiecewiseZeroSpreadedTermStructure(
                _base_curve(), _spread_quotes(), _spread_dates(),
                interpolator=BackwardFlatInterpolation,
            )
        ),
        "forward_spreaded_linear": lambda: _enabled(
            InterpolatedPiecewiseForwardSpreadedTermStructure(
                _base_curve(), _spread_quotes(), _spread_dates()
            )
        ),
        "forward_spreaded_backward_flat": lambda: _enabled(
            InterpolatedPiecewiseForwardSpreadedTermStructure(
                _base_curve(), _spread_quotes(), _spread_dates(),
                interpolator=BackwardFlatInterpolation,
            )
        ),
        "simple_zero_linear": lambda: _enabled(
            InterpolatedSimpleZeroCurve(
                [
                    _REF_DATE,
                    _REF_DATE + _years(1),
                    _REF_DATE + _years(3),
                    _REF_DATE + _years(5),
                    _REF_DATE + _years(10),
                ],
                [0.0250, 0.0280, 0.0315, 0.0330, 0.0345],
                Actual365Fixed(),
                calendar=TARGET(),
                interpolator=LinearInterpolation,
            )
        ),
    }


_Curves = dict[str, dict[str, list[float]]]
_Dates = dict[str, dict[str, int]]
_Refs = tuple[list[float], _Curves, _Dates]


@pytest.fixture(scope="module")
def refs() -> _Refs:
    raw = cast("dict[str, object]", json.loads(_REF_PATH.read_text()))
    times = cast("list[float]", raw["times"])
    blocks = {k: cast("dict[str, object]", v) for k, v in raw.items() if k != "times"}
    curves: _Curves = {
        k: {n: cast("list[float]", b[n]) for n in ("discount", "zero")}
        for k, b in blocks.items()
    }
    dates: _Dates = {
        k: {n: cast("int", b[n]) for n in ("reference_date", "max_date")}
        for k, b in blocks.items()
    }
    return times, curves, dates


def test_every_probe_case_is_covered(refs: _Refs) -> None:
    _, curves, _dates = refs
    assert set(_builders()) == set(curves)


@pytest.mark.parametrize("key", sorted(_builders()))
def test_discount_matches_cpp(refs: _Refs, key: str) -> None:
    times, curves, _dates = refs
    curve = _builders()[key]()
    for t, expected in zip(times, curves[key]["discount"], strict=True):
        tolerance.tight(
            curve.discount(t, extrapolate=True), expected, reason=f"{key}@t={t}"
        )


@pytest.mark.parametrize("key", sorted(_builders()))
def test_zero_rate_matches_cpp(refs: _Refs, key: str) -> None:
    times, curves, _dates = refs
    curve = _builders()[key]()
    for t, expected in zip(times, curves[key]["zero"], strict=True):
        rate = curve.zero_rate(
            t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
        ).rate()
        # t == 0 is finite-differenced over dt = 1e-4 by both C++ and the port
        # (yieldtermstructure.cpp), which multiplies double rounding by 1e4;
        # see test_interpolated_discount_curve_extrapolation for the derivation.
        check = tolerance.loose if t == 0.0 else tolerance.tight
        check(rate, expected, reason=f"{key}@t={t}")


@pytest.mark.parametrize("key", sorted(_builders()))
def test_reference_and_max_date_match_cpp(refs: _Refs, key: str) -> None:
    _times, _curves, dates = refs
    curve = _builders()[key]()
    assert curve.reference_date().serial == dates[key]["reference_date"], key
    assert curve.max_date().serial == dates[key]["max_date"], key
