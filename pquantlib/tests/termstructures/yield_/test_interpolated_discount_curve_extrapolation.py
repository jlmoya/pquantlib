"""InterpolatedDiscountCurve extrapolation, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/discountcurve.json
Probe:     migration-harness/cpp/probes/v143_ts_discountcurve/probe.cpp

``discountImpl`` interpolates up to the last node and then switches to an
explicit flat-forward extension. Under ``LogLinear`` those two rules coincide
exactly, so a port that simply kept interpolating would pass every LogLinear
test; ``Linear`` and ``Cubic`` on discount factors separate them, and the
instantaneous forward makes the difference loud (it is constant past the last
node iff the flat-forward branch is taken).

LogCubic is in the C++ probe but not exercised here: PQuantLib has no
``LogCubic`` interpolator (deferred in the L1-E interpolation carve-outs), so
there is nothing to compare. That is an interpolation gap, not a curve gap.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.interpolations.log_linear import LogLinearInterpolation
from pquantlib.termstructures.yield_.interpolated_discount_curve import (
    InterpolatedDiscountCurve,
)
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
    / "migration-harness/references/v143/ts/discountcurve.json"
)

_REF_DATE = Date.from_ymd(17, Month.January, 2024)
_DISCOUNTS = [1.0, 0.985, 0.968, 0.930, 0.815, 0.640]

_INTERPOLATORS: dict[str, Callable[[Array, Array], Interpolation]] = {
    "log_linear": LogLinearInterpolation,
    "linear": LinearInterpolation,
    "cubic_natural": CubicNaturalSpline,
}


def _dates() -> list[Date]:
    return [
        _REF_DATE,
        _REF_DATE + Period(6, TimeUnit.Months),
        _REF_DATE + Period(1, TimeUnit.Years),
        _REF_DATE + Period(2, TimeUnit.Years),
        _REF_DATE + Period(5, TimeUnit.Years),
        _REF_DATE + Period(10, TimeUnit.Years),
    ]


def _curve(key: str) -> InterpolatedDiscountCurve:
    curve = InterpolatedDiscountCurve(
        _dates(),
        _DISCOUNTS,
        Actual365Fixed(),
        calendar=TARGET(),
        interpolator=_INTERPOLATORS[key],
    )
    curve.enable_extrapolation()
    return curve


_Refs = tuple[list[float], dict[str, dict[str, list[float]]]]


@pytest.fixture(scope="module")
def refs() -> _Refs:
    raw = cast("dict[str, object]", json.loads(_REF_PATH.read_text()))
    times = cast("list[float]", raw["times"])
    curves = {
        k: cast("dict[str, list[float]]", v) for k, v in raw.items() if k != "times"
    }
    return times, curves


@pytest.mark.parametrize("key", sorted(_INTERPOLATORS))
def test_discount_matches_cpp(refs: _Refs, key: str) -> None:
    times, curves = refs
    curve = _curve(key)
    for t, expected in zip(times, curves[key]["discount"], strict=True):
        tolerance.tight(curve.discount(t, extrapolate=True), expected, reason=f"{key}@t={t}")


# Why the finite-differenced quantities below are LOOSE and not TIGHT.
#
# ``zeroRate(0)`` and ``forwardRate(t, t)`` are not read off the curve: C++
# evaluates them by finite difference over ``dt = 1e-4``
# (yieldtermstructure.cpp anonymous namespace), and the port does the same.
# rate = ln(compound) / dt with compound = discount(t1)/discount(t2) ~ 1. Each
# discount carries a few ulp of rounding, so compound carries a relative error
# of order 5e-16; since compound ~ 1, that is also the ABSOLUTE error of
# ln(compound). Dividing by dt = 1e-4 multiplies it by 1e4, giving an absolute
# rate error ~5e-12 and, at a rate of ~4%, a relative error ~1e-10 — an order
# of magnitude past the TIGHT tier's 1e-12, purely from the differencing.
# Observed disagreement is ~2e-11 relative, consistent with that bound. LOOSE
# (1e-8) sits comfortably above the noise and still an order of magnitude below
# any behavioural difference: an interpolated (rather than flat-forward)
# extrapolation moves the 25-year forward by ~1e-3.
@pytest.mark.parametrize("key", sorted(_INTERPOLATORS))
def test_zero_rate_matches_cpp(refs: _Refs, key: str) -> None:
    times, curves = refs
    curve = _curve(key)
    for t, expected in zip(times, curves[key]["zero"], strict=True):
        rate = curve.zero_rate(
            t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
        ).rate()
        # t == 0 is the only finite-differenced entry; everything else is read
        # straight off the curve and holds at TIGHT.
        check = tolerance.loose if t == 0.0 else tolerance.tight
        check(rate, expected, reason=f"{key}@t={t}")


@pytest.mark.parametrize("key", sorted(_INTERPOLATORS))
def test_instantaneous_forward_matches_cpp(refs: _Refs, key: str) -> None:
    times, curves = refs
    curve = _curve(key)
    for t, expected in zip(times, curves[key]["inst_forward"], strict=True):
        rate = curve.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
        ).rate()
        tolerance.loose(rate, expected, reason=f"{key}@t={t}")


@pytest.mark.parametrize("key", sorted(_INTERPOLATORS))
def test_forward_is_flat_past_the_last_node(key: str) -> None:
    """The discriminating property: extrapolation is flat-forward, not interpolated.

    Only holds if ``discount_impl`` takes the explicit exponential branch. An
    interpolating extrapolation gives a t-dependent forward for every
    interpolator except LogLinear — for ``cubic_natural`` the 25-year forward
    would move by ~1e-3, which is five orders of magnitude above the LOOSE
    bound used here (see the note above for why LOOSE and not TIGHT).
    """
    curve = _curve(key)
    t_max = curve.max_time()
    forwards = [
        curve.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
        ).rate()
        for t in (t_max + 1.0, t_max + 5.0, t_max + 15.0)
    ]
    for f in forwards[1:]:
        tolerance.loose(f, forwards[0], reason=key)
