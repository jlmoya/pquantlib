"""LocalVolSurface over NON-FLAT yield curves, cross-validated against v1.43.

Reference: migration-harness/references/v143/eqfx/localvol.json
Probe:     migration-harness/cpp/probes/v143_eqfx_localvol/probe.cpp

The yield curves enter ``localVolImpl`` twice — through the forward
``F(t) = S dq(t)/dr(t)`` that defines the log-moneyness, and through the
strike drift ``K exp(+-(r - q) dt)`` along which the time derivative is
taken. Both collapse when ``r == q == 0``, which is why the earlier port
(which hard-coded zero rates) matched its zero-rate reference exactly while
being wrong for every real curve. The ``zero_rates`` block below is that old
behaviour, kept as a regression guard; the other three blocks are the ones
that would have caught it.

``test_local_vol_surface.py`` covers the same class against the older L2-E
probe; this file is specifically about the curve dependence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_variance_surface import (
    BlackVarianceSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_surface import (
    LocalVolSurface,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

_REF = reference_reader.load("v143/eqfx/localvol")

_DC = Actual365Fixed()
_REF_DATE = Date.from_ymd(15, Month.June, 2026)
_DATES = [
    Date.from_ymd(15, Month.September, 2026),
    Date.from_ymd(15, Month.December, 2026),
    Date.from_ymd(15, Month.June, 2027),
    Date.from_ymd(15, Month.June, 2028),
]
_STRIKES = [80.0, 100.0, 120.0]

_GOOD_VOLS = np.asarray(
    [
        [0.20, 0.21, 0.22, 0.23],
        [0.10, 0.15, 0.20, 0.25],
        [0.20, 0.21, 0.22, 0.23],
    ],
    dtype=np.float64,
)
# Low strike's variance collapses between the first two pillars, so the
# non-decreasing-variance guard fires near S = 80.
_NON_MONOTONE_VOLS = np.asarray(
    [
        [0.40, 0.15, 0.14, 0.13],
        [0.20, 0.21, 0.22, 0.23],
        [0.20, 0.21, 0.22, 0.23],
    ],
    dtype=np.float64,
)

# Mirrors the probe's kTimes / kLevels label tables.
_TIMES: list[tuple[str, float]] = [
    ("t0", 0.0),
    ("t0p0001", 0.0001),
    ("t0p3", 0.3),
    ("t0p75", 0.75),
    ("t1p5", 1.5),
    ("t2p5_past_max", 2.5),
]
_LEVELS: list[tuple[str, float]] = [
    ("s70", 70.0),
    ("s80", 80.0),
    ("s100", 100.0),
    ("s120", 120.0),
    ("s130", 130.0),
]

_RATE_CASES: dict[str, tuple[float, float]] = {
    "zero_rates": (0.00, 0.00),
    "r5_q0": (0.05, 0.00),
    "r5_q2": (0.05, 0.02),
    "r2_q5": (0.02, 0.05),
}

# Tolerance derivation — why neither TIGHT nor LOOSE fits here.
#
# Dupire's denominator needs the SECOND strike difference
#     d2w/dy2 = (w+ - 2w + w-) / dy^2,  dy = 1e-4 * |y|
# For the probed grid dy is ~3e-5, so dy^2 ~ 1e-9 while the variances
# themselves are ~1e-2: the numerator cancels about seven decimal digits, and
# one ulp of input error is amplified by ~1e7 in d2w/dy2 — and from there into
# the result. Measured on this exact grid (perturbing each of w, w+, w- by
# +-1 ulp and taking the spread of the resulting local vol) the worst-case
# budget is 1.8e-6 relative, at (r5_q0, t=2.5, S=120).
#
# The inputs DO differ by an ulp between the two languages and cannot be made
# not to: clang contracts the bilinear interpolation's
# ``(1-t)(1-u)z1 + t(1-u)z2 + tu z3 + (1-t)u z4`` into FMAs on arm64, which
# numpy does not do. Observed worst-case disagreement over every block of this
# reference is 7.0e-8 relative — well inside the 1.8e-6 budget.
#
# rel_tol is therefore set to 5e-6: above the derived budget, and ~2000x below
# the ~1e-2 signal that separates the rate cases from each other (see
# ``test_non_flat_curves_actually_move_the_answer``), so it still catches a
# port that ignores the yield curves. ``test_the_second_difference_is_ill_
# conditioned`` demonstrates the amplification rather than asserting it.
_REL_TOL = 5e-6
_ABS_TOL = 1e-14
_TOL_REASON = (
    "Dupire's second strike difference over dy~3e-5 amplifies one ulp of "
    "variance by ~1e7; derived budget 1.8e-6, observed 7.0e-8"
)


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(
        _REF_DATE, rate, _DC, Compounding.Continuous, Frequency.Annual
    )


def _surface(vols: np.ndarray[Any, np.dtype[np.float64]]) -> BlackVarianceSurface:
    return BlackVarianceSurface(
        reference_date=_REF_DATE,
        calendar=NullCalendar(),
        dates=_DATES,
        strikes=_STRIKES,
        black_vol_matrix=vols,
        day_counter=_DC,
    )


def _assert_matches(expected: Any, call: Callable[[], float], *, where: str) -> None:
    if isinstance(expected, dict):
        assert expected == {"raises": True}, f"unexpected sentinel at {where}"
        with pytest.raises(LibraryException):
            call()
        return
    tolerance.custom(
        call(),
        float(expected),
        abs_tol=_ABS_TOL,
        rel_tol=_REL_TOL,
        reason=f"{where}: {_TOL_REASON}",
    )


def _check_grid(lvs: LocalVolSurface, expected: dict[str, Any], prefix: str) -> None:
    for t_label, t in _TIMES:
        for s_label, s in _LEVELS:
            key = f"{t_label}_{s_label}"
            _assert_matches(
                expected[key],
                lambda t=t, s=s: lvs.local_vol_at_time(t, s, extrapolate=True),
                where=f"{prefix}/{key}",
            )


@pytest.mark.parametrize("case", list(_RATE_CASES))
def test_delegated_accessors(case: str) -> None:
    r, q = _RATE_CASES[case]
    lvs = LocalVolSurface(
        black_ts=_surface(_GOOD_VOLS),
        risk_free_ts=_flat(r),
        dividend_ts=_flat(q),
        underlying=SimpleQuote(100.0),
    )
    expected = _REF["surface"][case]
    assert lvs.reference_date().serial_number() == expected["reference_date"]
    assert lvs.max_date().serial_number() == expected["max_date"]
    tolerance.tight(lvs.min_strike(), expected["min_strike"])
    tolerance.tight(lvs.max_strike(), expected["max_strike"])


@pytest.mark.parametrize("case", list(_RATE_CASES))
def test_local_vol_grid(case: str) -> None:
    r, q = _RATE_CASES[case]
    lvs = LocalVolSurface(
        black_ts=_surface(_GOOD_VOLS),
        risk_free_ts=_flat(r),
        dividend_ts=_flat(q),
        underlying=SimpleQuote(100.0),
    )
    _check_grid(lvs, _REF["surface"][case]["local_vol"], f"surface/{case}")


def test_real_underlying_overload_matches_quote_overload() -> None:
    """The ``Real underlying`` constructor must agree with the Quote one."""
    lvs = LocalVolSurface(
        black_ts=_surface(_GOOD_VOLS),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.02),
        underlying=100.0,
    )
    _check_grid(
        lvs,
        _REF["surface_real_underlying_r5_q2"]["local_vol"],
        "surface_real_underlying_r5_q2",
    )


def test_non_flat_curves_actually_move_the_answer() -> None:
    """Guard the discriminating power of the reference itself.

    Every block here would agree if the port ignored the curves, so assert
    the reference separates ``zero_rates`` from the rest by far more than
    rounding. The largest movers are the wing levels, where the forward
    shift changes the log-moneyness the most.
    """
    zero = _REF["surface"]["zero_rates"]["local_vol"]
    shifted = _REF["surface"]["r2_q5"]["local_vol"]
    moved = [
        abs(shifted[k] - zero[k]) / abs(zero[k])
        for k in zero
        if not isinstance(zero[k], dict) and zero[k] != 0.0
    ]
    assert max(moved) > 0.01


def test_the_second_difference_is_ill_conditioned() -> None:
    """Show the amplification the custom tolerance above is derived from.

    Nudge the Black variance the surface reports by one ulp and watch the
    local vol move by far more than the TIGHT tier allows. This uses the real
    ``LocalVolSurface`` — it is the class's own conditioning being measured,
    not a re-implementation of its formula.
    """

    class _NudgedSurface(BlackVarianceSurface):
        """Reports the CENTRE variance one ulp high, the wings untouched.

        Only the centre is nudged, because a uniform shift of all three
        samples cancels out of ``w+ - 2w + w-`` and would measure nothing.
        The centre is the only query at a whole-number strike: the wings sit
        at ``K exp(+-dy)`` and the time-derivative samples at
        ``K exp(+-(r - q) dt)``, none of which are integers.
        """

        def _black_variance_impl(self, t: float, strike: float) -> float:
            value = super()._black_variance_impl(t, strike)
            if strike == round(strike):
                return math.nextafter(value, math.inf)
            return value

    base = LocalVolSurface(
        black_ts=_surface(_GOOD_VOLS),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.0),
        underlying=SimpleQuote(100.0),
    )
    nudged = LocalVolSurface(
        black_ts=_NudgedSurface(
            reference_date=_REF_DATE,
            calendar=NullCalendar(),
            dates=_DATES,
            strikes=_STRIKES,
            black_vol_matrix=_GOOD_VOLS,
            day_counter=_DC,
        ),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.0),
        underlying=SimpleQuote(100.0),
    )
    a = base.local_vol_at_time(2.5, 120.0, extrapolate=True)
    b = nudged.local_vol_at_time(2.5, 120.0, extrapolate=True)
    moved = abs(a - b) / a
    assert moved > 1e-12, (
        "one ulp of variance no longer perturbs the local vol beyond the "
        "TIGHT tier; the custom tolerance in this module may no longer be "
        "justified"
    )
    assert moved < _REL_TOL, (
        "one ulp of variance now perturbs the local vol beyond the tolerance "
        "this module allows; the derived budget needs revisiting"
    )


def test_non_monotone_surface_raises_where_cpp_raises() -> None:
    """Plain LocalVolSurface propagates the non-decreasing-variance guard."""
    lvs = LocalVolSurface(
        black_ts=_surface(_NON_MONOTONE_VOLS),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.02),
        underlying=SimpleQuote(100.0),
    )
    expected = _REF["non_monotone_plain"]["local_vol"]
    assert any(isinstance(v, dict) for v in expected.values()), (
        "reference no longer contains a failing point; the case stopped "
        "discriminating"
    )
    _check_grid(lvs, expected, "non_monotone_plain")
