"""NoExceptLocalVolSurface, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/eqfx/localvol.json
Probe:     migration-harness/cpp/probes/v143_eqfx_localvol/probe.cpp

Two things have to hold and both are pinned: the wrapper returns the overwrite
value at exactly the points where the plain surface raises (and nowhere else),
and everywhere else it is inert — every value identical to the plain surface,
bit for bit.

Note the second block does NOT use a surface on which Dupire always succeeds:
even the smooth-vol surface produces a negative local variance at
``(t = 1.5, S = 100)`` and ``(t = 2.5, S = 100)`` under r = 5% / q = 2%, and
C++ substitutes the overwrite there too. That is a property of the Dupire
denominator rather than of the fallback, and it makes the block a better test
of "substitutes exactly where the plain surface fails" than a surface with no
failures would be.
"""

from __future__ import annotations

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
from pquantlib.termstructures.volatility.equity_fx.no_except_local_vol_surface import (
    NoExceptLocalVolSurface,
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
_NON_MONOTONE_VOLS = np.asarray(
    [
        [0.40, 0.15, 0.14, 0.13],
        [0.20, 0.21, 0.22, 0.23],
        [0.20, 0.21, 0.22, 0.23],
    ],
    dtype=np.float64,
)

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


# See the tolerance derivation in test_local_vol_surface_yield_curves.py:
# Dupire's second strike difference over dy ~ 3e-5 amplifies one ulp of
# variance by ~1e7, giving a derived budget of 1.8e-6 relative on this grid;
# the observed C++/Python disagreement is 7.0e-8. The overwrite value itself
# is compared exactly — it goes through no arithmetic at all.
_REL_TOL = 5e-6
_ABS_TOL = 1e-14
_TOL_REASON = (
    "Dupire's second strike difference amplifies one ulp of variance by ~1e7; "
    "derived budget 1.8e-6, observed 7.0e-8 (see "
    "test_local_vol_surface_yield_curves.py)"
)


def _check_grid(
    surface: NoExceptLocalVolSurface, expected: dict[str, Any], prefix: str
) -> None:
    for t_label, t in _TIMES:
        for s_label, s in _LEVELS:
            key = f"{t_label}_{s_label}"
            # The wrapper never raises — every reference entry must be a number.
            assert not isinstance(expected[key], dict), (
                f"{prefix}/{key}: NoExceptLocalVolSurface should not raise"
            )
            tolerance.custom(
                surface.local_vol_at_time(t, s, extrapolate=True),
                float(expected[key]),
                abs_tol=_ABS_TOL,
                rel_tol=_REL_TOL,
                reason=f"{prefix}/{key}: {_TOL_REASON}",
            )


def test_overwrite_replaces_exactly_the_failing_points() -> None:
    overwrite = float(_REF["non_monotone_noexcept"]["overwrite"])
    plain_ref = _REF["non_monotone_plain"]["local_vol"]
    noexcept_ref = _REF["non_monotone_noexcept"]["local_vol"]

    noexc = NoExceptLocalVolSurface(
        black_ts=_surface(_NON_MONOTONE_VOLS),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.02),
        underlying=SimpleQuote(100.0),
        illegal_local_vol_overwrite=overwrite,
    )
    _check_grid(noexc, noexcept_ref, "non_monotone_noexcept")

    # And the substitution happened at the failing points and only there.
    failing = {k for k, v in plain_ref.items() if isinstance(v, dict)}
    assert failing, "reference has no failing point; the case stopped discriminating"
    for key in failing:
        tolerance.exact(float(noexcept_ref[key]), overwrite)
    for key, value in plain_ref.items():
        if key not in failing:
            tolerance.exact(float(noexcept_ref[key]), float(value))


def test_inert_wherever_the_plain_surface_returns() -> None:
    overwrite = float(_REF["non_monotone_noexcept"]["overwrite"])
    noexc = NoExceptLocalVolSurface(
        black_ts=_surface(_GOOD_VOLS),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.02),
        underlying=100.0,
        illegal_local_vol_overwrite=overwrite,
    )
    _check_grid(
        noexc, _REF["good_noexcept_real_underlying"]["local_vol"], "good_noexcept"
    )

    # Same points through the plain surface: bit-identical where it returns,
    # the overwrite exactly where it raises.
    plain = LocalVolSurface(
        black_ts=_surface(_GOOD_VOLS),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.02),
        underlying=100.0,
    )
    substituted = 0
    for _, t in _TIMES:
        for _, s in _LEVELS:
            wrapped = noexc.local_vol_at_time(t, s, extrapolate=True)
            try:
                tolerance.exact(
                    wrapped, plain.local_vol_at_time(t, s, extrapolate=True)
                )
            except LibraryException:
                tolerance.exact(wrapped, overwrite)
                substituted += 1
    # The smooth surface still fails at (1.5, 100) and (2.5, 100); if that ever
    # stops being true the block no longer tests the substitution path.
    assert substituted == 2


def test_it_does_not_swallow_non_library_errors() -> None:
    """C++ catches ``Error&``, not ``...`` — a non-QuantLib failure escapes.

    A zero underlying makes ``log(strike / forward)`` divide by zero, which is
    a ``ZeroDivisionError`` in Python (and not a QuantLib ``Error`` in C++), so
    it must propagate rather than be replaced by the overwrite.
    """
    noexc = NoExceptLocalVolSurface(
        black_ts=_surface(_GOOD_VOLS),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.02),
        underlying=0.0,
        illegal_local_vol_overwrite=0.123456789,
    )
    with pytest.raises((ZeroDivisionError, ValueError)):
        noexc.local_vol_at_time(0.75, 100.0, extrapolate=True)


def test_library_exception_is_what_gets_caught() -> None:
    """Sanity: the plain surface raises LibraryException at a failing point."""
    plain = LocalVolSurface(
        black_ts=_surface(_NON_MONOTONE_VOLS),
        risk_free_ts=_flat(0.05),
        dividend_ts=_flat(0.02),
        underlying=SimpleQuote(100.0),
    )
    with pytest.raises(LibraryException):
        plain.local_vol_at_time(0.3, 80.0, extrapolate=True)
