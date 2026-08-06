"""ImpliedVolTermStructure, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/eqfx/impliedvol.json
Probe:     migration-harness/cpp/probes/v143_eqfx_impliedvol/probe.cpp

The probe re-anchors one BlackVarianceSurface (and one BlackVarianceCurve) at
four future dates: shift == 0, shift on a quoted pillar, shift between
pillars, and shift past the original's max date (where ``max_time()`` goes
negative). A port that dropped the date shift would reproduce only the
``surface_shift_zero`` block.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_variance_surface import (
    BlackVarianceSurface,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.volatility.equity_fx.implied_vol_term_structure import (
    ImpliedVolTermStructure,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("v143/eqfx/impliedvol")

_DC = Actual365Fixed()
_REF_DATE = Date.from_ymd(15, Month.June, 2026)
_DATES = [
    Date.from_ymd(15, Month.September, 2026),
    Date.from_ymd(15, Month.December, 2026),
    Date.from_ymd(15, Month.June, 2027),
    Date.from_ymd(15, Month.June, 2028),
]
_STRIKES = [80.0, 100.0, 120.0]
_VOL_MATRIX = np.asarray(
    [
        [0.20, 0.21, 0.22, 0.23],
        [0.10, 0.15, 0.20, 0.25],
        [0.20, 0.21, 0.22, 0.23],
    ],
    dtype=np.float64,
)
_CURVE_VOLS = [0.13, 0.16, 0.20, 0.25]

# Mirrors the probe's kTimes / kQueryStrikes label tables.
_TIMES: list[tuple[str, float]] = [
    ("t0p3", 0.3),
    ("t0p75", 0.75),
    ("t1p5", 1.5),
    ("t3p0", 3.0),
]
_QUERY_STRIKES: list[tuple[str, float]] = [
    ("k70", 70.0),
    ("k100", 100.0),
    ("k130", 130.0),
]


def _surface() -> BlackVarianceSurface:
    return BlackVarianceSurface(
        reference_date=_REF_DATE,
        calendar=NullCalendar(),
        dates=_DATES,
        strikes=_STRIKES,
        black_vol_matrix=_VOL_MATRIX,
        day_counter=_DC,
    )


def _curve() -> BlackVarianceCurve:
    return BlackVarianceCurve(
        reference_date=_REF_DATE,
        dates=_DATES,
        black_vol_curve=_CURVE_VOLS,
        day_counter=_DC,
        force_monotone_variance=True,
    )


_CASES: dict[str, tuple[str, Date]] = {
    "surface_shift_zero": ("surface", _REF_DATE),
    "surface_shift_on_pillar": ("surface", Date.from_ymd(15, Month.December, 2026)),
    "surface_shift_between_pillars": ("surface", Date.from_ymd(3, Month.March, 2027)),
    "surface_shift_past_max_date": ("surface", Date.from_ymd(15, Month.December, 2028)),
    "curve_shift_between_pillars": ("curve", Date.from_ymd(3, Month.March, 2027)),
}


def _implied(case: str) -> ImpliedVolTermStructure:
    kind, shifted = _CASES[case]
    original: BlackVolTermStructure = _surface() if kind == "surface" else _curve()
    return ImpliedVolTermStructure(original_ts=original, reference_date=shifted)


def _case_ids() -> Sequence[str]:
    return list(_CASES)


def _assert_strike_bound(actual: float, expected: float) -> None:
    """Compare a strike bound, allowing the port's unboundedness sentinel.

    C++ spells "no strike bound" as ``QL_MIN_REAL`` / ``QL_MAX_REAL``
    (``-+std::numeric_limits<Real>::max()``); PQuantLib spells it ``-inf`` /
    ``+inf`` throughout the volatility package (see the note in
    ``flat_smile_section.py``). The two sentinels mean the same thing to
    ``check_strike``, so the comparison accepts either — but only for the
    sentinel: any finite bound must match exactly.
    """
    if abs(expected) == sys.float_info.max:
        assert math.isinf(actual) or actual == expected
        assert (actual > 0) == (expected > 0)
        return
    tolerance.tight(actual, expected)


@pytest.mark.parametrize("case", _case_ids())
def test_dates_and_strike_range(case: str) -> None:
    ivts = _implied(case)
    expected = _REF[case]
    assert ivts.reference_date().serial_number() == expected["reference_date"]
    assert ivts.max_date().serial_number() == expected["max_date"]
    tolerance.tight(ivts.max_time(), expected["max_time"])
    _assert_strike_bound(ivts.min_strike(), expected["min_strike"])
    _assert_strike_bound(ivts.max_strike(), expected["max_strike"])


@pytest.mark.parametrize("case", _case_ids())
def test_black_variance_grid(case: str) -> None:
    ivts = _implied(case)
    expected = _REF[case]["variance"]
    for t_label, t in _TIMES:
        for k_label, k in _QUERY_STRIKES:
            tolerance.tight(
                ivts.black_variance_at_time(t, k, extrapolate=True),
                expected[f"{t_label}_{k_label}"],
            )


@pytest.mark.parametrize("case", _case_ids())
def test_black_vol_grid(case: str) -> None:
    ivts = _implied(case)
    expected = _REF[case]["vol"]
    for t_label, t in _TIMES:
        for k_label, k in _QUERY_STRIKES:
            tolerance.tight(
                ivts.black_vol_at_time(t, k, extrapolate=True),
                expected[f"{t_label}_{k_label}"],
            )


@pytest.mark.parametrize("case", _case_ids())
def test_black_forward_vol(case: str) -> None:
    ivts = _implied(case)
    expected = _REF[case]["forward_vol"]
    for i in range(len(_TIMES) - 1):
        t1_label, t1 = _TIMES[i]
        t2_label, t2 = _TIMES[i + 1]
        tolerance.tight(
            ivts.black_forward_vol_at_time(t1, t2, 100.0, extrapolate=True),
            expected[f"{t1_label}_to_{t2_label}_k100"],
        )
    # Degenerate t1 == t2: falls back to a central finite difference of
    # variance with eps = min(1e-5, t1) = 1e-5, i.e.
    #     fwd = sqrt((v(t+eps) - v(t-eps)) / (2 eps)).
    #
    # That subtraction is catastrophically cancelling and no port can be held
    # to the TIGHT tier through it. Bound, from the arithmetic: each variance
    # carries at most one ulp of representation error, |dv| <= v * 2^-52, so
    #     |d(fwd)/fwd| = 0.5 * |d(v+ - v-)| / (v+ - v-)
    #                 <= 0.5 * 2 * v * 2^-52 / (2 * eps * fwd^2)
    #                  = v * 2^-52 / (2 eps fwd^2).
    # Here v ~ 0.056, fwd ~ 0.29 (fwd^2 ~ 0.085), eps = 1e-5, giving
    # ~7.3e-12 — the observed gap on the BlackVarianceCurve case is 4.1e-12,
    # inside it. rel_tol is set to 1e-10, an order of magnitude above the
    # bound and seven orders below the ~1% error a wrong date shift produces,
    # so the case still discriminates.
    tolerance.custom(
        ivts.black_forward_vol_at_time(0.75, 0.75, 100.0, extrapolate=True),
        expected["t0p75_to_t0p75_k100"],
        abs_tol=1e-14,
        rel_tol=1e-10,
        reason=(
            "central finite difference over 2*eps=2e-5 amplifies one ulp of "
            "variance to ~7e-12 relative; see derivation above"
        ),
    )


@pytest.mark.parametrize("case", _case_ids())
def test_black_forward_variance(case: str) -> None:
    ivts = _implied(case)
    expected = _REF[case]["forward_variance"]
    for i in range(len(_TIMES) - 1):
        t1_label, t1 = _TIMES[i]
        t2_label, t2 = _TIMES[i + 1]
        tolerance.tight(
            ivts.black_forward_variance_at_time(t1, t2, 100.0, extrapolate=True),
            expected[f"{t1_label}_to_{t2_label}_k100"],
        )


@pytest.mark.parametrize("case", _case_ids())
def test_date_anchored_query(case: str) -> None:
    ivts = _implied(case)
    expected = _REF[case]["vol_by_date"]
    d = ivts.reference_date() + 200
    assert d.serial_number() == expected["date"]
    tolerance.tight(ivts.black_vol(d, 100.0, extrapolate=True), expected["k100"])
    tolerance.tight(
        ivts.black_variance(d, 100.0, extrapolate=True), expected["variance_k100"]
    )


def test_shift_zero_reproduces_the_original_surface() -> None:
    """With a zero shift the implied structure IS the original surface.

    Not read from the reference: it is a self-consistency property that the
    probe's ``surface_shift_zero`` block already encodes, restated here so a
    regression names the cause rather than a numeric mismatch.
    """
    original = _surface()
    ivts = ImpliedVolTermStructure(original_ts=original, reference_date=_REF_DATE)
    for _, t in _TIMES:
        for _, k in _QUERY_STRIKES:
            tolerance.tight(
                ivts.black_variance_at_time(t, k, extrapolate=True),
                original.black_variance_at_time(t, k, extrapolate=True),
            )


def test_day_counter_is_delegated() -> None:
    original = _surface()
    ivts = ImpliedVolTermStructure(
        original_ts=original, reference_date=Date.from_ymd(3, Month.March, 2027)
    )
    assert ivts.day_counter().name() == original.day_counter().name()
