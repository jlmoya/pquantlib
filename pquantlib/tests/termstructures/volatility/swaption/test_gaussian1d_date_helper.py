"""``Gaussian1dSwaptionVolatility.DateHelper`` against C++ v1.43.

Reference: ``migration-harness/references/v143/ts/datehelper.json``
(probe ``migration-harness/cpp/probes/v143_ts_datehelper/probe.cpp``).

``DateHelper`` (gaussian1dswaptionvolatility.hpp:70-86) exists for exactly one
job: to let ``NewtonSafe`` invert ``TermStructure::timeFromReference`` and
recover the DATE whose year fraction equals a given option TIME
(gaussian1dswaptionvolatility.cpp:46-53).

**Why this is not cosmetic.** The C++ call passes ``365.25 * optionTime +
referenceDate().serialNumber()`` as the solver's GUESS. PQuantLib previously
used that expression as the ANSWER, with a comment saying no inversion was
needed because "PQuantLib's call sites use Date directly". Under
Actual/365Fixed the true root is ``ref + 365 * optionTime``, so the guess is
already a day out at optionTime = 4 and five days out at optionTime = 20; the
sign of the error flips under Thirty360. Each case below carries both the
guess serial and the solved serial so the divergence is visible in the data,
not just in prose.

The probe pins four day counters x nine option times, plus the raw functor and
its forward-difference derivative at fractional serials.

The probe sets an evaluation date (``probe.cpp:127`` — 2024-01-15) because
``TermStructure::referenceDate()`` reads it; the fixture below pins the same
date and restores the previous value.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as ActualActualConvention
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.math.solvers1d.newton_safe import NewtonSafe
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.volatility.swaption.gaussian1d_swaption_volatility import (
    DateHelper,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month

CPP: dict[str, Any] = reference_reader.load("v143/ts/datehelper")

# probe.cpp:127 — Date EVAL_DATE(15, January, 2024)
_TODAY = Date.from_ymd(15, Month.January, 2024)

_DAY_COUNTERS: dict[str, DayCounter] = {
    "Actual365Fixed": Actual365Fixed(),
    "Actual360": Actual360(),
    "ActualActualISDA": ActualActual(ActualActualConvention.ISDA),
    "Thirty360BondBasis": Thirty360(Thirty360Convention.BondBasis),
}


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:
    s = ObservableSettings()
    prev = s.evaluation_date
    s.evaluation_date = _TODAY  # probe.cpp:127
    yield
    s.evaluation_date = prev


def _curve(name: str) -> FlatForward:
    # probe.cpp:157 — FlatForward(EVAL_DATE, SimpleQuote(0.03), dc)
    return FlatForward.from_rate(_TODAY, 0.03, _DAY_COUNTERS[name])


def test_reference_is_v143() -> None:
    assert CPP["quantlib_version"] == "1.43"


def test_eval_date_matches_the_probe() -> None:
    assert _TODAY.serial_number() == CPP["eval_date_serial"]


def _ids(block: str) -> list[str]:
    if block == "date_helper":
        return [
            f'{c["day_counter"]}-{c["option_time"]}' for c in CPP["date_helper"]
        ]
    return [str(c["offset"]) for c in CPP["date_helper_raw"]]


# --- the raw functor --------------------------------------------------------


@pytest.mark.parametrize(
    "case", CPP["date_helper_raw"], ids=_ids("date_helper_raw")
)
def test_date_helper_value_matches_cpp(case: dict[str, Any]) -> None:
    """``f(x)`` at integer AND fractional serials.

    The fractional cases are the point: C++ interpolates linearly between
    ``T(floor(x))`` and ``T(floor(x)+1)``, so a port that simply calls
    ``timeFromReference(Date(int(x)))`` agrees at every integer and disagrees
    everywhere else.
    """
    helper = DateHelper(_curve("Actual365Fixed"), case["t"])
    tolerance.tight(helper(case["x"]), case["value"])


@pytest.mark.parametrize(
    "case", CPP["date_helper_raw"], ids=_ids("date_helper_raw")
)
def test_date_helper_derivative_matches_cpp(case: dict[str, Any]) -> None:
    """Forward difference with step ``1e-6`` — hpp:81-85, verbatim.

    Not the analytic slope: the step size is observable, because ``NewtonSafe``
    decides between a Newton step and a bisection from it.

    TOLERANCE, derived rather than chosen. ``derivative`` is
    ``(f(x + 1e-6) - f(x)) * 1e6``, a difference of two nearly equal values.
    With ``t = 1`` and offset 100.75, ``f(x) = -0.7239726...`` while
    ``f(x + 1e-6) - f(x) = 2.7397e-9`` — a cancellation of about 8.4 decimal
    digits. A ONE-ULP difference in either operand, which is all it takes for
    two implementations to order the same additions differently, propagates to

        rel = ulp(f(x)) / |f(x+h) - f(x)|
            = 2^-53 * |f(x)| / (|derivative| * 1e-6)

    which for that case is 1.11e-16 * 0.72397 / 2.7397e-9 = 4.05e-8. The
    measured C++/Python gap on that case is 4.05e-8 — the one-ULP bound
    exactly, not an accumulation. The bound below allows 4 ULP; every other
    case, where ``f(x)`` is small and there is no cancellation, lands inside
    TIGHT and is checked at TIGHT.
    """
    helper = DateHelper(_curve("Actual365Fixed"), case["t"])
    got = helper.derivative(case["x"])
    expected = case["derivative"]
    delta = abs(expected) * 1e-6  # |f(x+h) - f(x)|
    one_ulp_rel = (
        2.0**-53 * abs(case["value"]) / delta if delta > 0.0 else 0.0
    )
    if one_ulp_rel <= 1e-12:
        tolerance.tight(got, expected)
        return
    tolerance.custom(
        got, expected, abs_tol=0.0, rel_tol=4.0 * one_ulp_rel,
        reason=(
            f"forward difference cancels {abs(case['value']):.4g} down to "
            f"{delta:.4g}; 1-ULP floor is {one_ulp_rel:.3g} relative"
        ),
    )


def test_derivative_is_the_forward_difference_not_the_exact_slope() -> None:
    """Pin the 1e-6 forward difference apart from an exact/central derivative.

    Under Actual/365Fixed the exact slope is 1/365 everywhere, and the forward
    difference over a piecewise-linear function reproduces it away from a knot
    — but AT an integer serial the forward step crosses into the next segment,
    and C++'s asymmetric difference is what a central difference would not
    give. The probe's ``offset = 0.0`` case sits exactly on a knot.
    """
    on_knot = [c for c in CPP["date_helper_raw"] if c["offset"] == 0.0]
    assert on_knot, "probe must include a case sitting on an integer serial"
    case = on_knot[0]
    helper = DateHelper(_curve("Actual365Fixed"), case["t"])
    fwd = helper.derivative(case["x"])
    central = (helper(case["x"] + 5e-7) - helper(case["x"] - 5e-7)) * 1e6
    tolerance.tight(fwd, case["derivative"])
    assert fwd != pytest.approx(central, abs=1e-12)


# --- the NewtonSafe inversion ----------------------------------------------


@pytest.mark.parametrize("case", CPP["date_helper"], ids=_ids("date_helper"))
def test_newton_safe_inversion_lands_on_the_cpp_date(case: dict[str, Any]) -> None:
    """The whole composite: DateHelper + NewtonSafe + serial truncation.

    # C++ parity: gaussian1dswaptionvolatility.cpp:48-53. Dates are integers,
    # so the tier is EXACT — a one-day miss is a failure, not a tolerance
    # question.
    """
    curve = _curve(case["day_counter"])
    helper = DateHelper(curve, case["option_time"])
    guess = 365.25 * case["option_time"] + float(_TODAY.serial_number())
    tolerance.tight(guess, case["guess"])
    solved = NewtonSafe().solve(helper, 0.1, guess, 1.0)
    assert int(solved) == case["solved_serial"]
    assert Date(int(solved)).serial_number() == case["solved_serial"]


@pytest.mark.parametrize("case", CPP["date_helper"], ids=_ids("date_helper"))
def test_calendar_adjusted_date_matches_cpp(case: dict[str, Any]) -> None:
    """``indexBase_->fixingCalendar().adjust(d)`` — cpp:57, on a TARGET calendar."""
    curve = _curve(case["day_counter"])
    helper = DateHelper(curve, case["option_time"])
    guess = 365.25 * case["option_time"] + float(_TODAY.serial_number())
    solved = NewtonSafe().solve(helper, 0.1, guess, 1.0)
    adjusted = TARGET().adjust(Date(int(solved)))
    assert adjusted.serial_number() == case["adjusted_serial"]


@pytest.mark.parametrize("case", CPP["date_helper"], ids=_ids("date_helper"))
def test_time_from_reference_at_the_solved_date_matches_cpp(
    case: dict[str, Any],
) -> None:
    """The root really is the date whose year fraction is ``option_time``.

    (Up to the one-day truncation C++ applies afterwards — which is why the
    probe emits the value rather than asserting it equals ``option_time``.)
    """
    curve = _curve(case["day_counter"])
    tolerance.tight(
        curve.time_from_reference(Date(case["solved_serial"])),
        case["time_at_solved_date"],
    )


def test_the_guess_is_not_the_answer() -> None:
    """The defect this module exists to prevent, stated as data.

    At least one probe case must have ``solved_serial != guess_serial``,
    otherwise the test suite could not tell the two implementations apart and
    the whole inversion would be untested.
    """
    differing = [
        c for c in CPP["date_helper"] if c["solved_serial"] != c["guess_serial"]
    ]
    assert len(differing) >= 5, (
        "probe must include cases where the NewtonSafe root differs from the "
        f"365.25-per-year guess; found {len(differing)}"
    )
    worst = max(abs(c["solved_serial"] - c["guess_serial"]) for c in differing)
    assert worst >= 5, f"expected a multi-day gap somewhere; worst was {worst}"
