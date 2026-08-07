"""``detail::BootstrapHelperSorter``'s effect, against C++ v1.43.

Reference: ``migration-harness/references/v143/ts/helpersort.json``
(probe ``migration-harness/cpp/probes/v143_ts_helpersort/probe.cpp``).

C++ ``detail::BootstrapHelperSorter`` (bootstraphelper.hpp:248-258) is a
comparator whose entire body is::

    return (h1->pillarDate() < h2->pillarDate());

and whose only use is::

    std::sort(ts_->instruments_.begin(), ts_->instruments_.end(),
              detail::BootstrapHelperSorter());     # iterativebootstrap.hpp:163

PQuantLib expresses the same thing as ``self._instruments.sort(key=lambda h:
h.pillar_date())`` — ``iterative_bootstrap.py:162`` and
``local_bootstrap.py:169``. There is no standalone Python class, so this
module pins the comparator's OBSERVABLE effect instead: the curve built from
helpers supplied in a scrambled order must be identical, node for node, to the
curve C++ builds — and to the one built from the same helpers in pillar order.

That is not a tautology. ``IterativeBootstrap`` solves pillar ``i`` against an
interpolation spanning pillars ``0..i-1``, so an unsorted instrument list
produces a different (and wrong) curve rather than merely a permuted one.

The probe sets an evaluation date (``probe.cpp:174`` — 2024-01-15) because
``DepositRateHelper`` is a relative-date helper; the fixture pins the same date
and restores the previous value.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

CPP: dict[str, Any] = reference_reader.load("v143/ts/helpersort")

# probe.cpp:174 — Date EVAL_DATE(15, January, 2024)
_TODAY = Date.from_ymd(15, Month.January, 2024)


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    prev = s.evaluation_date
    s.evaluation_date = _TODAY  # probe.cpp:174
    yield
    s.evaluation_date = prev


def _helpers(order: str) -> list[DepositRateHelper]:
    months = CPP[f"deposit_tenor_months_{order}"]
    rates = CPP[f"deposit_rates_{order}"]
    return [
        DepositRateHelper(
            SimpleQuote(rate),
            tenor=Period(m, TimeUnit.Months),
            fixing_days=2,
            calendar=TARGET(),
            convention=BusinessDayConvention.ModifiedFollowing,
            end_of_month=True,
            day_counter=Actual360(),
            evaluation_date=_TODAY,
        )
        for m, rate in zip(months, rates, strict=True)
    ]


def _curve(order: str) -> PiecewiseYieldCurve:
    curve = PiecewiseYieldCurve(
        Discount, _TODAY, _helpers(order), Actual365Fixed()
    )
    curve.discount(0.5)  # force the bootstrap
    return curve


def test_reference_is_v143() -> None:
    assert CPP["quantlib_version"] == "1.43"
    assert _TODAY.serial_number() == CPP["eval_date_serial"]


def test_the_scrambled_input_really_is_out_of_order() -> None:
    """Without this the rest of the module would be vacuous.

    Both the probe's helper list and this module's must be genuinely unsorted;
    a fixture that happened to be in pillar order would exercise nothing.
    """
    scrambled = CPP["curves"]["scrambled"]["input_pillar_serials"]
    assert scrambled != sorted(scrambled)
    got = [h.pillar_date().serial_number() for h in _helpers("scrambled")]
    assert got == scrambled


@pytest.mark.parametrize("order", ["ascending", "scrambled"])
def test_pillar_dates_match_cpp(order: str) -> None:
    """Curve nodes come out in pillar order regardless of input order.

    EXACT — dates are integers.
    """
    block = CPP["curves"][order]
    got = [d.serial_number() for d in _curve(order).dates()]
    assert got == block["dates"]
    assert got == sorted(got)


@pytest.mark.parametrize("order", ["ascending", "scrambled"])
def test_times_match_cpp(order: str) -> None:
    block = CPP["curves"][order]
    for got, expected in zip(_curve(order).times(), block["times"], strict=True):
        tolerance.tight(got, expected)


@pytest.mark.parametrize("order", ["ascending", "scrambled"])
def test_bootstrapped_data_matches_cpp(order: str) -> None:
    """The discount factors the bootstrap solved for, per pillar.

    This is the assertion that actually catches a missing sort: solving pillar
    ``i`` before pillar ``i-1`` exists changes these numbers, it does not merely
    reorder them.
    """
    block = CPP["curves"][order]
    for got, expected in zip(_curve(order).data(), block["data"], strict=True):
        tolerance.tight(got, expected)


@pytest.mark.parametrize("order", ["ascending", "scrambled"])
def test_discounts_match_cpp(order: str) -> None:
    block = CPP["curves"][order]
    curve = _curve(order)
    for t, expected in zip(
        block["discount_times"], block["discounts"], strict=True
    ):
        tolerance.tight(curve.discount(t, extrapolate=True), expected)


def test_input_order_does_not_change_the_curve() -> None:
    """The comparator's whole purpose, stated as an invariant.

    C++ emits both curves so the equality is measured against the reference
    rather than asserted between two Python runs.
    """
    asc = CPP["curves"]["ascending"]
    scr = CPP["curves"]["scrambled"]
    assert asc["dates"] == scr["dates"]
    assert asc["data"] == scr["data"]
    assert asc["discounts"] == scr["discounts"]

    py_asc = _curve("ascending")
    py_scr = _curve("scrambled")
    assert [d.serial_number() for d in py_asc.dates()] == [
        d.serial_number() for d in py_scr.dates()
    ]
    for a, b in zip(py_asc.data(), py_scr.data(), strict=True):
        tolerance.exact(a, b)
