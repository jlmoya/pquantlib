"""Cross-validation of ``SimpleZeroYield`` against C++ v1.43.

# C++ parity: ql/termstructures/yield/bootstraptraits.hpp:312-404.

Reference: ``migration-harness/references/v143/ts/traits.json``, produced by
``migration-harness/cpp/probes/v143_ts_traits/probe.cpp``.

The trait that a port most easily gets wrong: ``SimpleZeroYield`` is the
only yield trait whose transforms are NOT the identity and whose
``minValueAfter`` carries a barrier. Both come from the same constraint —
the discount factor is ``1 / (1 + R t)``, which blows through zero at
``R = -1/t``, so every bound is clamped to ``-1/t + 1e-8``. Whether the
barrier binds depends on the pillar:

- pillar 1 sits at ``t ~ 0.4986`` where ``-1/t + 1e-8 ~ -2.0055`` is BELOW
  ``-maxRate = -1``, so the clamp is inert and the bound stays ``-1``;
- pillars 2, 3, 5 sit at ``t > 1`` where the barrier is above ``-1`` and
  therefore BINDS.

Both regimes are pinned below, so dropping the clamp cannot pass.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
from pquantlib.termstructures.yield_.interpolated_simple_zero_curve import (
    InterpolatedSimpleZeroCurve,
)
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.yield_traits import SimpleZeroYield
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

CPP = reference_reader.load("v143/ts/traits")

# probe.cpp:57 — ``const Date kRef(15, January, 2024)``, also the probe's
# evaluation date (probe.cpp:196).
REF_DATE = Date.from_ymd(15, Month.January, 2024)

# probe.cpp:59-67 / 71-79.
NODE_DATES = [
    REF_DATE,
    REF_DATE + Period(6, TimeUnit.Months),
    REF_DATE + Period(1, TimeUnit.Years),
    REF_DATE + Period(2, TimeUnit.Years),
    REF_DATE + Period(5, TimeUnit.Years),
    REF_DATE + Period(10, TimeUnit.Years),
]
NODE_RATES = [0.030, 0.031, -0.005, 0.035, 0.0405, 0.0446]
PILLARS = [1, 2, 3, 5]
TRANSFORM_DIRECT_X = -0.35
# probe.cpp:90 — kSimpleTransformInverseX.
TRANSFORM_INVERSE_X = 0.05

# probe.cpp:172-174 — depositHelpers().
DEPOSIT_QUOTES = [(3, 0.0325), (6, 0.0340), (12, 0.0355), (24, 0.0370), (60, 0.0390)]
# probe.cpp:226 — the times at which the bootstrapped curve is read.
DISCOUNT_TIMES = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0]


@pytest.fixture
def evaluation_date() -> Iterator[Date]:
    """Pin the evaluation date to the probe's, restore the previous value.

    # probe.cpp:196 — ``Settings::instance().evaluationDate() = kRef``.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = REF_DATE
    yield REF_DATE
    settings.evaluation_date = previous


def _simple_zero_curve() -> InterpolatedSimpleZeroCurve:
    c = InterpolatedSimpleZeroCurve(
        dates=NODE_DATES,
        yields=NODE_RATES,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        interpolator=LinearInterpolation,
    )
    c.enable_extrapolation()
    return c


def _deposit_helpers() -> list[DepositRateHelper]:
    return [
        DepositRateHelper(
            SimpleQuote(rate),
            tenor=Period(months, TimeUnit.Months),
            fixing_days=2,
            calendar=TARGET(),
            convention=BusinessDayConvention.ModifiedFollowing,
            end_of_month=True,
            day_counter=Actual360(),
            evaluation_date=REF_DATE,
        )
        for months, rate in DEPOSIT_QUOTES
    ]


# --- static trait members ---------------------------------------------------


def test_simple_zero_scalars() -> None:
    c = _simple_zero_curve()
    exact(SimpleZeroYield().initial_value(c), CPP["simplezero_initial_value"])
    assert (
        SimpleZeroYield().initial_date(c).serial_number()
        == CPP["simplezero_initial_date"]
    )
    assert SimpleZeroYield().max_iterations() == int(CPP["simplezero_max_iterations"])
    for got, want in zip(c.times(), CPP["simplezero_times"], strict=True):
        tight(got, want)


@pytest.mark.parametrize("i", PILLARS)
def test_simple_zero_guess(i: int) -> None:
    """``i > 1`` extrapolates via ``zeroRate(d, dc, Simple, Annual, true)``.

    # C++ parity: bootstraptraits.hpp:344-346 — ``Simple``, not
    # ``Continuous`` as :class:`ZeroYield` uses.
    """
    c = _simple_zero_curve()
    tight(SimpleZeroYield().guess(i, c, True, 0), CPP[f"simplezero_i{i}_guess_valid"])
    tight(
        SimpleZeroYield().guess(i, c, False, 0), CPP[f"simplezero_i{i}_guess_invalid"]
    )


@pytest.mark.parametrize("i", PILLARS)
def test_simple_zero_bracket(i: int) -> None:
    """``min_value_after`` clamps at ``-1/times[i] + 1e-8``; ``max`` does not.

    # C++ parity: bootstraptraits.hpp:350-380.
    """
    c = _simple_zero_curve()
    tight(
        SimpleZeroYield().min_value_after(i, c, True, 0),
        CPP[f"simplezero_i{i}_min_valid"],
    )
    tight(
        SimpleZeroYield().min_value_after(i, c, False, 0),
        CPP[f"simplezero_i{i}_min_invalid"],
    )
    tight(
        SimpleZeroYield().max_value_after(i, c, True, 0),
        CPP[f"simplezero_i{i}_max_valid"],
    )
    tight(
        SimpleZeroYield().max_value_after(i, c, False, 0),
        CPP[f"simplezero_i{i}_max_invalid"],
    )


def test_simple_zero_barrier_binds_only_past_one_year() -> None:
    """The regime split the clamp exists for — guards the two branches above.

    Without the ``max(result, -1/t + 1e-8)`` line, pillars 2/3/5 would all
    report ``-maxRate``; without ``-maxRate``, pillar 1 would report the
    barrier. Both directions are wrong, and this asserts which is which.
    """
    c = _simple_zero_curve()
    # Pillar 1: t < 1 ⇒ barrier below -1 ⇒ inert.
    exact(SimpleZeroYield().min_value_after(1, c, False, 0), -1.0)
    # Pillars past t = 1: barrier binds and is strictly greater than -1.
    for i in (2, 3, 5):
        got = SimpleZeroYield().min_value_after(i, c, False, 0)
        assert got > -1.0
        tight(got, -1.0 / c.times()[i] + 1e-8)


@pytest.mark.parametrize("i", PILLARS)
def test_simple_zero_transforms_are_shifted_not_identity(i: int) -> None:
    """``exp(x) + (-1/t + 1e-8)`` / ``log(x - (-1/t + 1e-8))``.

    # C++ parity: bootstraptraits.hpp:383-392.
    """
    c = _simple_zero_curve()
    tight(
        SimpleZeroYield().transform_direct(TRANSFORM_DIRECT_X, i, c),
        CPP[f"simplezero_i{i}_transform_direct"],
    )
    tight(
        SimpleZeroYield().transform_inverse(TRANSFORM_INVERSE_X, i, c),
        CPP[f"simplezero_i{i}_transform_inverse"],
    )
    # The shift is real: neither transform is the identity anywhere here.
    assert CPP[f"simplezero_i{i}_transform_direct"] != TRANSFORM_DIRECT_X
    assert CPP[f"simplezero_i{i}_transform_inverse"] != TRANSFORM_INVERSE_X


@pytest.mark.parametrize("i", PILLARS)
def test_simple_zero_transform_roundtrip(i: int) -> None:
    """``transform_direct(transform_inverse(x)) == x`` on the admissible side."""
    c = _simple_zero_curve()
    t = SimpleZeroYield()
    x = 0.037
    tight(t.transform_direct(t.transform_inverse(x, i, c), i, c), x)


@pytest.mark.parametrize("i", [1, 2])
def test_simple_zero_update_guess(i: int) -> None:
    """Pillar 1 is mirrored into pillar 0. # C++ parity: bootstraptraits.hpp:395-401."""
    data = list(NODE_RATES)
    SimpleZeroYield().update_guess(data, 0.123456789, i)
    for got, want in zip(data, CPP[f"simplezero_i{i}_update_guess"], strict=True):
        exact(got, want)


# --- bootstrapped PiecewiseYieldCurve<SimpleZeroYield, Linear> --------------


def test_piecewise_simple_zero_grid(evaluation_date: Date) -> None:
    """The bootstrap grid (dates / times) must match C++ before the data can.

    # probe.cpp:216-224.
    """
    curve = PiecewiseYieldCurve(
        SimpleZeroYield, evaluation_date, _deposit_helpers(), Actual360()
    )
    curve.enable_extrapolation()
    assert [d.serial_number() for d in curve.dates()] == CPP["pw_simplezero_dates"]
    assert curve.max_date().serial_number() == CPP["pw_simplezero_max_date"]
    for got, want in zip(curve.times(), CPP["pw_simplezero_times"], strict=True):
        tight(got, want)


def test_piecewise_simple_zero_bootstrapped_data(evaluation_date: Date) -> None:
    """Bootstrapped simple zero rates at each pillar.

    TIGHT tier: these are Brent roots rather than closed form, but with the
    guess, the bracket and the interpolation extent all aligned to C++ the
    two solvers land on the same doubles (worst observed relative gap over
    the six values: 2e-16, i.e. one ulp).
    """
    curve = PiecewiseYieldCurve(
        SimpleZeroYield, evaluation_date, _deposit_helpers(), Actual360()
    )
    curve.enable_extrapolation()
    for got, want in zip(curve.data(), CPP["pw_simplezero_data"], strict=True):
        tight(got, want)


def test_piecewise_simple_zero_discounts(evaluation_date: Date) -> None:
    """Discount factors on, between and past the pillars. # probe.cpp:225-228."""
    curve = PiecewiseYieldCurve(
        SimpleZeroYield, evaluation_date, _deposit_helpers(), Actual360()
    )
    curve.enable_extrapolation()
    got = [curve.discount(t, True) for t in DISCOUNT_TIMES]
    for g, w in zip(got, CPP["pw_simplezero_discounts"], strict=True):
        tight(g, w)


def test_piecewise_simple_zero_reprices_its_helpers(evaluation_date: Date) -> None:
    """Sanity: a correctly bootstrapped curve returns each deposit quote."""
    helpers = _deposit_helpers()
    curve = PiecewiseYieldCurve(
        SimpleZeroYield, evaluation_date, helpers, Actual360()
    )
    curve.enable_extrapolation()
    curve.data()  # force the bootstrap
    for h in helpers:
        assert abs(h.quote_error()) < 1e-10
