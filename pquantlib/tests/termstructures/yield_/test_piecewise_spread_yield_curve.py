"""Cross-validation of ``PiecewiseSpreadYieldCurve`` against C++ v1.43.

# C++ parity: ql/termstructures/yield/piecewisespreadyieldcurve.hpp,
#             ql/termstructures/yield/spreadbootstraptraits.hpp.

Reference: ``migration-harness/references/v143/ts/traits.json``, produced by
``migration-harness/cpp/probes/v143_ts_traits/probe.cpp``.

The bootstrap solves the multiplicative *spread* discount factors, not the
outright ones: the curve's discount factor is
``base.discount(t) * spread(t)`` and the helpers are repriced through that
product. ``detail::SpreadTraits<Discount>`` contributes nothing of its own
— it inherits every member from ``Discount`` and only swaps the underlying
curve type — which is exactly what the first test asserts.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.interpolations.log_linear import LogLinearInterpolation
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.interpolated_discount_curve import (
    InterpolatedDiscountCurve,
)
from pquantlib.termstructures.yield_.interpolated_spread_discount_curve import (
    InterpolatedSpreadDiscountCurve,
)
from pquantlib.termstructures.yield_.piecewise_spread_yield_curve import (
    PiecewiseSpreadYieldCurve,
    SpreadTraits,
)
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

CPP = reference_reader.load("v143/ts/traits")

# probe.cpp:57 / 196.
REF_DATE = Date.from_ymd(15, Month.January, 2024)
# probe.cpp:59-73 — the curve the SpreadTraits members are probed against.
NODE_DATES = [
    REF_DATE,
    REF_DATE + Period(6, TimeUnit.Months),
    REF_DATE + Period(1, TimeUnit.Years),
    REF_DATE + Period(2, TimeUnit.Years),
    REF_DATE + Period(5, TimeUnit.Years),
    REF_DATE + Period(10, TimeUnit.Years),
]
NODE_DISCOUNTS = [1.0, 0.985, 0.968, 0.930, 0.815, 0.640]
PILLARS = [1, 2, 3, 5]
TRANSFORM_DIRECT_X = -0.35
TRANSFORM_INVERSE_X = 0.87

# probe.cpp:172-174.
DEPOSIT_QUOTES = [(3, 0.0325), (6, 0.0340), (12, 0.0355), (24, 0.0370), (60, 0.0390)]
# probe.cpp:232-234 — the flat base curve.
BASE_RATE = 0.030
# probe.cpp:243.
DISCOUNT_TIMES = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0]


@pytest.fixture
def evaluation_date() -> Iterator[Date]:
    """# probe.cpp:196 — ``Settings::instance().evaluationDate() = kRef``."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = REF_DATE
    yield REF_DATE
    settings.evaluation_date = previous


def _base_curve() -> FlatForward:
    return FlatForward(
        reference_date=REF_DATE,
        forward=SimpleQuote(BASE_RATE),
        day_counter=Actual365Fixed(),
        compounding=Compounding.Continuous,
        frequency=Frequency.Annual,
    )


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


def _spread_curve(base: YieldTermStructure) -> PiecewiseSpreadYieldCurve:
    curve = PiecewiseSpreadYieldCurve(base, _deposit_helpers())
    curve.enable_extrapolation()
    return curve


# --- SpreadTraits<Discount> -------------------------------------------------


def test_spread_traits_is_a_discount_trait() -> None:
    """# C++ parity: ``struct SpreadTraits<Discount> : Discount``.

    (spreadbootstraptraits.hpp:16-22 — the whole body is the curve typedef.)
    """
    assert issubclass(SpreadTraits, Discount)


@pytest.mark.parametrize("i", PILLARS)
def test_spread_traits_arithmetic_equals_discount(i: int) -> None:
    """Every SpreadTraits member must reproduce the plain Discount reference.

    The probe emits both key families against the same curve; if they ever
    diverged, ``SpreadTraits`` would have grown a body it does not have.
    """
    curve = InterpolatedDiscountCurve(
        dates=NODE_DATES,
        dfs=NODE_DISCOUNTS,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        interpolator=LogLinearInterpolation,
    )
    curve.enable_extrapolation()
    t = SpreadTraits()
    for suffix, got in (
        ("guess_valid", t.guess(i, curve, True, 0)),
        ("guess_invalid", t.guess(i, curve, False, 0)),
        ("min_valid", t.min_value_after(i, curve, True, 0)),
        ("min_invalid", t.min_value_after(i, curve, False, 0)),
        ("max_valid", t.max_value_after(i, curve, True, 0)),
        ("max_invalid", t.max_value_after(i, curve, False, 0)),
        ("transform_direct", t.transform_direct(TRANSFORM_DIRECT_X, i, curve)),
        ("transform_inverse", t.transform_inverse(TRANSFORM_INVERSE_X, i, curve)),
    ):
        want = CPP[f"spreadtraits_i{i}_{suffix}"]
        tight(got, want)
        # ... and the C++ SpreadTraits value equals the C++ Discount value.
        exact(want, CPP[f"discount_i{i}_{suffix}"])


def test_spread_traits_scalars() -> None:
    curve = InterpolatedDiscountCurve(
        dates=NODE_DATES,
        dfs=NODE_DISCOUNTS,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        interpolator=LogLinearInterpolation,
    )
    t = SpreadTraits()
    exact(t.initial_value(curve), CPP["spreadtraits_initial_value"])
    assert t.initial_date(curve).serial_number() == CPP["spreadtraits_initial_date"]
    assert t.max_iterations() == int(CPP["spreadtraits_max_iterations"])


# --- the bootstrapped curve -------------------------------------------------


def test_spread_curve_underlying_is_the_spread_discount_curve(
    evaluation_date: Date,
) -> None:
    """# C++ parity: ``SpreadTraits<Discount>::curve<I>::type``."""
    del evaluation_date
    base = _base_curve()
    curve = _spread_curve(base)
    curve.data()  # force the bootstrap
    underlying = curve._underlying  # pyright: ignore[reportPrivateUsage]
    assert isinstance(underlying, InterpolatedSpreadDiscountCurve)
    assert curve.base_curve() is base
    assert curve.traits() is SpreadTraits


def test_spread_curve_forwards_conventions_to_the_base(
    evaluation_date: Date,
) -> None:
    """# C++ parity: spreaddiscountcurve.hpp:117-137."""
    del evaluation_date
    base = _base_curve()
    curve = _spread_curve(base)
    assert curve.reference_date() == base.reference_date()
    assert curve.day_counter() == base.day_counter()


def test_spread_curve_grid(evaluation_date: Date) -> None:
    """Grid dates/times use the BASE curve's day counter. # probe.cpp:236-241."""
    del evaluation_date
    curve = _spread_curve(_base_curve())
    assert [d.serial_number() for d in curve.dates()] == CPP["pw_spread_dates"]
    assert curve.max_date().serial_number() == CPP["pw_spread_max_date"]
    for got, want in zip(curve.times(), CPP["pw_spread_times"], strict=True):
        tight(got, want)


def test_spread_curve_bootstrapped_spreads(evaluation_date: Date) -> None:
    """The bootstrap state is the SPREAD discount factor, anchored at 1.0.

    TIGHT tier: with the guess, the bracket and the interpolation extent
    aligned to C++, the two Brent runs land on the same doubles.
    """
    del evaluation_date
    curve = _spread_curve(_base_curve())
    data = curve.data()
    exact(data[0], 1.0)
    for got, want in zip(data, CPP["pw_spread_data"], strict=True):
        tight(got, want)
    # A spread, not an outright discount factor: at every pillar the stored
    # value is strictly above the curve's own discount factor there, because
    # the outright is that value times the (sub-unit) base discount.
    times = curve.times()
    for i in range(1, len(data)):
        assert data[i] > curve.discount(times[i], True)


def test_spread_curve_discounts_are_base_times_spread(
    evaluation_date: Date,
) -> None:
    """# C++ parity: ``discountImpl = baseCurve->discount(t) * calcSpread(t)``."""
    del evaluation_date
    base = _base_curve()
    curve = _spread_curve(base)
    for t, want, want_base in zip(
        DISCOUNT_TIMES,
        CPP["pw_spread_discounts"],
        CPP["pw_spread_base_discounts"],
        strict=True,
    ):
        tight(base.discount(t, True), want_base)
        tight(curve.discount(t, True), want)


def test_spread_curve_reprices_its_helpers(evaluation_date: Date) -> None:
    """Sanity: a correctly bootstrapped spread returns each deposit quote."""
    del evaluation_date
    helpers = _deposit_helpers()
    curve = PiecewiseSpreadYieldCurve(_base_curve(), helpers)
    curve.enable_extrapolation()
    curve.data()  # force the bootstrap
    for h in helpers:
        assert abs(h.quote_error()) < 1e-10
