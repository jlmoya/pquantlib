"""The bootstrapped curve must reach the furthest date any helper needs.

C++ ``IterativeBootstrap::initialize`` (iterativebootstrap.hpp:179-210)
accumulates ``maxDate = max(pillarDate, latestRelevantDate)`` over the helpers
and writes it onto the curve. This port stopped the curve at its last pillar,
which is only the same answer when every helper pillars at its latest relevant
date — the default, and the reason the gap stayed invisible.

An OIS helper with a payment lag is the case that separates them: the swap
accrues to its maturity date but pays two business days later, so under
``Pillar.MaturityDate`` the node sits at the accrual end while the curve is
still read at the payment date.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.bootstrap_helper import PillarChoice
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
from pquantlib.termstructures.yield_.ois_rate_helper import OISRateHelper
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_EVAL_DATE = Date.from_ymd(15, Month.January, 2024)
_PAYMENT_LAG = 2


@pytest.fixture
def pinned_evaluation_date() -> Iterator[Date]:
    ObservableSettings().evaluation_date = _EVAL_DATE
    yield _EVAL_DATE
    ObservableSettings().evaluation_date = None


def _lagged_ois_helper(years: int, rate: float, pillar: PillarChoice) -> OISRateHelper:
    return OISRateHelper(
        settlement_days=2,
        tenor=Period(years, TimeUnit.Years),
        fixed_rate=SimpleQuote(rate),
        overnight_index=Sofr(),
        payment_lag=_PAYMENT_LAG,
        pillar=pillar,
        evaluation_date=_EVAL_DATE,
    )


def test_lagged_ois_helper_reports_the_payment_date_as_latest_relevant(
    pinned_evaluation_date: Date,
) -> None:
    """C++ parity: oisratehelper.cpp:171-176.

    ``latestRelevantDate`` is the later of the maturity and the last payment
    date. With a two-business-day lag the payment date wins; this port used to
    report the maturity for both, so the helper under-reported how far into
    the curve it actually reads.
    """
    del pinned_evaluation_date
    helper = _lagged_ois_helper(5, 0.04, PillarChoice.MaturityDate)
    assert helper.latest_relevant_date() > helper.maturity_date()
    assert helper.pillar_date() == helper.maturity_date()

    # Under LastRelevantDate the node moves out with it — the two pillar
    # choices were previously indistinguishable.
    on_relevant = _lagged_ois_helper(5, 0.04, PillarChoice.LastRelevantDate)
    assert on_relevant.pillar_date() == on_relevant.latest_relevant_date()
    assert on_relevant.pillar_date() > helper.pillar_date()


def test_curve_extends_past_its_last_pillar(pinned_evaluation_date: Date) -> None:
    """The curve's max date is the accumulated latest relevant date."""
    d = pinned_evaluation_date
    helpers = [
        _lagged_ois_helper(years, rate, PillarChoice.MaturityDate)
        for years, rate in [(1, 0.04), (2, 0.042), (5, 0.045)]
    ]
    curve = PiecewiseYieldCurve(Discount, d, helpers, Actual360())
    _ = curve.discount(0.5)

    last_pillar = curve.dates()[-1]
    assert last_pillar == helpers[-1].pillar_date()
    assert curve.max_date() == helpers[-1].latest_relevant_date()
    assert curve.max_date() > last_pillar

    # And the curve is actually usable out there — before the fix this raised
    # "date is past max curve date" for exactly this shape of helper.
    assert 0.0 < curve.discount(curve.max_date()) < 1.0

    for h, expected in zip(helpers, [0.04, 0.042, 0.045], strict=True):
        loose(h.implied_quote(), expected)


def test_unlagged_helpers_leave_the_max_date_at_the_last_pillar(
    pinned_evaluation_date: Date,
) -> None:
    """The common case is unchanged: pillar == latest relevant == max date."""
    d = pinned_evaluation_date
    helpers = [
        DepositRateHelper(
            SimpleQuote(rate),
            tenor=Period(months, TimeUnit.Months),
            fixing_days=2,
            calendar=TARGET(),
            convention=BusinessDayConvention.ModifiedFollowing,
            end_of_month=True,
            day_counter=Actual360(),
            evaluation_date=d,
        )
        for months, rate in [(1, 0.02), (3, 0.025), (6, 0.03)]
    ]
    curve = PiecewiseYieldCurve(Discount, d, helpers, Actual360())
    _ = curve.discount(0.5)
    assert curve.max_date() == curve.dates()[-1]
