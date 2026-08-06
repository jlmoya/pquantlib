"""OvernightIndexFuture + its rate helpers, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/onfuture.json
Probe:     migration-harness/cpp/probes/v143_ts_onfuture/probe.cpp

Both averaging branches over the same period, at three valuation regimes
(before the reference period, inside it with today's fixing published, inside
it without), plus a weekend maturity where the daily roll overshoots — the
averaged branch caps it at maturity and the compounded branch does not.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.instruments.overnight_index_future import OvernightIndexFuture
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.bootstrap_helper import PillarChoice
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.overnight_index_future_rate_helper import (
    OvernightIndexFutureRateHelper,
    SofrFutureRateHelper,
)
from pquantlib.testing import tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

_REF_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration-harness/references/v143/ts/onfuture.json"
)

_CURVE_REF = Date.from_ymd(2, Month.January, 2024)
_Q1_START = Date.from_ymd(20, Month.March, 2024)
_Q1_END = Date.from_ymd(19, Month.June, 2024)


def _curve() -> YieldTermStructureProtocol:
    curve = FlatForward.from_rate(
        _CURVE_REF, 0.0525, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    curve.enable_extrapolation()
    return cast("YieldTermStructureProtocol", curve)


def _seed_fixings() -> None:
    """Mirror the probe's fixing history: business days 20 Mar - 5 Apr 2024."""
    index = Sofr()
    cal = index.fixing_calendar()
    d = Date.from_ymd(20, Month.March, 2024)
    r = 0.0530
    last = Date.from_ymd(5, Month.April, 2024)
    while d <= last:
        if cal.is_business_day(d):
            index.add_fixing(d, r, True)
        r += 0.00001
        d = d + 1


@pytest.fixture(autouse=True)
def _settings() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    previous = settings.evaluation_date
    IndexManager().clear_history(Sofr().name())
    try:
        yield
    finally:
        settings.evaluation_date = previous
        IndexManager().clear_history(Sofr().name())


@pytest.fixture(scope="module")
def refs() -> dict[str, dict[str, float]]:
    raw = cast("dict[str, object]", json.loads(_REF_PATH.read_text()))
    return {k: cast("dict[str, float]", v) for k, v in raw.items()}


def _future(
    value_date: Date,
    maturity_date: Date,
    averaging: RateAveraging,
    evaluation_date: Date,
    convexity: float | None = None,
) -> OvernightIndexFuture:
    ObservableSettings().evaluation_date = evaluation_date
    return OvernightIndexFuture(
        Sofr(_curve()),
        value_date,
        maturity_date,
        None if convexity is None else SimpleQuote(convexity),
        averaging,
    )


_BEFORE = Date.from_ymd(2, Month.January, 2024)
_APR_1 = Date.from_ymd(1, Month.April, 2024)
_MAY_1 = Date.from_ymd(1, Month.May, 2024)
_MAY_4 = Date.from_ymd(4, Month.May, 2024)
_JUL_1 = Date.from_ymd(1, Month.July, 2024)

_FUTURE_CASES: dict[str, tuple[Date, Date, RateAveraging, Date, float | None]] = {
    "compounded_before_start": (_Q1_START, _Q1_END, RateAveraging.Compound, _BEFORE, None),
    "averaged_before_start": (_Q1_START, _Q1_END, RateAveraging.Simple, _BEFORE, None),
    "compounded_with_convexity": (
        _Q1_START, _Q1_END, RateAveraging.Compound, _BEFORE, 0.0012,
    ),
    "monthly_averaged_before_start": (
        _APR_1, _MAY_1, RateAveraging.Simple, _BEFORE, None,
    ),
    "monthly_compounded_before_start": (
        _APR_1, _MAY_1, RateAveraging.Compound, _BEFORE, None,
    ),
    "weekend_maturity_averaged": (_APR_1, _MAY_4, RateAveraging.Simple, _BEFORE, None),
    "weekend_maturity_compounded": (
        _APR_1, _MAY_4, RateAveraging.Compound, _BEFORE, None,
    ),
}

_INSIDE_CASES: dict[str, tuple[Date, Date, RateAveraging, Date, float | None]] = {
    "compounded_inside_fixing_published": (
        _Q1_START, _Q1_END, RateAveraging.Compound,
        Date.from_ymd(5, Month.April, 2024), None,
    ),
    "averaged_inside_fixing_published": (
        _Q1_START, _Q1_END, RateAveraging.Simple,
        Date.from_ymd(5, Month.April, 2024), None,
    ),
    "compounded_inside_fixing_missing": (
        _Q1_START, _Q1_END, RateAveraging.Compound,
        Date.from_ymd(8, Month.April, 2024), None,
    ),
}


@pytest.mark.parametrize("key", sorted(_FUTURE_CASES))
def test_future_before_start_matches_cpp(
    refs: dict[str, dict[str, float]], key: str
) -> None:
    f = _future(*_FUTURE_CASES[key])
    block = refs[key]
    assert f.value_date().serial == block["value_date"], key
    assert f.maturity_date().serial == block["maturity_date"], key
    assert f.is_expired() == bool(block["is_expired"]), key
    tolerance.tight(f.convexity_adjustment(), block["convexity_adjustment"], reason=key)
    tolerance.tight(f.npv(), block["npv"], reason=key)


@pytest.mark.parametrize("key", sorted(_INSIDE_CASES))
def test_future_inside_period_matches_cpp(
    refs: dict[str, dict[str, float]], key: str
) -> None:
    _seed_fixings()
    f = _future(*_INSIDE_CASES[key])
    tolerance.tight(f.npv(), refs[key]["npv"], reason=key)


def test_averaging_branches_disagree(refs: dict[str, dict[str, float]]) -> None:
    """Guard: simple and compounded must not have collapsed into each other.

    They differ only by the cross terms, so a port that wired both branches to
    the same code would still price plausibly.
    """
    assert (
        refs["compounded_before_start"]["npv"] != refs["averaged_before_start"]["npv"]
    )
    assert (
        refs["weekend_maturity_averaged"]["npv"]
        != refs["weekend_maturity_compounded"]["npv"]
    )


def _helpers() -> dict[str, OvernightIndexFutureRateHelper]:
    ObservableSettings().evaluation_date = _BEFORE
    return {
        "helper_quarterly": SofrFutureRateHelper(
            94.75, Month.March, 2024, Frequency.Quarterly
        ),
        "helper_quarterly_convexity": SofrFutureRateHelper(
            94.75, Month.March, 2024, Frequency.Quarterly, 0.0012
        ),
        "helper_monthly": SofrFutureRateHelper(
            94.70, Month.April, 2024, Frequency.Monthly
        ),
        "helper_september_quarterly": SofrFutureRateHelper(
            94.60, Month.September, 2024, Frequency.Quarterly
        ),
        "helper_december_monthly": SofrFutureRateHelper(
            94.55, Month.December, 2024, Frequency.Monthly
        ),
        "helper_generic_compound": OvernightIndexFutureRateHelper(
            SimpleQuote(94.80), _APR_1, _JUL_1, Sofr(), None,
            RateAveraging.Compound, PillarChoice.MaturityDate,
        ),
        "helper_generic_custom_pillar": OvernightIndexFutureRateHelper(
            SimpleQuote(94.80), _APR_1, _JUL_1, Sofr(), None,
            RateAveraging.Simple, PillarChoice.CustomDate,
            Date.from_ymd(3, Month.June, 2024),
        ),
    }


@pytest.mark.parametrize(
    "key",
    [
        "helper_quarterly",
        "helper_quarterly_convexity",
        "helper_monthly",
        "helper_september_quarterly",
        "helper_december_monthly",
        "helper_generic_compound",
        "helper_generic_custom_pillar",
    ],
)
def test_helper_matches_cpp(refs: dict[str, dict[str, float]], key: str) -> None:
    helper = _helpers()[key]
    helper.set_term_structure(_curve())
    block = refs[key]
    assert helper.earliest_date().serial == block["earliest_date"], key
    assert helper.latest_date().serial == block["latest_date"], key
    assert helper.pillar_date().serial == block["pillar_date"], key
    tolerance.tight(
        helper.convexity_adjustment(), block["convexity_adjustment"], reason=key
    )
    tolerance.tight(helper.implied_quote(), block["implied_quote"], reason=key)


def test_sofr_reference_period_conventions(refs: dict[str, dict[str, float]]) -> None:
    """Quarterly runs 3rd Wednesday to 3rd Wednesday; monthly 1st to 1st."""
    q = refs["helper_quarterly"]
    assert Date(int(q["earliest_date"])) == Date.from_ymd(20, Month.March, 2024)
    assert Date(int(q["latest_date"])) == Date.from_ymd(19, Month.June, 2024)
    m = refs["helper_monthly"]
    assert Date(int(m["earliest_date"])) == Date.from_ymd(1, Month.April, 2024)
    assert Date(int(m["latest_date"])) == Date.from_ymd(1, Month.May, 2024)


def test_sofr_rejects_unsupported_frequency() -> None:
    with pytest.raises(Exception, match="only monthly and quarterly"):
        SofrFutureRateHelper(94.75, Month.March, 2024, Frequency.Annual)
