"""SwapRateHelper dates + implied quote, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/swaphelper.json
Probe:     migration-harness/cpp/probes/v143_ts_swaphelper/probe.cpp

The probe pins earliest / maturity / latest-relevant / pillar dates and the
implied quote for 25 configurations. Three of them
(``us_cal_5y_target_holiday_maturity*``) have
``latest_relevant_date == maturity_date + 1``, which is exactly the case the
old ``calendar.advance``-based approximation could not produce.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.bootstrap_helper import PillarChoice
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.swap_rate_helper import SwapRateHelper
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration-harness/references/v143/ts/swaphelper.json"
)

_EVAL = Date.from_ymd(17, Month.January, 2024)
_EVAL_EOM = Date.from_ymd(26, Month.February, 2024)
_EVAL_MAY = Date.from_ymd(26, Month.April, 2023)

_THIRTY360 = Thirty360(Thirty360Convention.BondBasis)
_MF = BusinessDayConvention.ModifiedFollowing
_ZERO = Period(0, TimeUnit.Days)
_Y = TimeUnit.Years
_M = TimeUnit.Months

# Mirrors the probe: the settlementDays == 0 case starts on the evaluation
# date and the fwdStart == -1M case a month before it, so both need their
# past Euribor6M fixing.
_PAST_FIXINGS = {
    Date.from_ymd(15, Month.December, 2023): 0.038,
    Date.from_ymd(15, Month.January, 2024): 0.039,
}


@pytest.fixture(scope="module")
def refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


def _curve(eval_date: Date) -> YieldTermStructureProtocol:
    """The probe's flat 3% continuously-compounded Actual/365F curve."""
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.03, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
        ),
    )


def _euribor(months: int) -> IborIndex:
    return Euribor(Period(months, TimeUnit.Months))


def _us() -> Calendar:
    return UnitedStates(UnitedStates.Market.GovernmentBond)


def _helper(
    eval_date: Date,
    tenor: Period,
    calendar: Calendar,
    fixed_frequency: Frequency,
    fixed_convention: BusinessDayConvention,
    fixed_day_count: DayCounter,
    index: IborIndex,
    *,
    fwd_start: Period = _ZERO,
    settlement_days: int | None = None,
    pillar: PillarChoice = PillarChoice.LastRelevantDate,
    custom_pillar_date: Date | None = None,
    end_of_month: bool = False,
    use_indexed_coupons: bool | None = None,
    spread: float | None = None,
) -> SwapRateHelper:
    helper = SwapRateHelper(
        SimpleQuote(0.04),
        tenor=tenor,
        calendar=calendar,
        fixed_frequency=fixed_frequency,
        fixed_convention=fixed_convention,
        fixed_day_count=fixed_day_count,
        ibor_index=index,
        spread=None if spread is None else SimpleQuote(spread),
        fwd_start=fwd_start,
        settlement_days=settlement_days,
        pillar=pillar,
        custom_pillar_date=custom_pillar_date,
        end_of_month=end_of_month,
        use_indexed_coupons=use_indexed_coupons,
        evaluation_date=eval_date,
    )
    helper.set_term_structure(_curve(eval_date))
    return helper


def _build_cases() -> dict[str, SwapRateHelper]:
    e6m, e3m, e12m = _euribor(6), _euribor(3), _euribor(12)
    target, us = TARGET(), _us()
    cases = {
        "eur_5y_euribor6m": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m
        ),
        "eur_5y_euribor3m": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e3m
        ),
        "eur_18m_euribor6m": _helper(
            _EVAL, Period(18, _M), target, Frequency.Annual, _MF, _THIRTY360, e6m
        ),
        "eur_4m_euribor3m": _helper(
            _EVAL, Period(4, _M), target, Frequency.Quarterly, _MF, _THIRTY360, e3m
        ),
        "eur_1y_euribor12m_once": _helper(
            _EVAL, Period(1, _Y), target, Frequency.Once, _MF, _THIRTY360, e12m
        ),
        "eur_5y_fwd1y": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            fwd_start=Period(1, _Y),
        ),
        "eur_5y_fwd_minus1m": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            fwd_start=Period(-1, _M),
        ),
        "eur_5y_settlement0": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            settlement_days=0,
        ),
        "eur_5y_settlement5": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            settlement_days=5,
        ),
        "us_cal_5y_euribor6m": _helper(
            _EVAL, Period(5, _Y), us, Frequency.Annual, _MF, _THIRTY360, e6m
        ),
        "us_cal_7y_euribor6m": _helper(
            _EVAL, Period(7, _Y), us, Frequency.Annual, _MF, _THIRTY360, e6m
        ),
        "us_cal_10y_euribor3m": _helper(
            _EVAL, Period(10, _Y), us, Frequency.Annual, _MF, _THIRTY360, e3m
        ),
        "eur_18m_indexed": _helper(
            _EVAL, Period(18, _M), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            use_indexed_coupons=True,
        ),
        "us_cal_5y_indexed": _helper(
            _EVAL, Period(5, _Y), us, Frequency.Annual, _MF, _THIRTY360, e6m,
            use_indexed_coupons=True,
        ),
        "eur_5y_at_par_explicit": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            use_indexed_coupons=False,
        ),
        "eur_5y_pillar_maturity": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            pillar=PillarChoice.MaturityDate,
        ),
        "eur_5y_pillar_custom": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            pillar=PillarChoice.CustomDate,
            custom_pillar_date=Date.from_ymd(19, Month.January, 2027),
        ),
        "eur_5y_spread_25bp": _helper(
            _EVAL, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            spread=0.0025,
        ),
        "eur_7y_semi_act360": _helper(
            _EVAL, Period(7, _Y), target, Frequency.Semiannual,
            BusinessDayConvention.Following, Actual360(), e6m,
        ),
        "eom_5y_euribor6m": _helper(
            _EVAL_EOM, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            settlement_days=3, end_of_month=True,
        ),
        "eom_off_5y_euribor6m": _helper(
            _EVAL_EOM, Period(5, _Y), target, Frequency.Annual, _MF, _THIRTY360, e6m,
            settlement_days=3, end_of_month=False,
        ),
        "eom_us_cal_5y_euribor6m": _helper(
            _EVAL_EOM, Period(5, _Y), us, Frequency.Annual, _MF, _THIRTY360, e6m,
            settlement_days=3, end_of_month=True,
        ),
        "us_cal_5y_target_holiday_maturity": _helper(
            _EVAL_MAY, Period(5, _Y), us, Frequency.Annual, _MF, _THIRTY360, e6m,
            settlement_days=3,
        ),
        "us_cal_5y_target_holiday_maturity_indexed": _helper(
            _EVAL_MAY, Period(5, _Y), us, Frequency.Annual, _MF, _THIRTY360, e6m,
            settlement_days=3, use_indexed_coupons=True,
        ),
        "us_cal_5y_target_holiday_maturity_pillar_maturity": _helper(
            _EVAL_MAY, Period(5, _Y), us, Frequency.Annual, _MF, _THIRTY360, e6m,
            settlement_days=3, pillar=PillarChoice.MaturityDate,
        ),
    }
    return cases


@pytest.fixture(scope="module")
def cases() -> Iterator[dict[str, SwapRateHelper]]:
    index_name = _euribor(6).name()
    for d, f in _PAST_FIXINGS.items():
        IndexManager().add_fixing(index_name, d, f, True)
    try:
        yield _build_cases()
    finally:
        IndexManager().clear_history(index_name)


def test_every_probe_case_is_covered(
    refs: dict[str, dict[str, float]], cases: dict[str, SwapRateHelper]
) -> None:
    assert set(cases) == set(refs)


@pytest.mark.parametrize(
    "field",
    [
        "earliest_date",
        "maturity_date",
        "latest_relevant_date",
        "pillar_date",
        "latest_date",
    ],
)
def test_dates_match_cpp(
    refs: dict[str, dict[str, float]], cases: dict[str, SwapRateHelper], field: str
) -> None:
    getter = {
        "earliest_date": SwapRateHelper.earliest_date,
        "maturity_date": SwapRateHelper.maturity_date,
        "latest_relevant_date": SwapRateHelper.latest_relevant_date,
        "pillar_date": SwapRateHelper.pillar_date,
        "latest_date": SwapRateHelper.latest_date,
    }[field]
    for key, helper in cases.items():
        assert getter(helper).serial == refs[key][field], f"{key}.{field}"


def test_latest_relevant_date_exceeds_maturity_where_cpp_says_so(
    refs: dict[str, dict[str, float]], cases: dict[str, SwapRateHelper]
) -> None:
    """The discriminating cases: a schedule calendar the index doesn't share.

    Guards against a regression to the old approximation, which could only
    ever return ``latest_relevant_date == maturity_date``.
    """
    strictly_later = {
        k for k, v in refs.items() if v["latest_relevant_date"] > v["maturity_date"]
    }
    assert strictly_later == {
        "us_cal_5y_target_holiday_maturity",
        "us_cal_5y_target_holiday_maturity_indexed",
        "us_cal_5y_target_holiday_maturity_pillar_maturity",
    }
    for key in strictly_later:
        helper = cases[key]
        assert helper.latest_relevant_date() > helper.maturity_date()


def test_implied_quote_matches_cpp(
    refs: dict[str, dict[str, float]], cases: dict[str, SwapRateHelper]
) -> None:
    for key, helper in cases.items():
        tolerance.tight(helper.implied_quote(), refs[key]["implied_quote"], reason=key)
