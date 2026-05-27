"""Tests for SwapRateHelper.

L3-C closes the implied_quote carry-over; the deferred-state test was
replaced by a roundtrip test against the cluster_l3c probe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.swap_rate_helper import SwapRateHelper
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = Path(__file__).resolve().parents[4] / "migration-harness/references/cluster/l3c.json"


def test_swap_rate_helper_via_swap_index_inherits_meta() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    swap_idx = EuriborSwapIsdaFixA(Period(5, TimeUnit.Years))
    helper = SwapRateHelper(
        SimpleQuote(0.04),
        swap_index=swap_idx,
        evaluation_date=eval_date,
    )
    assert helper.spread() == 0.0
    assert helper.forward_start() == Period(0, TimeUnit.Days)
    # Dates initialized:
    assert helper.earliest_date().serial > 0
    assert helper.maturity_date().serial > helper.earliest_date().serial


def test_swap_rate_helper_explicit_conventions() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    euribor3m = Euribor.three_months()
    helper = SwapRateHelper(
        SimpleQuote(0.04),
        tenor=Period(5, TimeUnit.Years),
        calendar=TARGET(),
        fixed_frequency=Frequency.Annual,
        fixed_convention=BusinessDayConvention.ModifiedFollowing,
        fixed_day_count=Thirty360(Thirty360Convention.BondBasis),
        ibor_index=euribor3m,
        evaluation_date=eval_date,
    )
    assert helper.earliest_date().serial > 0
    assert helper.maturity_date().serial > helper.earliest_date().serial


def test_swap_rate_helper_implied_quote_roundtrip() -> None:
    """SwapRateHelper.implied_quote — closed L2-C carry-over.

    Build a FlatForward(5%) curve, register a 5y SwapRateHelper at quote 5%,
    assert implied_quote ≈ probe value. LOOSE — the par-swap rate from a
    flat forward 5% (Continuous/Annual) is not exactly 5% (it's ~5.21%).
    """
    refs = json.loads(_REF_PATH.read_text())
    expected_implied = refs["swap_rate_helper"]["implied_quote"]

    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    euribor3m = Euribor.three_months(curve)
    helper = SwapRateHelper(
        SimpleQuote(0.05),
        tenor=Period(5, TimeUnit.Years),
        calendar=TARGET(),
        fixed_frequency=Frequency.Annual,
        fixed_convention=BusinessDayConvention.ModifiedFollowing,
        fixed_day_count=Thirty360(Thirty360Convention.BondBasis),
        ibor_index=euribor3m,
        evaluation_date=eval_date,
    )
    helper.set_term_structure(curve)
    tolerance.loose(helper.implied_quote(), expected_implied)


def test_swap_rate_helper_with_spread() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    swap_idx = EuriborSwapIsdaFixA(Period(5, TimeUnit.Years))
    helper = SwapRateHelper(
        SimpleQuote(0.04),
        swap_index=swap_idx,
        spread=SimpleQuote(0.0010),
        evaluation_date=eval_date,
    )
    assert helper.spread() == 0.0010
