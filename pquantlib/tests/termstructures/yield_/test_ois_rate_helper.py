"""Tests for OISRateHelper.

L3-C closes the implied_quote carry-over; the deferred-state test was
replaced by a roundtrip test against the cluster_l3c probe.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.ois_rate_helper import OISRateHelper
from pquantlib.testing import tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = Path(__file__).resolve().parents[4] / "migration-harness/references/cluster/l3c.json"


_EVAL_DATE = Date.from_ymd(17, Month.January, 2024)


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the global evaluation date to the one the C++ probe used.

    ``OISRateHelper.implied_quote`` builds an ``OvernightIndexedSwap`` and reads
    its ``fair_rate()``; ``Swap.is_expired()`` consults
    ``ObservableSettings().evaluation_date`` through ``CashFlow.has_occurred()``
    (ql/instruments/swap.cpp:68-75). Without this pin the module was silently
    wall-clock dependent: past 2026-01 the 2024-2026 swap reports expired and
    ``fair_rate()`` raises "result not available". The probe
    (``migration-harness/cpp/probes/cluster_l3c/probe.cpp``) builds everything at
    2024-01-17, the same date passed to ``FlatForward.from_rate`` below.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _EVAL_DATE
    try:
        yield
    finally:
        settings.evaluation_date = previous


def test_ois_rate_helper_constructs_with_dates() -> None:
    eval_date = _EVAL_DATE
    helper = OISRateHelper(
        settlement_days=2,
        tenor=Period(5, TimeUnit.Years),
        fixed_rate=SimpleQuote(0.04),
        overnight_index=Sofr(),
        evaluation_date=eval_date,
    )
    assert helper.earliest_date().serial > 0
    assert helper.maturity_date().serial > helper.earliest_date().serial
    assert helper.telescopic_value_dates() is False


def test_ois_rate_helper_implied_quote_roundtrip() -> None:
    """OISRateHelper.implied_quote — closed L2-C carry-over.

    Build a FlatForward(4%) curve, register a 2y OISRateHelper at quote 4%,
    assert implied_quote ≈ probe value. LOOSE tier.
    """
    refs = json.loads(_REF_PATH.read_text())
    expected_implied = refs["ois_rate_helper"]["implied_quote"]

    eval_date = _EVAL_DATE
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.04, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    sofr = Sofr(curve)
    helper = OISRateHelper(
        settlement_days=2,
        tenor=Period(2, TimeUnit.Years),
        fixed_rate=SimpleQuote(0.04),
        overnight_index=sofr,
        evaluation_date=eval_date,
    )
    helper.set_term_structure(curve)
    tolerance.loose(helper.implied_quote(), expected_implied)
