"""Shared fixtures for Phase 11 W4-A multi-asset exotic option tests.

Builds a deterministic 3-asset basket setup matching the C++ probe
(cluster_w4a/probe.cpp):

* today = 15-Jan-2024 (TARGET calendar, Actual/360 day counter).
* assets: spots 100/95/105, vols 0.20/0.25/0.30, r=0.05, q=0.02.
* correlation: identity + 0.3 off-diagonal (3x3).

The 2-asset analytic-engine tests build their own per-asset setup.
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def today() -> Date:
    return Date.from_ymd(15, Month.January, 2024)


@pytest.fixture(scope="module")
def day_counter() -> Actual360:
    return Actual360()


@pytest.fixture(scope="module")
def calendar() -> TARGET:
    return TARGET()


def _make_bsm(
    today: Date, spot: float, rate: float, div: float, vol: float
) -> GeneralizedBlackScholesProcess:
    dc = Actual360()
    s = SimpleQuote(spot)
    r = FlatForward.from_rate(today, rate, dc)
    q = FlatForward.from_rate(today, div, dc)
    v = BlackConstantVol(
        reference_date=today,
        calendar=NullCalendar(),
        day_counter=dc,
        volatility=vol,
    )
    return GeneralizedBlackScholesProcess(
        x0=s, dividend_ts=q, risk_free_ts=r, black_vol_ts=v
    )


@pytest.fixture(scope="module")
def basket3(today: Date) -> StochasticProcessArray:
    """3-asset basket — matches the probe.cpp configuration."""
    rate = 0.05
    div = 0.02
    procs = [
        _make_bsm(today, 100.0, rate, div, 0.20),
        _make_bsm(today, 95.0, rate, div, 0.25),
        _make_bsm(today, 105.0, rate, div, 0.30),
    ]
    corr = np.array(
        [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    return StochasticProcessArray(procs, corr)


def make_bsm_for_two_asset(
    today: Date, spot: float, rate: float, div: float, vol: float
) -> GeneralizedBlackScholesProcess:
    """Helper used by 2-asset analytic-engine tests."""
    return _make_bsm(today, spot, rate, div, vol)
