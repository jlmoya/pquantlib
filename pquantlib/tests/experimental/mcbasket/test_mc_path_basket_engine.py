"""Tests for the experimental mcbasket framework (W8-C, batch c).

# C++ parity: ql/experimental/mcbasket/* @ v1.42.1.

Covers PathPayoff / AdaptedPathPayoff / PathMultiAssetOption +
MCPathBasketEngine (European). We validate the engine by:

  * a 1-asset European call basket converging to the analytic
    Black-Scholes price (within the MC error estimate);
  * a 2-asset correlated average-basket call producing a finite,
    diversified value below the single-asset ATM call;
  * the AdaptedPathPayoff "looking into the future" guard.

We deliberately do NOT cross-validate a fixed-seed C++ MC value: the
C++ and Python pseudo-random sequences differ, so seed-for-seed match
is not meaningful. Convergence-to-analytic is the robust reference.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.mcbasket.adapted_path_payoff import (
    AdaptedPathPayoff,
    ValuationData,
)
from pquantlib.experimental.mcbasket.mc_path_basket_engine import MCPathBasketEngine
from pquantlib.experimental.mcbasket.path_multi_asset_option import (
    PathMultiAssetOption,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.processes.black_scholes_merton_process import (
    BlackScholesMertonProcess,
)
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _today() -> Date:
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today
    return today


def _bsm(today: Date, spot: float, sigma: float) -> BlackScholesMertonProcess:
    dc = Actual365Fixed()
    r_ts = FlatForward.from_rate(reference_date=today, forward_rate=0.05, day_counter=dc)
    q_ts = FlatForward.from_rate(reference_date=today, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(
        reference_date=today, calendar=NullCalendar(), day_counter=dc, volatility=sigma
    )
    return BlackScholesMertonProcess(
        x0=SimpleQuote(spot), dividend_ts=q_ts, risk_free_ts=r_ts, black_vol_ts=vol
    )


class _CallAtLastFixing(AdaptedPathPayoff):
    """max(S_asset0(T) - K, 0) paid at the last fixing."""

    def __init__(self, strike: float) -> None:
        self._k = strike

    def name(self) -> str:
        return "call-last"

    def description(self) -> str:
        return "European call on last fixing"

    def basis_system_dimension(self) -> int:
        return 1

    def _evaluate(self, data: ValuationData) -> None:
        last = data.number_of_times() - 1
        s = data.get_asset_value(last, 0)
        data.set_payoff_value(last, max(s - self._k, 0.0))


class _AverageBasketCall(AdaptedPathPayoff):
    """max(mean of all assets at T - K, 0)."""

    def __init__(self, strike: float) -> None:
        self._k = strike

    def name(self) -> str:
        return "avg-basket-call"

    def description(self) -> str:
        return "average basket call"

    def basis_system_dimension(self) -> int:
        return 1

    def _evaluate(self, data: ValuationData) -> None:
        last = data.number_of_times() - 1
        n = data.number_of_assets()
        avg = sum(data.get_asset_value(last, j) for j in range(n)) / n
        data.set_payoff_value(last, max(avg - self._k, 0.0))


class _AnticipatingPayoff(AdaptedPathPayoff):
    """Illegally writes a payment that depends on a later fixing."""

    def name(self) -> str:
        return "bad"

    def description(self) -> str:
        return "anticipating"

    def basis_system_dimension(self) -> int:
        return 1

    def _evaluate(self, data: ValuationData) -> None:
        last = data.number_of_times() - 1
        # Read the LATEST fixing, then try to write the FIRST payment.
        _ = data.get_asset_value(last, 0)
        data.set_payoff_value(0, 1.0)  # must raise: looking into the future


def _bs_call(spot: float, strike: float, r: float, q: float, sigma: float, t: float) -> float:
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    n = NormalDist().cdf
    return spot * math.exp(-q * t) * n(d1) - strike * math.exp(-r * t) * n(d2)


def test_single_asset_european_converges_to_bs() -> None:
    today = _today()
    proc = _bsm(today, 100.0, 0.20)
    arr = StochasticProcessArray([proc], [[1.0]])
    opt = PathMultiAssetOption(_CallAtLastFixing(100.0), [today + 365])
    opt.set_pricing_engine(
        MCPathBasketEngine(arr, 1, None, False, False, False, 100_000, None, None, 42)
    )
    mc = opt.npv()
    bs = _bs_call(100.0, 100.0, 0.05, 0.0, 0.20, 1.0)
    # Within ~4 standard errors of the analytic price.
    assert abs(mc - bs) < 4.0 * opt.error_estimate()
    # And within a loose absolute band as a backstop.
    assert abs(mc - bs) < 0.2


def test_two_asset_basket_is_diversified() -> None:
    today = _today()
    arr = StochasticProcessArray(
        [_bsm(today, 100.0, 0.20), _bsm(today, 100.0, 0.20)],
        [[1.0, 0.5], [0.5, 1.0]],
    )
    opt = PathMultiAssetOption(_AverageBasketCall(100.0), [today + 365])
    opt.set_pricing_engine(
        MCPathBasketEngine(arr, 1, None, False, True, False, 80_000, None, None, 123)
    )
    mc = opt.npv()
    single = _bs_call(100.0, 100.0, 0.05, 0.0, 0.20, 1.0)
    # Imperfect correlation => basket vol < 20% => price below the single ATM call.
    assert 0.0 < mc < single


def test_adapted_payoff_guard_raises() -> None:
    today = _today()
    proc = _bsm(today, 100.0, 0.20)
    arr = StochasticProcessArray([proc], [[1.0]])
    # Two fixings so "write first after reading last" is genuinely anticipating.
    opt = PathMultiAssetOption(_AnticipatingPayoff(), [today + 182, today + 365])
    opt.set_pricing_engine(
        MCPathBasketEngine(arr, 2, None, False, False, False, 1_023, None, None, 7)
    )
    with pytest.raises(LibraryException, match="looking into the future"):
        opt.npv()


def test_valuation_data_accessors() -> None:
    # Direct ValuationData unit test (no MC).
    path = np.array([[100.0, 110.0], [100.0, 90.0]], dtype=np.float64)  # 2 assets x 2 times
    payments = np.zeros(2, dtype=np.float64)
    exercises = np.empty(0, dtype=np.float64)
    states: list[np.ndarray] = []
    today = _today()
    r_ts = FlatForward.from_rate(
        reference_date=today, forward_rate=0.05, day_counter=Actual365Fixed()
    )
    data = ValuationData(path, [r_ts, r_ts], payments, exercises, states)
    assert data.number_of_times() == 2
    assert data.number_of_assets() == 2
    assert data.get_asset_value(1, 0) == 110.0
    data.set_payoff_value(1, 42.0)
    assert payments[1] == 42.0
