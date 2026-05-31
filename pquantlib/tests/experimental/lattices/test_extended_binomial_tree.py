"""Tests for the ExtendedBinomialTree family (W8-C, batch d).

# C++ parity: ql/experimental/lattices/extendedbinomialtree.{hpp,cpp}
#             @ v1.42.1.

The extended trees recompute drift / std / variance per slice. Under a
constant-coefficient Black-Scholes-Merton process they reduce to the
plain binomial trees, so their European value converges to the analytic
Black-Scholes price. We verify convergence (LOOSE / per-scheme tier)
rather than cross-validating a single C++ probe value, since the
closed-form BS price is itself the exact reference.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.lattices.extended_binomial_tree import (
    ExtendedAdditiveEQPBinomialTree,
    ExtendedBinomialTree,
    ExtendedCoxRossRubinstein,
    ExtendedJarrowRudd,
    ExtendedJoshi4,
    ExtendedLeisenReimer,
    ExtendedTian,
    ExtendedTrigeorgis,
)
from pquantlib.processes.black_scholes_merton_process import (
    BlackScholesMertonProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_S = 100.0
_K = 100.0
_R = 0.05
_Q = 0.02
_SIGMA = 0.25
_T = 1.0


def _process() -> StochasticProcess1D:
    today = Date.from_ymd(15, Month.January, 2024)
    dc = Actual365Fixed()
    spot = SimpleQuote(_S)
    r_ts = FlatForward.from_rate(reference_date=today, forward_rate=_R, day_counter=dc)
    q_ts = FlatForward.from_rate(reference_date=today, forward_rate=_Q, day_counter=dc)
    vol = BlackConstantVol(
        reference_date=today, calendar=NullCalendar(), day_counter=dc, volatility=_SIGMA
    )
    return BlackScholesMertonProcess(
        x0=spot, dividend_ts=q_ts, risk_free_ts=r_ts, black_vol_ts=vol
    )


def _bs_call() -> float:
    d1 = (math.log(_S / _K) + (_R - _Q + 0.5 * _SIGMA**2) * _T) / (_SIGMA * math.sqrt(_T))
    d2 = d1 - _SIGMA * math.sqrt(_T)
    n = NormalDist().cdf
    return _S * math.exp(-_Q * _T) * n(d1) - _K * math.exp(-_R * _T) * n(d2)


def _price_european_call(tree: ExtendedBinomialTree) -> float:
    n = tree.columns() - 1
    dt = _T / n
    disc = math.exp(-_R * dt)
    vals = np.array([max(tree.underlying(n, j) - _K, 0.0) for j in range(tree.size(n))])
    for i in range(n - 1, -1, -1):
        nxt = np.empty(tree.size(i))
        for j in range(tree.size(i)):
            pu = tree.probability(i, j, 1)
            pd = tree.probability(i, j, 0)
            nxt[j] = disc * (pd * vals[j] + pu * vals[j + 1])
        vals = nxt
    return float(vals[0])


# Each scheme converges at a different rate; tolerances reflect the
# expected O(1/N) (or faster, for LR/Joshi) error at the given step count.
_SCHEMES = [
    (ExtendedCoxRossRubinstein, 501, 1e-2),
    (ExtendedJarrowRudd, 501, 1e-2),
    (ExtendedAdditiveEQPBinomialTree, 501, 1e-2),
    (ExtendedTrigeorgis, 501, 1e-2),
    (ExtendedTian, 501, 1e-2),
    (ExtendedLeisenReimer, 501, 1e-4),
    (ExtendedJoshi4, 501, 1e-4),
]


@pytest.mark.parametrize(("tree_cls", "steps", "tol"), _SCHEMES)
def test_extended_tree_converges_to_bs(
    tree_cls: type[ExtendedBinomialTree], steps: int, tol: float
) -> None:
    tree = tree_cls(_process(), _T, steps, _K)
    price = _price_european_call(tree)
    assert abs(price - _bs_call()) < tol


def test_extended_leisen_reimer_high_accuracy() -> None:
    # LR is the recommended scheme: << 1e-4 even at moderate step counts.
    tree = ExtendedLeisenReimer(_process(), _T, 301, _K)
    assert abs(_price_european_call(tree) - _bs_call()) < 1e-4


def test_extended_tree_size_and_descendant() -> None:
    tree = ExtendedCoxRossRubinstein(_process(), _T, 10, _K)
    assert tree.columns() == 11
    assert tree.size(0) == 1
    assert tree.size(5) == 6
    # Binomial descendant: index + branch.
    assert tree.descendant(3, 2, 0) == 2
    assert tree.descendant(3, 2, 1) == 3


def test_extended_equal_probabilities_are_half() -> None:
    tree = ExtendedJarrowRudd(_process(), _T, 20, _K)
    for branch in (0, 1):
        assert tree.probability(5, 2, branch) == 0.5


def test_extended_crr_probabilities_sum_to_one() -> None:
    tree = ExtendedCoxRossRubinstein(_process(), _T, 20, _K)
    # Time-dependent probability still sums to 1 at any slice.
    assert abs(tree.probability(7, 3, 0) + tree.probability(7, 3, 1) - 1.0) < 1e-12
