"""Unit tests for ``StochasticProcessArray``.

Independent of any reference JSON — verifies algebraic invariants
(recovered correlation = input correlation; ``initial_values``
slice-equality; size/factors).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
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
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _make_bsm(today: Date, spot: float, vol: float) -> GeneralizedBlackScholesProcess:
    dc = Actual360()
    s = SimpleQuote(spot)
    r = FlatForward.from_rate(today, 0.05, dc)
    q = FlatForward.from_rate(today, 0.02, dc)
    v = BlackConstantVol(
        reference_date=today, calendar=NullCalendar(), day_counter=dc, volatility=vol
    )
    return GeneralizedBlackScholesProcess(
        x0=s, dividend_ts=q, risk_free_ts=r, black_vol_ts=v
    )


@pytest.fixture
def today() -> Date:
    return Date.from_ymd(15, Month.January, 2024)


def test_size_and_factors(today: Date) -> None:
    """size == len(processes), factors == size (default)."""
    procs = [_make_bsm(today, 100.0, 0.20), _make_bsm(today, 95.0, 0.25)]
    corr = np.eye(2)
    arr = StochasticProcessArray(procs, corr)
    assert arr.size() == 2
    assert arr.factors() == 2


def test_initial_values_match_underlyings(today: Date) -> None:
    """initial_values() returns each process's x0 in order."""
    procs = [_make_bsm(today, 100.0, 0.20), _make_bsm(today, 95.0, 0.25), _make_bsm(today, 105.0, 0.30)]
    corr = np.eye(3)
    arr = StochasticProcessArray(procs, corr)
    iv = arr.initial_values()
    assert iv[0] == 100.0
    assert iv[1] == 95.0
    assert iv[2] == 105.0


def test_correlation_recovery(today: Date) -> None:
    """``correlation() == input correlation`` (spectral sqrt is idempotent
    on the proper correlation matrix)."""
    procs = [_make_bsm(today, 100.0, 0.20), _make_bsm(today, 95.0, 0.25), _make_bsm(today, 105.0, 0.30)]
    corr = np.array(
        [
            [1.0, 0.3, 0.5],
            [0.3, 1.0, 0.2],
            [0.5, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    arr = StochasticProcessArray(procs, corr)
    recovered = arr.correlation()
    np.testing.assert_allclose(recovered, corr, atol=1e-13)


def test_correlation_size_mismatch_raises(today: Date) -> None:
    procs = [_make_bsm(today, 100.0, 0.20), _make_bsm(today, 95.0, 0.25)]
    bad_corr = np.eye(3)
    with pytest.raises(LibraryException):
        StochasticProcessArray(procs, bad_corr)


def test_empty_processes_raises(today: Date) -> None:
    """At least one process required."""
    with pytest.raises(LibraryException):
        StochasticProcessArray([], np.eye(0))


def test_process_accessor(today: Date) -> None:
    """``process(i)`` returns the i-th constituent."""
    procs = [_make_bsm(today, 100.0, 0.20), _make_bsm(today, 95.0, 0.25)]
    arr = StochasticProcessArray(procs, np.eye(2))
    assert arr.process(0) is procs[0]
    assert arr.process(1) is procs[1]


def test_time_delegates_to_first(today: Date) -> None:
    """time(date) is delegated to processes[0].time(date)."""
    procs = [_make_bsm(today, 100.0, 0.20), _make_bsm(today, 95.0, 0.25)]
    arr = StochasticProcessArray(procs, np.eye(2))
    later = today + 365  # int = days
    t_arr = arr.time(later)
    t_p0 = procs[0].time(later)
    assert t_arr == t_p0


def test_evolve_with_independent_inputs_is_componentwise(today: Date) -> None:
    """For identity correlation, evolve(dw) component-wise equals
    each 1-D process's evolve_1d."""
    procs = [_make_bsm(today, 100.0, 0.20), _make_bsm(today, 95.0, 0.25)]
    arr = StochasticProcessArray(procs, np.eye(2))
    x0 = np.array([100.0, 95.0], dtype=np.float64)
    dw = np.array([0.5, -0.3], dtype=np.float64)
    out = arr.evolve(0.0, x0, 0.1, dw)
    expected0 = procs[0].evolve_1d(0.0, 100.0, 0.1, 0.5)
    expected1 = procs[1].evolve_1d(0.0, 95.0, 0.1, -0.3)
    assert math.isclose(out[0], expected0, rel_tol=1e-14)
    assert math.isclose(out[1], expected1, rel_tol=1e-14)
