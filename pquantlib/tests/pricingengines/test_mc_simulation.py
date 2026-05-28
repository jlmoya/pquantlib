"""Tests for ``pquantlib.pricingengines.mc_simulation.McSimulation``.

# C++ parity: ql/pricingengines/mcsimulation.hpp (v1.42.1).

Builds a minimal ``McSimulation`` subclass mimicking MCEuropeanEngine
and exercises the ``calculate(tolerance=..., samples=...)`` paths.
"""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


def _make_bsm() -> GeneralizedBlackScholesProcess:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.May, 2026)
    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(
        reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20
    )
    return GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )


class _CallPricer(PathPricer[Path]):
    def __init__(self, strike: float, discount: float) -> None:
        self._strike = strike
        self._discount = discount

    def __call__(self, path: Path) -> float:
        return self._discount * max(path.back() - self._strike, 0.0)


class _ToyMcEuropeanEngine(McSimulation[Path]):
    """Minimal concrete McSimulation — drops in a single-step Euler MC for a
    textbook BSM European call.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        strike: float,
        discount: float,
        seed: int,
        antithetic: bool = False,
    ) -> None:
        super().__init__(antithetic_variate=antithetic, control_variate=False)
        self._process = process
        self._strike = strike
        self._discount = discount
        self._seed = seed

    def time_grid(self) -> TimeGrid:
        return TimeGrid.regular(1.0, 1)

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        gsg = make_pseudo_random_rsg(1, self._seed)
        return PathGenerator(self._process, length=1.0, time_steps=1, generator=gsg)

    def path_pricer(self) -> PathPricer[Path]:
        return _CallPricer(self._strike, self._discount)


def test_calculate_with_required_samples_runs_to_target_count() -> None:
    process = _make_bsm()
    discount = math.exp(-0.05 * 1.0)
    engine = _ToyMcEuropeanEngine(process, 100.0, discount, seed=42)
    engine.calculate(required_tolerance=None, required_samples=10000, max_samples=None)
    npv = engine.sample_accumulator().mean()
    err = engine.error_estimate()
    assert engine.sample_accumulator().samples() == 10000
    # Textbook BSM call ~= 10.4506; 3-sigma band at 10k.
    assert abs(npv - 10.4506) < 3 * err


def test_calculate_with_required_tolerance_converges() -> None:
    process = _make_bsm()
    discount = math.exp(-0.05 * 1.0)
    engine = _ToyMcEuropeanEngine(process, 100.0, discount, seed=42)
    engine.calculate(
        required_tolerance=0.05, required_samples=None, max_samples=100_000
    )
    err = engine.error_estimate()
    assert err <= 0.05
    # Error <= 0.05 means we ran ~16k+ samples (~ (0.20/0.05)^2 * 1023 ≈ 16k).
    assert engine.sample_accumulator().samples() >= 1023


def test_calculate_with_neither_target_raises() -> None:
    process = _make_bsm()
    discount = math.exp(-0.05 * 1.0)
    engine = _ToyMcEuropeanEngine(process, 100.0, discount, seed=42)
    with pytest.raises(LibraryException, match="neither tolerance nor number"):
        engine.calculate(required_tolerance=None, required_samples=None, max_samples=None)


def test_value_with_samples_starts_from_min_default() -> None:
    """``value(tol)`` with no prior samples should kick off at min_samples=1023."""
    process = _make_bsm()
    discount = math.exp(-0.05 * 1.0)
    engine = _ToyMcEuropeanEngine(process, 100.0, discount, seed=42)
    # Use a generous tolerance that 1023 samples should satisfy immediately.
    engine.calculate(required_tolerance=1.0, required_samples=None, max_samples=None)
    assert engine.sample_accumulator().samples() >= 1023


def test_value_with_max_samples_raises_on_unreachable_tolerance() -> None:
    process = _make_bsm()
    discount = math.exp(-0.05 * 1.0)
    engine = _ToyMcEuropeanEngine(process, 100.0, discount, seed=42)
    with pytest.raises(LibraryException, match="max number of samples"):
        engine.calculate(
            required_tolerance=1e-6, required_samples=None, max_samples=2000
        )
