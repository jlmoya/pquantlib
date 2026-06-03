"""Processes sample — explores ``GeneralizedBlackScholesProcess`` / Euler discretization.

Port of ``org.jquantlib.samples.Processes``. Builds a generalized Black-Scholes
process with a :class:`BlackVarianceCurve` vol term structure and flat
dividend/risk-free curves, then reads off the process's drift, diffusion,
standard deviation, variance, expectation and a single Euler ``evolve`` step at
t = 18 days from today.

PQuantLib's process exposes the 1-D scalar overrides (``drift_1d`` etc.) since
the generalized BS process is single-factor; the Java ``Array``-valued overloads
collapse to these scalar calls. The Euler ``evolve_1d`` is driven by a fixed
Gaussian draw (rather than ``Math.random()``) so the sample's output is
deterministic and smoke-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib_samples.util.stop_clock import StopClock


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    drift: float
    diffusion: float
    std_deviation: float
    variance: float
    expectation: float
    evolve: float


def compute() -> ProcessResult:
    today = Date.todays_date()
    date10 = today + 10
    date15 = today + 15
    date18 = today + 18
    date20 = today + 20
    date25 = today + 25
    date30 = today + 30
    date40 = today + 40

    day_counter = Actual365Fixed()

    stock_quote = SimpleQuote(5.6)

    # Black variance curve over the date axis.
    dates = [date10, date15, date20, date25, date30, date40]
    volatilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    variance_curve = BlackVarianceCurve(
        reference_date=today,
        dates=dates,
        black_vol_curve=volatilities,
        day_counter=day_counter,
        force_monotone_variance=False,
    )

    dividend_ts = FlatForward(
        today,
        SimpleQuote(0.3),
        day_counter,
        Compounding.Continuous,
        Frequency.Daily,
    )
    risk_free_ts = FlatForward(
        today,
        SimpleQuote(0.3),
        day_counter,
        Compounding.Continuous,
        Frequency.Daily,
    )

    process = GeneralizedBlackScholesProcess(
        x0=stock_quote,
        dividend_ts=dividend_ts,
        risk_free_ts=risk_free_ts,
        black_vol_ts=variance_curve,
    )

    t18 = process.time(date18)
    x = stock_quote.value()
    dt = 0.01

    # Deterministic Gaussian draw (median) keeps the evolve step reproducible.
    dw = 0.0

    return ProcessResult(
        drift=process.drift_1d(t18, x),
        diffusion=process.diffusion_1d(t18, x),
        std_deviation=process.std_deviation_1d(t18, x, dt),
        variance=process.variance_1d(t18, x, dt),
        expectation=process.expectation_1d(t18, x, dt),
        evolve=process.evolve_1d(t18, 6.7, 0.001, dw),
    )


def run() -> None:
    print("::::: Processes :::::")

    clock = StopClock()
    clock.start_clock()

    print("//============================StochasticProcess1D/LinearDiscretization=========================")
    r = compute()
    print(f"The drift of the process after time = 18th day from today = {r.drift}")
    print(f"The diffusion of the process after time = 18th day from today = {r.diffusion}")
    print(f"The stdDeviation of the process after time = 18th day from today = {r.std_deviation}")
    print(f"The variance of the process after time = 18th day from today = {r.variance}")
    print(f"Expected value = {r.expectation}")
    print(f"Exact value (one Euler step) = {r.evolve}")

    clock.stop_clock()
    clock.log()


if __name__ == "__main__":
    run()
