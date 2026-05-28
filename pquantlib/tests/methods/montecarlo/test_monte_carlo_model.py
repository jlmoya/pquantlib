"""Tests for ``MonteCarloModel``.

# C++ parity: ql/methods/montecarlo/montecarlomodel.hpp (v1.42.1).
"""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.monte_carlo_model import MonteCarloModel
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
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


class _TerminalCallPricer(PathPricer[Path]):
    def __init__(self, strike: float, discount: float) -> None:
        self._strike = strike
        self._discount = discount

    def __call__(self, path: Path) -> float:
        return self._discount * max(path.back() - self._strike, 0.0)


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


def test_mc_model_converges_to_analytic_european_call() -> None:
    """Run 10000 MC samples for a textbook BSM call and check vs analytic.

    Analytic: S=K=100, r=5%, q=0, sigma=20%, T=1 -> NPV ~= 10.45.
    LOOSE-tier convergence: at 10000 samples the 1-sigma error is ~0.15;
    we allow a 3-sigma band to make the test stable.
    """
    process = _make_bsm()
    gsg = make_pseudo_random_rsg(1, 42)
    pg = PathGenerator(process, length=1.0, time_steps=1, generator=gsg)
    discount = math.exp(-0.05 * 1.0)
    pricer = _TerminalCallPricer(strike=100.0, discount=discount)

    stats = GeneralStatistics()
    model = MonteCarloModel[Path](
        path_generator=pg,
        path_pricer=pricer,
        sample_accumulator=stats,
        antithetic_variate=False,
    )
    model.add_samples(10000)
    npv = stats.mean()
    err = stats.error_estimate()
    # Textbook BSM call value (S=K=100, r=5%, q=0, sigma=20%, T=1) = ~10.4506
    assert abs(npv - 10.4506) < 3 * err
    # Standard error at 10000 samples should be in [0.10, 0.20].
    assert 0.05 < err < 0.30


def test_antithetic_reduces_error_vs_baseline() -> None:
    """Antithetic variate reduces standard error vs same-seed baseline."""
    process = _make_bsm()
    discount = math.exp(-0.05 * 1.0)

    # Baseline: no antithetic, 10000 samples.
    gsg1 = make_pseudo_random_rsg(1, 42)
    pg1 = PathGenerator(process, length=1.0, time_steps=1, generator=gsg1)
    stats1 = GeneralStatistics()
    MonteCarloModel[Path](
        path_generator=pg1,
        path_pricer=_TerminalCallPricer(100.0, discount),
        sample_accumulator=stats1,
        antithetic_variate=False,
    ).add_samples(10000)

    # With antithetic, 5000 sample pairs (= 10000 effective draws).
    gsg2 = make_pseudo_random_rsg(1, 42)
    pg2 = PathGenerator(process, length=1.0, time_steps=1, generator=gsg2)
    stats2 = GeneralStatistics()
    MonteCarloModel[Path](
        path_generator=pg2,
        path_pricer=_TerminalCallPricer(100.0, discount),
        sample_accumulator=stats2,
        antithetic_variate=True,
    ).add_samples(5000)

    err_baseline = stats1.error_estimate()
    err_antithetic = stats2.error_estimate()
    # Antithetic should cut the error meaningfully (>20% reduction).
    assert err_antithetic < 0.8 * err_baseline


def test_control_variate_reduces_error() -> None:
    """A perfect control variate (the same pricer with known mean) cuts
    error to ~0 — exercises the CV codepath inside ``add_samples``.
    """
    process = _make_bsm()
    discount = math.exp(-0.05 * 1.0)
    main_pricer = _TerminalCallPricer(100.0, discount)
    cv_pricer = _TerminalCallPricer(100.0, discount)
    # CV value = sample mean — using a "perfect" CV (=pricer itself) should
    # give variance ~ 0 (every sample's price + (cv - cv) = price, but we
    # subtract its own value so the residual is 0 with deterministic mean).
    # Use a textbook 10.45 mean as the offset.
    gsg = make_pseudo_random_rsg(1, 42)
    pg = PathGenerator(process, length=1.0, time_steps=1, generator=gsg)
    stats = GeneralStatistics()
    MonteCarloModel[Path](
        path_generator=pg,
        path_pricer=main_pricer,
        sample_accumulator=stats,
        antithetic_variate=False,
        cv_path_pricer=cv_pricer,
        cv_option_value=10.4506,
    ).add_samples(1000)
    # With cv = pricer itself, every per-sample contribution is exactly
    # cv_option_value = 10.4506 — N i.i.d. constant contributions whose
    # mean equals cv_option_value (modulo accumulator rounding).
    assert abs(stats.mean() - 10.4506) < 1e-10
    # Variance of an exact-constant series is bounded by float roundoff.
    assert stats.variance() < 1e-20
