"""ReplicationError — Monte Carlo evaluation of a discrete-hedging strategy.

Port of QuantLib's ``Examples/DiscreteHedging`` ``ReplicationError`` (the Java
``util/ReplicationError.java`` was a "Work in progress" stub guarded by an
``EXPERIMENTAL`` system property; this follows the complete C++ original).

The class carries out a Monte Carlo simulation over randomly generated stock
paths (Black-Scholes dynamics), pricing each path's discrete-hedging P&L with
:class:`ReplicationPathPricer`, and compares the resulting standard deviation
with Derman & Kamal's closed-form approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.monte_carlo_model import MonteCarloModel
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib_samples.util.replication_path_pricer import ReplicationPathPricer


@dataclass(frozen=True, slots=True)
class HedgeStats:
    """One row of the replication-error table."""

    samples: int
    trades: int
    pl_mean: float
    pl_std_dev: float
    derman_kamal: float
    pl_skew: float
    pl_kurt: float


class ReplicationError:
    """Monte-Carlo replication-error analysis (Derman & Kamal)."""

    def __init__(
        self,
        option_type: OptionType,
        maturity: float,
        strike: float,
        s0: float,
        sigma: float,
        r: float,
    ) -> None:
        self._maturity = maturity
        self._payoff = PlainVanillaPayoff(option_type, strike)
        self._s0 = s0
        self._sigma = sigma
        self._r = r

        r_discount = math.exp(-r * maturity)
        q_discount = 1.0
        forward = s0 * q_discount / r_discount
        std_dev = math.sqrt(sigma * sigma * maturity)
        black = BlackCalculator(self._payoff, forward, std_dev, r_discount)
        self.option_value = black.value()
        # store vega for Derman & Kamal's formula
        self._vega = black.vega(maturity)

    def compute(self, n_time_steps: int, n_samples: int) -> HedgeStats:
        if n_time_steps <= 0:
            raise ValueError("the number of steps must be > 0")

        calendar = TARGET()
        today = Date.todays_date()
        ObservableSettings().evaluation_date = today
        day_count = Actual365Fixed()

        state_variable = SimpleQuote(self._s0)
        risk_free_rate = FlatForward.from_rate(today, self._r, day_count)
        dividend_yield = FlatForward.from_rate(today, 0.0, day_count)
        volatility = BlackConstantVol(
            reference_date=today,
            calendar=calendar,
            day_counter=day_count,
            volatility=self._sigma,
        )
        diffusion = BlackScholesMertonProcess(
            x0=state_variable,
            dividend_ts=dividend_yield,
            risk_free_ts=risk_free_rate,
            black_vol_ts=volatility,
        )

        # C++ seeds with 0 (SeedGenerator); pquantlib's pseudo-random RSG
        # requires a nonzero seed, so we use a fixed 1 for reproducibility.
        rsg = make_pseudo_random_rsg(n_time_steps, 1)
        path_generator = PathGenerator(diffusion, self._maturity, n_time_steps, rsg, brownian_bridge=False)

        path_pricer = ReplicationPathPricer(
            self._payoff.option_type(),
            self._payoff.strike(),
            self._r,
            self._maturity,
            self._sigma,
        )

        stats = GeneralStatistics()
        model: MonteCarloModel[Path] = MonteCarloModel(path_generator, path_pricer, stats, False)
        model.add_samples(n_samples)

        acc = model.sample_accumulator()
        pl_mean = acc.mean()
        pl_std_dev = acc.standard_deviation()
        pl_skew = acc.skewness()
        pl_kurt = acc.kurtosis()

        theor_std = math.sqrt(math.pi / 4 / n_time_steps) * self._vega * self._sigma

        return HedgeStats(
            samples=n_samples,
            trades=n_time_steps,
            pl_mean=pl_mean,
            pl_std_dev=pl_std_dev,
            derman_kamal=theor_std,
            pl_skew=pl_skew,
            pl_kurt=pl_kurt,
        )
