"""EquityOptions sample — vanilla equity-option pricing via many methods.

Port of QuantLib's ``Examples/EquityOption`` (Java ``EquityOptions.java``, which
the Java AllSamples kept in its ``incomplete`` bucket). Prices a 40-strike put on
a 36 underlying under European / Bermudan / American exercise using every method
pquantlib ships:

* Black-Scholes analytic (European);
* the four binomial trees (Jarrow-Rudd, Cox-Ross-Rubinstein, Tian,
  Leisen-Reimer) for all three exercise styles;
* finite differences (European / American / Bermudan);
* a crude Monte Carlo (European).

This sample is bucketed INCOMPLETE (mirroring the Java original): the American
analytic *approximations* of the C++ example — Barone-Adesi/Whaley,
Bjerksund/Stensland, Ju Quadratic and the IntegralEngine — are not ported in
pquantlib, so those rows are printed as ``N/A``. Everything else runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import AmericanExercise, BermudanExercise, EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.binomial_engine import (
    BinomialVanillaEngine,
    TreeBuilder,
)
from pquantlib.pricingengines.vanilla.fd_black_scholes_vanilla_engine import (
    FdBlackScholesVanillaEngine,
)
from pquantlib.pricingengines.vanilla.mc_european_engine import MCEuropeanEngine
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from pquantlib_samples.util.stop_clock import StopClock

# value used where a method does not apply to a given exercise style.
_NA = float("nan")

_TREES: tuple[tuple[str, TreeBuilder], ...] = (
    ("Binomial Jarrow-Rudd", TreeBuilder.JarrowRudd),
    ("Binomial Cox-Ross-Rubinstein", TreeBuilder.CoxRossRubinstein),
    ("Binomial Tian", TreeBuilder.Tian),
    ("Binomial Leisen-Reimer", TreeBuilder.LeisenReimer),
)


@dataclass(frozen=True, slots=True)
class MethodRow:
    """One method row: European / Bermudan / American NPVs (NaN = N/A)."""

    method: str
    european: float
    bermudan: float
    american: float


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    rows: list[MethodRow] = field(default_factory=list[MethodRow])


def compute() -> EquityResult:
    calendar = TARGET()
    todays_date = Date.from_ymd(15, Month.May, 1998)
    settlement_date = Date.from_ymd(17, Month.May, 1998)
    ObservableSettings().evaluation_date = todays_date

    option_type = OptionType.Put
    strike = 40.0
    underlying = 36.0
    risk_free_rate = 0.06
    volatility = 0.20
    dividend_yield = 0.00

    maturity = Date.from_ymd(17, Month.May, 1999)
    dc = Actual365Fixed()

    european_exercise = EuropeanExercise(maturity)

    exercise_dates = [settlement_date + Period(3 * i, TimeUnit.Months) for i in range(1, 5)]
    bermudan_exercise = BermudanExercise(exercise_dates)
    american_exercise = AmericanExercise(settlement_date, maturity)

    underlying_h = SimpleQuote(underlying)
    flat_dividend_ts = FlatForward.from_rate(settlement_date, dividend_yield, dc)
    flat_term_structure = FlatForward.from_rate(settlement_date, risk_free_rate, dc)
    flat_vol_ts = BlackConstantVol(
        reference_date=settlement_date,
        calendar=calendar,
        day_counter=dc,
        volatility=volatility,
    )
    payoff = PlainVanillaPayoff(option_type, strike)

    bsm_process = BlackScholesMertonProcess(
        x0=underlying_h,
        dividend_ts=flat_dividend_ts,
        risk_free_ts=flat_term_structure,
        black_vol_ts=flat_vol_ts,
    )

    european_option = EuropeanOption(payoff, european_exercise)
    american_option = VanillaOption(payoff, american_exercise)

    # Java parity: EquityOptions.java guards every Bermudan NPV behind an
    # ``EXPERIMENTAL`` system property and otherwise leaves the column NaN; the
    # Bermudan exercise is built (above) but not priced, matching the default
    # Java run. (pquantlib's FD engine also does not yet support Bermudan.)
    _ = bermudan_exercise

    rows: list[MethodRow] = []

    # Black-Scholes (European only).
    european_option.set_pricing_engine(AnalyticEuropeanEngine(bsm_process))
    rows.append(MethodRow("Black-Scholes", european_option.npv(), _NA, _NA))

    # American analytic approximations — not ported in pquantlib.
    for m in ("Barone-Adesi/Whaley", "Bjerksund/Stensland", "Ju Quadratic", "Integral"):
        rows.append(MethodRow(f"{m} (N/A in pquantlib)", _NA, _NA, _NA))

    time_steps = 801

    # Binomial trees (European + American; Bermudan left N/A — see above).
    for name, builder in _TREES:
        european_option.set_pricing_engine(BinomialVanillaEngine(bsm_process, time_steps, builder))
        american_option.set_pricing_engine(BinomialVanillaEngine(bsm_process, time_steps, builder))
        rows.append(
            MethodRow(
                name,
                european_option.npv(),
                _NA,
                american_option.npv(),
            )
        )

    # Finite differences (European + American).
    european_option.set_pricing_engine(FdBlackScholesVanillaEngine(bsm_process, time_steps, time_steps - 1))
    american_option.set_pricing_engine(FdBlackScholesVanillaEngine(bsm_process, time_steps, time_steps - 1))
    rows.append(
        MethodRow(
            "Finite differences",
            european_option.npv(),
            _NA,
            american_option.npv(),
        )
    )

    # Monte Carlo (crude) — European only.
    european_option.set_pricing_engine(
        MCEuropeanEngine(
            bsm_process,
            time_steps=1,
            antithetic_variate=True,
            required_tolerance=0.02,
            max_samples=1_048_576,
            seed=42,
        )
    )
    rows.append(MethodRow("Monte Carlo (crude)", european_option.npv(), _NA, _NA))

    return EquityResult(rows=rows)


def _fmt(x: float) -> str:
    return "          N/A" if math.isnan(x) else f"{x:13.9f}"


def run() -> None:
    print("::::: EquityOptions :::::")

    clock = StopClock()
    clock.start_clock()

    r = compute()
    print(f"{'Method':>34} {'European':>13} {'Bermudan':>13} {'American':>13}")
    print("================================== ============= ============= =============")
    for row in r.rows:
        print(f"{row.method:>34} {_fmt(row.european)} {_fmt(row.bermudan)} {_fmt(row.american)}")

    clock.stop_clock()
    clock.log()


if __name__ == "__main__":
    run()
