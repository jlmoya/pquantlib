"""ConvertibleBonds sample — Tsiveriotis-Fernandes convertible-bond pricing.

Port of ``org.jquantlib.samples.ConvertibleBonds`` (itself a port of QuantLib's
``Examples/ConvertibleBonds/ConvertibleBonds.cpp``). Builds a 5-year annual
fixed-coupon convertible with a soft-call schedule (years 2, 4), a put (year 3)
and semiannual fixed dividends, then prices both a European-exercise and an
American-exercise flavour through :class:`BinomialConvertibleEngine` over each
available binomial tree (Jarrow-Rudd, Cox-Ross-Rubinstein, Tian, Leisen-Reimer).

Unblocked by W-S4, which landed ``ConvertibleFixedCouponBond`` +
``BinomialConvertibleEngine`` + ``SoftCallability`` into pquantlib core.

Divergences from the Java/C++ originals (documented inline):

* PQuantLib's :class:`ConvertibleFixedCouponBond` takes the dividend schedule on
  the *engine* (``BinomialConvertibleEngine(..., dividends=...)``), not the bond
  constructor — mirroring the C++ v1.42.1 split where the engine subtracts
  discrete dividends from the spot.
* pquantlib ships four of the C++ binomial trees (Jarrow-Rudd,
  Cox-Ross-Rubinstein, Tian, Leisen-Reimer); the additive-equiprobabilities,
  Trigeorgis and Joshi4 variants are not ported, so those rows are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib.cashflows.dividend import Dividend, FixedDividend
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exercise import AmericanExercise, EuropeanExercise, Exercise
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.bonds.convertible_bonds import ConvertibleFixedCouponBond
from pquantlib.instruments.callability import Callability, CallabilityType
from pquantlib.instruments.soft_callability import SoftCallability
from pquantlib.methods.lattices.binomial_tree import (
    CoxRossRubinstein,
    JarrowRudd,
    LeisenReimer,
    Tian,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.binomial_convertible_engine import (
    BinomialConvertibleEngine,
    ConvertibleTreeBuilder,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit
from pquantlib_samples.util.stop_clock import StopClock

# Tree builders, in the C++ print order (minus the three not ported).
_TREES: tuple[tuple[str, ConvertibleTreeBuilder], ...] = (
    ("Jarrow-Rudd", JarrowRudd),
    ("Cox-Ross-Rubinstein", CoxRossRubinstein),
    ("Tian", Tian),
    ("Leisen-Reimer", LeisenReimer),
)


@dataclass(frozen=True, slots=True)
class ConvertibleResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    maturity: float
    underlying: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    # tree-name -> (european NPV, american NPV)
    npvs: tuple[tuple[str, float, float], ...]


def _build_callability(schedule: Schedule) -> list[Callability]:
    """Soft calls at years 2 & 4, a plain put at year 3 (C++ scenario)."""
    call_length = [2, 4]
    call_prices = [101.5, 100.85]
    put_length = [3]
    put_prices = [105.0]
    out: list[Callability] = [
        SoftCallability(BondPrice(call_prices[i], BondPriceType.Clean), schedule.date(cl), 1.20)
        for i, cl in enumerate(call_length)
    ]
    out.extend(
        Callability(
            BondPrice(put_prices[j], BondPriceType.Clean),
            CallabilityType.Put,
            schedule.date(pl),
        )
        for j, pl in enumerate(put_length)
    )
    return out


def _build_dividends(today: Date, exercise_date: Date) -> list[Dividend]:
    """Fixed 1.0 dividends every 6 months from today up to exercise."""
    out: list[Dividend] = []
    d = today + Period(6, TimeUnit.Months)
    while d < exercise_date:
        out.append(FixedDividend(1.0, d))
        d = d + Period(6, TimeUnit.Months)
    return out


def compute() -> ConvertibleResult:
    underlying = 36.0
    spread_rate = 0.005
    dividend_yield = 0.02
    risk_free_rate = 0.06
    volatility = 0.20

    settlement_days = 3
    length = 5
    redemption = 100.0
    conversion_ratio = redemption / underlying

    calendar = TARGET()
    today = calendar.adjust(Date.todays_date())
    ObservableSettings().evaluation_date = today

    settlement_date = calendar.advance(today, settlement_days, TimeUnit.Days)
    exercise_date = calendar.advance(settlement_date, length, TimeUnit.Years)
    issue_date = calendar.advance(exercise_date, -length, TimeUnit.Years)

    convention = BusinessDayConvention.ModifiedFollowing
    frequency = Frequency.Annual

    schedule = Schedule.from_rule(
        issue_date,
        exercise_date,
        Period.from_frequency(frequency),
        calendar,
        convention,
        convention,
        DateGeneration.Backward,
        False,
    )

    coupons = [0.05]
    bond_day_count = Thirty360(Convention.BondBasis)

    callability = _build_callability(schedule)
    dividends = _build_dividends(today, exercise_date)

    day_counter = Actual365Fixed()
    maturity = day_counter.year_fraction(settlement_date, exercise_date)

    european_exercise = EuropeanExercise(exercise_date)
    american_exercise = AmericanExercise(settlement_date, exercise_date)

    underlying_h = SimpleQuote(underlying)
    flat_term_structure = FlatForward.from_rate(settlement_date, risk_free_rate, day_counter)
    flat_dividend_ts = FlatForward.from_rate(settlement_date, dividend_yield, day_counter)
    flat_vol_ts = BlackConstantVol(
        reference_date=settlement_date,
        calendar=calendar,
        day_counter=day_counter,
        volatility=volatility,
    )
    process = BlackScholesMertonProcess(
        x0=underlying_h,
        dividend_ts=flat_dividend_ts,
        risk_free_ts=flat_term_structure,
        black_vol_ts=flat_vol_ts,
    )

    time_steps = 801
    credit_spread = SimpleQuote(spread_rate)

    def make_bond(exercise: Exercise) -> ConvertibleFixedCouponBond:
        return ConvertibleFixedCouponBond(
            exercise=exercise,
            conversion_ratio=conversion_ratio,
            callability=callability,
            issue_date=issue_date,
            settlement_days=settlement_days,
            coupons=coupons,
            day_counter=bond_day_count,
            schedule=schedule,
            redemption=redemption,
        )

    npvs: list[tuple[str, float, float]] = []
    for name, tree in _TREES:
        european_bond = make_bond(european_exercise)
        american_bond = make_bond(american_exercise)
        eng_eu = BinomialConvertibleEngine(
            tree,
            process,
            time_steps,
            credit_spread,
            dividends=dividends,
        )
        eng_am = BinomialConvertibleEngine(
            tree,
            process,
            time_steps,
            credit_spread,
            dividends=dividends,
        )
        european_bond.set_pricing_engine(eng_eu)
        american_bond.set_pricing_engine(eng_am)
        npvs.append((name, european_bond.npv(), american_bond.npv()))

    return ConvertibleResult(
        maturity=maturity,
        underlying=underlying,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        npvs=tuple(npvs),
    )


def run() -> None:
    print("::::: ConvertibleBonds :::::")

    clock = StopClock()
    clock.start_clock()

    r = compute()

    print(f"Time to maturity        = {r.maturity}")
    print(f"Underlying price        = {r.underlying}")
    print(f"Risk-free interest rate = {r.risk_free_rate}")
    print(f"Dividend yield          = {r.dividend_yield}")
    print(f"Volatility              = {r.volatility}")
    print()
    print()
    print("Tsiveriotis-Fernandes method")
    print(f"{'Tree type':>34} {'European':>13} {'American':>13}")
    print("================================== ============= =============")
    for name, eu, am in r.npvs:
        print(f"{name:>34} {eu:13.9f} {am:13.9f}")

    clock.stop_clock()
    clock.log()


if __name__ == "__main__":
    run()
