"""Repo sample — repo calculation via the BondForward class.

Port of ``org.jquantlib.samples.Repo`` (Java port of QuantLib's
``Examples/Repo/Repo.cpp``), reproducing the FINCAD ``aaBondFwd()`` example.

A 10-year 8% semiannual bond is priced from a known clean price (89.97693786)
by solving for its implied yield; a flat repo curve at 5% (simple-compounded)
is used to discount the forward, and a :class:`BondForward` produces the clean
and dirty forward prices. The clean forward price reproduces FINCAD's 88.2408
under the simplest assumptions (NullCalendar, 0 settlement days).

Divergences from the Java original (both documented inline):

* PQuantLib has no ``RelinkableHandle``; the bond's discount curve is rebuilt
  with the solved yield and the engine re-set on the bond (the idiomatic Python
  equivalent of relinking a handle).
* ``BondForward`` in pquantlib core does not expose ``implied_yield(...)``; that
  print line from the Java sample is therefore omitted. The market repo rate
  (read back off the flat repo curve) is still printed.
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.bond_forward import BondForward, BondForwardPosition
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule


@dataclass(frozen=True, slots=True)
class RepoResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    bond_clean_price: float
    bond_dirty_price: float
    accrued_at_settlement: float
    accrued_at_delivery: float
    spot_income: float
    fwd_income: float
    strike: float
    npv: float
    clean_forward_price: float
    dirty_forward_price: float
    market_repo_rate: float


def compute() -> RepoResult:
    repo_settlement_date = Date.from_ymd(14, Month.February, 2000)
    repo_delivery_date = Date.from_ymd(15, Month.August, 2000)
    repo_rate = 0.05
    repo_day_count = Actual360()
    repo_compounding = Compounding.Simple
    repo_compound_freq = Frequency.Annual

    # Ten-year bond — the exact tenor is irrelevant to the example.
    bond_issue_date = Date.from_ymd(15, Month.September, 1995)
    bond_dated_date = Date.from_ymd(15, Month.September, 1995)
    bond_maturity_date = Date.from_ymd(15, Month.September, 2005)
    bond_coupon = 0.08
    bond_coupon_frequency = Frequency.Semiannual
    bond_calendar = NullCalendar()
    bond_day_count = Thirty360(Convention.BondBasis)
    bond_settlement_days = 0
    bond_bdc = BusinessDayConvention.Unadjusted
    bond_clean_price = 89.97693786
    bond_redemption = 100.0
    face_amount = 100.0

    ObservableSettings().evaluation_date = repo_settlement_date

    # Dummy bond curve (relinked below to the bond's implied yield).
    bond_curve = FlatForward(
        repo_settlement_date,
        SimpleQuote(0.01),
        bond_day_count,
        Compounding.Compounded,
        bond_coupon_frequency,
    )

    bond_schedule = Schedule.from_rule(
        bond_dated_date,
        bond_maturity_date,
        Period.from_frequency(bond_coupon_frequency),
        bond_calendar,
        bond_bdc,
        bond_bdc,
        DateGeneration.Backward,
        False,
    )

    bond = FixedRateBond(
        bond_settlement_days,
        face_amount,
        bond_schedule,
        [bond_coupon],
        bond_day_count,
        bond_bdc,
        bond_redemption,
        bond_issue_date,
    )
    bond.set_pricing_engine(DiscountingBondEngine(bond_curve))

    # Solve the bond's implied yield from its clean price, then rebuild the
    # discount curve at that yield and re-set the engine (handle-relink analogue).
    implied_yield = bond.yield_from_price(
        BondPrice(bond_clean_price, BondPriceType.Clean),
        bond_day_count,
        Compounding.Compounded,
        bond_coupon_frequency,
    )
    bond_curve = FlatForward(
        repo_settlement_date,
        SimpleQuote(implied_yield),
        bond_day_count,
        Compounding.Compounded,
        bond_coupon_frequency,
    )
    bond.set_pricing_engine(DiscountingBondEngine(bond_curve))

    dummy_strike = 91.5745

    repo_curve = FlatForward(
        repo_settlement_date,
        SimpleQuote(repo_rate),
        repo_day_count,
        repo_compounding,
        repo_compound_freq,
    )

    bond_fwd = BondForward(
        repo_settlement_date,
        repo_delivery_date,
        BondForwardPosition.Long,
        dummy_strike,
        bond_settlement_days,
        repo_day_count,
        bond_calendar,
        bond_bdc,
        bond,
        repo_curve,
        repo_curve,
    )

    spot_income = bond_fwd.spot_income(repo_curve)
    fwd_income = spot_income / repo_curve.discount(repo_delivery_date)
    market_repo_rate = repo_curve.zero_rate(repo_delivery_date, repo_compounding, repo_compound_freq).rate()

    return RepoResult(
        bond_clean_price=bond.clean_price(),
        bond_dirty_price=bond.dirty_price(),
        accrued_at_settlement=bond.accrued_amount(repo_settlement_date),
        accrued_at_delivery=bond.accrued_amount(repo_delivery_date),
        spot_income=spot_income,
        fwd_income=fwd_income,
        strike=dummy_strike,
        npv=bond_fwd.npv(),
        clean_forward_price=bond_fwd.clean_forward_price(),
        dirty_forward_price=bond_fwd.forward_price(),
        market_repo_rate=market_repo_rate,
    )


def run() -> None:
    print("::::: Repo :::::")
    r = compute()

    print(f"Underlying bond clean price: {r.bond_clean_price}")
    print(f"Underlying bond dirty price: {r.bond_dirty_price}")
    print(f"Underlying bond accrued at settlement: {r.accrued_at_settlement}")
    print(f"Underlying bond accrued at delivery:   {r.accrued_at_delivery}")
    print(f"Underlying bond spot income: {r.spot_income}")
    print(f"Underlying bond fwd income:  {r.fwd_income}")
    print(f"Repo strike: {r.strike}")
    print(f"Repo NPV:    {r.npv}")
    print(f"Repo clean forward price: {r.clean_forward_price}")
    print(f"Repo dirty forward price: {r.dirty_forward_price}")
    print(f"Market repo rate:   {r.market_repo_rate}")
    print()
    print("Compare with example given at")
    print("http://www.fincad.com/support/developerFunc/mathref/BFWD.htm")
    print("Clean forward price = 88.2408")
    print()
    print("In that example, it is unknown what bond calendar they are")
    print("using, as well as settlement Days. For that reason, we have")
    print("made the simplest possible assumptions here: NullCalendar")
    print("and 0 settlement days.")


if __name__ == "__main__":
    run()
