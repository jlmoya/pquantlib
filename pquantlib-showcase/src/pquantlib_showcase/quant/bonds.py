"""Fixed-rate bond pricing: clean/dirty price, yield, accrued, cashflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

from .common import flat_curve, pin_evaluation_date, reference_date

_DC = Thirty360(Convention.BondBasis)


@dataclass(frozen=True, slots=True)
class CashFlow:
    date: str
    amount: float


@dataclass(frozen=True, slots=True)
class BondResult:
    clean_price: float
    dirty_price: float
    accrued: float
    ytm: float
    npv: float
    cashflows: list[CashFlow]


def _build_bond(coupon: float, years: float, freq_months: int, face: float) -> tuple[FixedRateBond, Date]:
    ref = reference_date()
    pin_evaluation_date(ref)
    maturity = ref + max(1, round(years * 365.0))
    schedule = Schedule.from_rule(
        ref,
        maturity,
        Period(freq_months, TimeUnit.Months),
        TARGET(),
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Backward,
        False,
    )
    bond = FixedRateBond(2, face, schedule, [coupon], _DC, BusinessDayConvention.Following, face, ref)
    return bond, ref


def price_bond(
    coupon: float, years: float, curve_rate: float, freq_months: int = 6, face: float = 100.0
) -> BondResult:
    """Price a fixed-rate bond off a flat discount curve."""
    bond, ref = _build_bond(coupon, years, freq_months, face)
    bond.set_pricing_engine(DiscountingBondEngine(flat_curve(curve_rate, ref)))
    freq = Frequency.Semiannual if freq_months == 6 else Frequency.Annual
    ytm = bond.yield_rate(_DC, Compounding.Compounded, freq)
    flows = [CashFlow(str(cf.date()), cf.amount()) for cf in bond.cashflows()]
    return BondResult(
        clean_price=bond.clean_price(),
        dirty_price=bond.dirty_price(),
        accrued=bond.accrued_amount(),
        ytm=ytm,
        npv=bond.npv(),
        cashflows=flows,
    )


def price_vs_curve(
    coupon: float, years: float, freq_months: int, face: float, rate_grid: list[float]
) -> tuple[list[float], list[float]]:
    """Clean price as the flat discount rate sweeps ``rate_grid`` (the DV01 picture)."""
    bond, ref = _build_bond(coupon, years, freq_months, face)
    prices: list[float] = []
    for rate in rate_grid:
        bond.set_pricing_engine(DiscountingBondEngine(flat_curve(rate, ref)))
        prices.append(bond.clean_price())
    return list(rate_grid), prices


def default_rate_grid(low: float = 0.0, high: float = 0.12, n: int = 60) -> list[float]:
    return list(np.linspace(low, high, n))
