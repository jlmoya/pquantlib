"""Yield-curve construction: flat curves and bootstrapped deposit curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

from .common import flat_curve, pin_evaluation_date, reference_date


@dataclass(frozen=True, slots=True)
class CurveProfile:
    """A curve sampled across a grid of maturities."""

    tenors: list[float]
    discount_factors: list[float]
    zero_rates: list[float]
    forward_rates: list[float]


def _sample(curve, tenors: list[float]) -> CurveProfile:
    dfs, zeros, fwds = [], [], []
    for i, t in enumerate(tenors):
        dfs.append(curve.discount(t))
        zeros.append(curve.zero_rate(t, Compounding.Continuous, Frequency.Annual).rate())
        t0 = tenors[i - 1] if i > 0 else 0.0
        t1 = max(t, t0 + 1e-6)
        fwds.append(curve.forward_rate(t0, t1, Compounding.Continuous, Frequency.Annual).rate())
    return CurveProfile(tenors, dfs, zeros, fwds)


def flat_curve_profile(rate: float, max_years: float = 10.0, n: int = 60) -> CurveProfile:
    """Sample a flat curve across ``[0, max_years]``."""
    ref = reference_date()
    curve = flat_curve(rate, ref)
    tenors = list(np.linspace(0.25, max_years, n))
    return _sample(curve, tenors)


# Standard money-market deposit pillars (label, period, default quote).
DEPOSIT_PILLARS: list[tuple[str, int, TimeUnit]] = [
    ("1M", 1, TimeUnit.Months),
    ("3M", 3, TimeUnit.Months),
    ("6M", 6, TimeUnit.Months),
    ("9M", 9, TimeUnit.Months),
    ("1Y", 12, TimeUnit.Months),
]


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A bootstrapped curve plus the pillars it reprices."""

    pillar_labels: list[str]
    pillar_years: list[float]
    pillar_quotes: list[float]
    profile: CurveProfile


def bootstrap_deposit_curve(quotes: list[float], max_years: float = 1.0, n: int = 60) -> BootstrapResult:
    """Bootstrap a discount curve from money-market deposit quotes.

    ``quotes`` aligns with :data:`DEPOSIT_PILLARS`.
    """
    ref = reference_date()
    pin_evaluation_date(ref)
    helpers = [
        DepositRateHelper(
            SimpleQuote(rate),
            tenor=Period(n_, unit),
            fixing_days=2,
            calendar=TARGET(),
            convention=BusinessDayConvention.ModifiedFollowing,
            end_of_month=True,
            day_counter=Actual360(),
            evaluation_date=ref,
        )
        for (rate, (_, n_, unit)) in zip(quotes, DEPOSIT_PILLARS, strict=True)
    ]
    curve = PiecewiseYieldCurve(Discount, ref, helpers, Actual360())
    tenors = list(np.linspace(1.0 / 12.0, max_years, n))
    pillar_years = [n_ / 12.0 if unit == TimeUnit.Months else float(n_) for (_, n_, unit) in DEPOSIT_PILLARS]
    return BootstrapResult(
        pillar_labels=[lbl for (lbl, _, _) in DEPOSIT_PILLARS],
        pillar_years=pillar_years,
        pillar_quotes=list(quotes),
        profile=_sample(curve, tenors),
    )
