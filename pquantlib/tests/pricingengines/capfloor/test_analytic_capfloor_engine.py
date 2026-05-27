"""AnalyticCapFloorEngine cross-validation vs C++ probe.

Uses the same hand-coded Hull-White stub as
``test_jamshidian_swaption_engine`` (shared verbatim — the real
HullWhite from L4-B will satisfy both Protocols).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.cap_floor import Cap
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.capfloor.analytic_capfloor_engine import (
    AffineModelLike,
    AnalyticCapFloorEngine,
)
from pquantlib.pricingengines.swaption.jamshidian_swaption_engine import (
    TermStructureConsistentModelLike,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = (
    Path(__file__).resolve().parents[4] / "migration-harness/references/cluster/l4e.json"
)

_QL_EPSILON = 1.0e-12


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


class _MinimalHullWhite4Arg:
    """Hand-coded HW with the 4-arg discount-bond-option used by
    AnalyticCapFloorEngine.

    C++ parity: hullwhite.cpp:89-105 (4-arg variant).
    """

    def __init__(
        self, ts: YieldTermStructureProtocol, a: float, sigma: float
    ) -> None:
        self._term_structure: YieldTermStructureProtocol = ts
        self._a: float = a
        self._sigma: float = sigma

    @property
    def term_structure(self) -> YieldTermStructureProtocol:
        return self._term_structure

    def _instantaneous_forward(self, t: float) -> float:
        ts = self._term_structure
        rate_fn = getattr(ts, "forward_rate", None)
        if rate_fn is not None:
            try:
                ir = rate_fn(t, t, Compounding.Continuous, Frequency.NoFrequency)
                return ir.rate()
            except Exception:
                pass
        eps = 1e-6
        d1 = ts.discount(max(t - eps, 0.0))
        d2 = ts.discount(t + eps)
        return -(math.log(d2) - math.log(d1)) / (2.0 * eps)

    def _B(self, t: float, T: float) -> float:  # noqa: N802, N803  # C++ math convention (capital T for terminal date)
        a = self._a
        if a < math.sqrt(_QL_EPSILON):
            return T - t
        return (1.0 - math.exp(-a * (T - t))) / a

    def discount(self, t: float) -> float:
        return self._term_structure.discount(t)

    def discount_bond_option(
        self,
        option_type: int,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        # C++ parity: hullwhite.cpp:89-105.
        a = self._a
        if a < math.sqrt(_QL_EPSILON):
            v = self._sigma * self._B(maturity, bond_maturity) * math.sqrt(maturity)
        else:
            v = (
                self._sigma
                * self._B(maturity, bond_maturity)
                * math.sqrt(0.5 * (1.0 - math.exp(-2.0 * a * maturity)) / a)
            )
        ts = self._term_structure
        f = ts.discount(bond_maturity)
        k = ts.discount(maturity) * strike
        opt = OptionType.Call if option_type == 1 else OptionType.Put
        return black_formula(opt, k, f, v, 1.0, 0.0)


def test_minimal_hw_satisfies_affine_protocol() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    hw = _MinimalHullWhite4Arg(curve, 0.1, 0.01)
    assert isinstance(hw, AffineModelLike)
    assert isinstance(hw, TermStructureConsistentModelLike)


def _five_year_cap_setup() -> tuple[list[CashFlow], YieldTermStructureProtocol]:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    cal = TARGET()
    start = cal.advance(eval_date, 2, TimeUnit.Days)
    end = cal.advance(start, 5, TimeUnit.Years)
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    schedule = Schedule.from_rule(
        start, end, Period(3, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    leg = ibor_leg(
        schedule, idx, [1_000_000.0],
        payment_adjustment=BusinessDayConvention.ModifiedFollowing,
        fixing_days=2,
    )
    set_coupon_pricer(leg, IborCouponPricer())
    return leg, curve


def test_analytic_cap_engine_under_hw_matches_probe(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    expected = cluster_refs["analytic_cap_5y_4pct_hw"]
    leg, curve = _five_year_cap_setup()
    cap = Cap(leg, [0.04])
    hw = _MinimalHullWhite4Arg(curve, expected["hw_a"], expected["hw_sigma"])
    cap.set_pricing_engine(AnalyticCapFloorEngine(hw))
    tolerance.loose(cap.npv(), expected["npv"])
