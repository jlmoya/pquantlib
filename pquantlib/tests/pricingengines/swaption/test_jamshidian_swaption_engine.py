"""JamshidianSwaptionEngine cross-validation vs C++ probe.

Uses a self-contained minimal Hull-White model (just enough surface to
satisfy ``OneFactorAffineModelLike`` + ``TermStructureConsistentModelLike``).
The real ``HullWhite`` class lands in L4-B; once it merges, this test
will continue to pass against the real model since it satisfies the
same structural Protocol.

C++ parity: HullWhite analytic formulas at v1.42.1 (hullwhite.cpp:75-131,
vasicek.cpp:36-55, onefactormodel.hpp:132-138).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exercise import EuropeanExercise
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import Swaption
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.pricingengines.swaption.jamshidian_swaption_engine import (
    JamshidianSwaptionEngine,
    OneFactorAffineModelLike,
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


# --- Minimal Hull-White model for testing the engine independently of L4-B ---


class _MinimalHullWhite:
    """Hand-coded HW closed-form formulas — just enough surface to drive
    JamshidianSwaptionEngine.

    C++ parity: HullWhite (hullwhite.cpp:75-131) + Vasicek::B (vasicek.cpp:49-55)
    + OneFactorAffineModel::discountBond (onefactormodel.hpp:136-138).

    Satisfies both ``OneFactorAffineModelLike`` and
    ``TermStructureConsistentModelLike``.
    """

    def __init__(
        self,
        term_structure_: YieldTermStructureProtocol,
        a: float,
        sigma: float,
    ) -> None:
        self._term_structure: YieldTermStructureProtocol = term_structure_
        self._a: float = a
        self._sigma: float = sigma
        # r0 = forward(0,0) per HullWhite::HullWhite (hullwhite.cpp:33).
        self._r0: float = self._instantaneous_forward(0.0)

    @property
    def term_structure(self) -> YieldTermStructureProtocol:
        return self._term_structure

    def _instantaneous_forward(self, t: float) -> float:
        # Forward rate via FlatForward — for a flat curve at rate r,
        # f(0,t) = r.
        ts = self._term_structure
        # Pull a small forward via the curve's forwardRate
        # interface — but FlatForward might not expose it directly; for
        # the test fixture (5% Continuous Annual Actual360) the
        # instantaneous forward IS just 0.05.
        rate_fn = getattr(ts, "forward_rate", None)
        if rate_fn is not None:
            try:
                # Try (t1, t2, compounding, freq) signature.
                ir = rate_fn(t, t, Compounding.Continuous, Frequency.NoFrequency)
                # Returns InterestRate; pull .rate().
                return ir.rate()
            except Exception:
                pass
        # Fallback: numerical f(0,t) = -d ln(D(t))/dt around t.
        eps = 1e-6
        d1 = ts.discount(max(t - eps, 0.0))
        d2 = ts.discount(t + eps)
        return -(math.log(d2) - math.log(d1)) / (2.0 * eps)

    # --- Vasicek B + HW A -------------------------------------------------
    # Method + arg names match the C++ math convention (vasicek.cpp:49,
    # hullwhite.cpp:75); we suppress the lower-case-only lint here.

    def _B(self, t: float, T: float) -> float:  # noqa: N802, N803
        a = self._a
        if a < math.sqrt(_QL_EPSILON):
            return T - t
        return (1.0 - math.exp(-a * (T - t))) / a

    def _A(self, t: float, T: float) -> float:  # noqa: N802, N803
        ts = self._term_structure
        d1 = ts.discount(t)
        d2 = ts.discount(T)
        forward = self._instantaneous_forward(t)
        temp = self._sigma * self._B(t, T)
        value = self._B(t, T) * forward - 0.25 * temp * temp * self._B(0.0, 2.0 * t)
        return math.exp(value) * d2 / d1

    # --- OneFactorAffineModelLike ----------------------------------------

    def discount(self, t: float) -> float:
        return self._term_structure.discount(t)

    def discount_bond(self, now: float, maturity: float, x: float) -> float:
        return self._A(now, maturity) * math.exp(-self._B(now, maturity) * x)

    def discount_bond_option(
        self,
        option_type: int,
        strike: float,
        maturity: float,
        value_time: float,
        bond_maturity: float,
    ) -> float:
        # C++ parity: hullwhite.cpp:107-131 (5-arg variant).
        a = self._a
        if a < math.sqrt(_QL_EPSILON):
            v = self._sigma * self._B(value_time, bond_maturity) * math.sqrt(maturity)
        else:
            # C++ formula in hullwhite.cpp:116-126.
            c = (
                math.exp(-2.0 * a * (value_time - maturity))
                - math.exp(-2.0 * a * value_time)
                - 2.0
                * (
                    math.exp(-a * (value_time + bond_maturity - 2.0 * maturity))
                    - math.exp(-a * (value_time + bond_maturity))
                )
                + math.exp(-2.0 * a * (bond_maturity - maturity))
                - math.exp(-2.0 * a * bond_maturity)
            )
            v = (self._sigma / (a * math.sqrt(2.0 * a))) * math.sqrt(max(c, 0.0))
        ts = self._term_structure
        f = ts.discount(bond_maturity)
        k = ts.discount(value_time) * strike
        # Black formula with our OptionType enum.
        opt = OptionType.Call if option_type == 1 else OptionType.Put
        return black_formula(opt, k, f, v, 1.0, 0.0)


# Confirm the stub satisfies the engine's structural typing.
def test_minimal_hw_satisfies_protocols() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    hw = _MinimalHullWhite(curve, 0.1, 0.01)
    assert isinstance(hw, OneFactorAffineModelLike)
    assert isinstance(hw, TermStructureConsistentModelLike)


# --- Build the same 5y10y swaption as the C++ probe ---


def _five_by_ten_receiver_swaption(
    curve: YieldTermStructureProtocol,
) -> Swaption:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    cal = TARGET()
    settle = cal.advance(eval_date, 5, TimeUnit.Years)
    end = cal.advance(settle, 10, TimeUnit.Years)
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    fixed_sched = Schedule.from_rule(
        settle, end, Period(6, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    float_sched = Schedule.from_rule(
        settle, end, Period(3, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    swap = VanillaSwap(
        SwapType.Receiver, 1_000_000.0,
        fixed_sched, 0.03, Thirty360(Convention.BondBasis),
        float_sched, idx, 0.0, idx.day_counter(),
    )
    swap.set_pricing_engine(DiscountingSwapEngine(curve))
    exercise = EuropeanExercise(settle)
    return Swaption(swap, exercise)


def test_jamshidian_swaption_npv_matches_probe(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    expected = cluster_refs["jamshidian_swaption_5y10y"]
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    swaption = _five_by_ten_receiver_swaption(curve)
    hw = _MinimalHullWhite(curve, expected["hw_a"], expected["hw_sigma"])
    swaption.set_pricing_engine(JamshidianSwaptionEngine(hw))
    # LOOSE: Jamshidian solver + HW closed-form + per-coupon DBO sum
    # accumulates float work.
    tolerance.loose(swaption.npv(), expected["npv"])
