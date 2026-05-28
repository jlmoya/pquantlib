"""TreeCapFloorEngine cross-validation vs Analytic + C++ probe.

5y cap @ 4% on Euribor3M (starting 3M forward to avoid past-fixing)
under HullWhite(a=0.1, sigma=0.01) with 100 timesteps; compare
against the AnalyticCapFloorEngine value (LOOSE — tree convergence).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.cap_floor import Cap
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.pricingengines.capfloor.tree_capfloor_engine import TreeCapFloorEngine
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import custom
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


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, Any]:
    return load_reference("cluster/l5b")


def _build_cap() -> tuple[Cap, FlatForward]:
    """Build the same 5y cap@4% as the L5-B C++ probe — start 3M forward
    to avoid the past-fixing setup.
    """
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = FlatForward.from_rate(
        eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
    )
    cal = TARGET()
    cap_start = cal.advance(eval_date, 3, TimeUnit.Months)
    cap_end = cal.advance(cap_start, 5, TimeUnit.Years)
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    sched = Schedule.from_rule(
        cap_start, cap_end, Period(3, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    leg = ibor_leg(
        sched,
        idx,
        nominals=[1_000_000.0],
        payment_day_counter=idx.day_counter(),
        payment_adjustment=BusinessDayConvention.ModifiedFollowing,
    )
    return Cap(leg, [0.04]), curve


def test_tree_capfloor_engine_matches_cpp_probe(
    cluster_refs: dict[str, Any],
) -> None:
    """TreeCapFloorEngine NPV under HW @ 100 steps reproduces the C++
    probe value to within ~10bp of notional (state-price accumulation
    + date-handling). The Python and C++ algorithms agree on the
    closed-form check vs Analytic (within tree convergence ~1e-3).
    """
    expected: dict[str, Any] = cluster_refs["tree_cap_hw"]
    cap, curve = _build_cap()
    hw = HullWhite(curve, a=float(expected["hw_a"]), sigma=float(expected["hw_sigma"]))
    cap.set_pricing_engine(
        TreeCapFloorEngine(hw, int(expected["time_steps"]), term_structure=curve)
    )
    npv = cap.npv()
    # The C++ tree NPV is ~53097; Python and C++ should agree within
    # ~10bp of notional (1e6) — leg construction may differ slightly.
    custom(
        npv,
        float(expected["npv"]),
        abs_tol=100.0,  # 10 bp of 1e6 notional
        rel_tol=2e-3,
        reason="HW tree cap engine — leg-build + state-price accumulation",
    )


def test_tree_capfloor_engine_close_to_analytic_ref(
    cluster_refs: dict[str, Any],
) -> None:
    """The tree engine NPV is within tree-convergence tolerance of the
    C++ analytic reference (Analytic = 53030; Tree @ N=100 = 53097 —
    relative error ~1.3e-3).
    """
    analytic_ref = float(cluster_refs["analytic_cap_ref"]["npv"])
    cap, curve = _build_cap()
    hw = HullWhite(curve, a=0.1, sigma=0.01)
    cap.set_pricing_engine(TreeCapFloorEngine(hw, 100, term_structure=curve))
    tree_npv = cap.npv()
    # Tree @ 100 steps converges to analytic to ~3%.
    assert abs(tree_npv - analytic_ref) / abs(analytic_ref) < 0.03
