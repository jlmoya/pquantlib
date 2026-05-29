"""Gaussian1dNonstandardSwaptionEngine cross-validation vs C++ probe.

5y-into-5y amortizing payer swaption on Euribor6M, fixed rate 3%,
notional decaying linearly from 1M to 0.5M, Gsr(sigma=0.01,
reversion=0.05).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.nonstandard_swap import NonstandardSwap
from pquantlib.instruments.nonstandard_swaption import NonstandardSwaption
from pquantlib.instruments.swap import SwapType
from pquantlib.models.shortrate.onefactor.gsr import Gsr
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swaption.gaussian1d_nonstandard_swaption_engine import (
    Gaussian1dNonstandardSwaptionEngine,
)
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
    return load_reference("cluster/w1b")


@pytest.fixture(autouse=True)
def _eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = Date.from_ymd(15, Month.May, 2026)


def _build_swaption(expiry: Date) -> tuple[NonstandardSwaption, FlatForward]:
    eval_date = Date.from_ymd(15, Month.May, 2026)
    curve = FlatForward.from_rate(
        eval_date, 0.03, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    cal = TARGET()
    e6m = Euribor(Period(6, TimeUnit.Months), curve)
    start = cal.advance(eval_date, 5, TimeUnit.Years)
    end = cal.advance(start, 5, TimeUnit.Years)
    fixed_sched = Schedule.from_rule(
        start, end, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    float_sched = Schedule.from_rule(
        start, end, Period(6, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    n_fix = len(fixed_sched) - 1
    fixed_nominal = [1_000_000.0 * (1.0 - 0.1 * float(i)) for i in range(n_fix)]
    fixed_rate = [0.03] * n_fix
    n_float = len(float_sched) - 1
    float_nominal: list[float] = []
    for i in range(n_float):
        fix_idx = min(i // 2, n_fix - 1)
        float_nominal.append(fixed_nominal[fix_idx])

    swap = NonstandardSwap(
        type_=SwapType.Payer,
        fixed_nominal=fixed_nominal,
        floating_nominal=float_nominal,
        fixed_schedule=fixed_sched,
        fixed_rate=fixed_rate,
        fixed_day_count=Actual360(),
        floating_schedule=float_sched,
        ibor_index=e6m,
        gearing=1.0,
        spread=0.0,
        floating_day_count=Actual360(),
    )
    ex = EuropeanExercise(expiry)
    return NonstandardSwaption(swap, ex), curve


def test_gaussian1d_nonstandard_swaption_matches_cpp_probe(
    cluster_refs: dict[str, Any],
) -> None:
    """Amortizing 5y-into-5y payer swaption — matches C++ probe."""
    expected: dict[str, Any] = cluster_refs["gaussian1d_nonstandard_swaption"]
    expiry = Date(int(expected["exercise_serial"]))
    swaption, curve = _build_swaption(expiry)
    gsr = Gsr(
        curve,
        volstepdates=[],
        volatilities=[float(expected["gsr_sigma"])],
        reversion=float(expected["gsr_reversion"]),
        T=60.0,
    )
    engine = Gaussian1dNonstandardSwaptionEngine(
        gsr,
        integration_points=int(expected["integration_points"]),
        stddevs=float(expected["stddevs"]),
    )
    swaption.set_pricing_engine(engine)
    npv = swaption.npv()
    # Empirical delta ~0.7 / 23312 = 3e-5 (essentially LOOSE-tier).
    # Source: per-exercise-date natural-BC vs Lagrange-BC cubic-spline
    # boundary divergence. Body of the integration is bit-identical.
    custom(
        npv,
        float(expected["ns_npv"]),
        abs_tol=5.0,
        rel_tol=1e-4,
        reason=(
            "Backward-induction state-grid engine; natural-BC vs Lagrange-BC "
            "cubic-spline boundary divergence per exercise date. Single "
            "European exercise → minimal compounding. Empirical delta "
            "~3e-5 relative on a 23k NPV (notional ~1M average)."
        ),
    )


def test_gaussian1d_nonstandard_swaption_zero_vol() -> None:
    """At Gsr(sigma->0), the engine collapses to deterministic NPV.

    Sanity: NPV is non-negative and finite.
    """
    expiry = Date.from_ymd(15, Month.May, 2031)
    swaption, curve = _build_swaption(expiry)
    gsr = Gsr(
        curve,
        volstepdates=[],
        volatilities=[1e-8],
        reversion=0.05,
        T=60.0,
    )
    engine = Gaussian1dNonstandardSwaptionEngine(
        gsr,
        integration_points=16,
        stddevs=3.0,
    )
    swaption.set_pricing_engine(engine)
    npv = swaption.npv()
    assert npv >= -1e-6, f"NPV must be non-negative for an option, got {npv}"
    assert 0.0 <= npv < 100_000.0, f"NPV out of bounds: {npv}"
