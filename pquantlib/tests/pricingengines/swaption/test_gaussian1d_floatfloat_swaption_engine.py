"""Gaussian1dFloatFloatSwaptionEngine cross-validation vs C++ probe.

Probe (cluster/w1b.json) describes a 5y-into-5y synthetic float-float
swap (Euribor6M payer vs Euribor3M + (-50bp) receiver, 1M notional)
priced under Gsr(sigma=0.01, reversion=0.05, T=60y). LOOSE — the
engine's per-event payoff replication has different cubic boundary
treatment between PQuantLib (natural BC) and C++ (Lagrange BC); the
divergence is amplified by the backward-induction event count.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.float_float_swap import FloatFloatSwap
from pquantlib.instruments.float_float_swaption import FloatFloatSwaption
from pquantlib.instruments.swap import SwapType
from pquantlib.models.shortrate.onefactor.gsr import Gsr
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swaption.gaussian1d_floatfloat_swaption_engine import (
    Gaussian1dFloatFloatSwaptionEngine,
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


def _build_swaption(
    spread2: float,
    expiry: Date,
) -> tuple[FloatFloatSwaption, FlatForward]:
    eval_date = Date.from_ymd(15, Month.May, 2026)
    curve = FlatForward.from_rate(
        eval_date, 0.03, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    cal = TARGET()
    e3m = Euribor(Period(3, TimeUnit.Months), curve)
    e6m = Euribor(Period(6, TimeUnit.Months), curve)
    start = cal.advance(eval_date, 5, TimeUnit.Years)
    end = cal.advance(start, 5, TimeUnit.Years)
    sched1 = Schedule.from_rule(
        start, end, Period(6, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    sched2 = Schedule.from_rule(
        start, end, Period(3, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    swap = FloatFloatSwap(
        type_=SwapType.Payer,
        nominal1=1_000_000.0,
        nominal2=1_000_000.0,
        schedule1=sched1,
        index1=e6m,
        day_count1=Actual360(),
        schedule2=sched2,
        index2=e3m,
        day_count2=Actual360(),
        gearing1=1.0,
        spread1=0.0,
        gearing2=1.0,
        spread2=spread2,
    )
    ex = EuropeanExercise(expiry)
    return FloatFloatSwaption(swap, ex), curve


def test_gaussian1d_floatfloat_swaption_matches_cpp_probe(
    cluster_refs: dict[str, Any],
) -> None:
    """5y-into-5y FloatFloat swaption — Gsr(0.01, 0.05), 32 grid pts, 5 stddevs.

    The engine's NPV must match the C++ probe within ~1% of the option
    value (the engine's payoff is a thin spread; absolute NPV is small
    so even small per-event divergences compound).
    """
    expected: dict[str, Any] = cluster_refs["gaussian1d_floatfloat_swaption"]
    expiry_serial = int(expected["exercise_serial"])
    expiry = Date(expiry_serial)
    swaption, curve = _build_swaption(float(expected["spread2"]), expiry)
    gsr = Gsr(
        curve,
        volstepdates=[],
        volatilities=[float(expected["gsr_sigma"])],
        reversion=float(expected["gsr_reversion"]),
        T=60.0,
    )
    engine = Gaussian1dFloatFloatSwaptionEngine(
        gsr,
        integration_points=int(expected["integration_points"]),
        stddevs=float(expected["stddevs"]),
    )
    swaption.set_pricing_engine(engine)
    npv = swaption.npv()
    # Empirical delta ~0.024 / 10.04 = 0.24%. Source: per-event
    # natural-BC vs Lagrange-BC cubic interpolation divergence in the
    # backward-induction roll-back, compounded over ~30 event dates.
    custom(
        npv,
        float(expected["ff_npv"]),
        abs_tol=1.0,
        rel_tol=1e-2,
        reason=(
            "Backward-induction state-grid engine; per-event cubic-spline "
            "boundary divergence (natural BC vs Lagrange BC) compounds "
            "over multiple event dates (~30 events for a 5y-into-5y "
            "swap with Q+S coupon frequencies). Empirical delta 0.24% "
            "on an option NPV ~10 / notional 1M (~1bp of notional)."
        ),
    )


def test_gaussian1d_floatfloat_swaption_zero_vol() -> None:
    """At Gsr(sigma->0), the engine should reduce to deterministic NPV.

    Sanity: the option NPV at sigma=1e-8 collapses to ``max(forward
    underlying NPV, 0)`` — deterministic floor at zero on the payer.
    Specific tolerance is loose because the C++ probe doesn't separately
    capture this; we just verify the value is non-negative and finite.
    """
    expiry = Date.from_ymd(15, Month.May, 2031)
    swaption, curve = _build_swaption(-0.0050, expiry)
    gsr = Gsr(
        curve,
        volstepdates=[],
        volatilities=[1e-8],
        reversion=0.05,
        T=60.0,
    )
    engine = Gaussian1dFloatFloatSwaptionEngine(
        gsr,
        integration_points=16,
        stddevs=3.0,
    )
    swaption.set_pricing_engine(engine)
    npv = swaption.npv()
    # Sanity bounds: must be non-negative (American option), finite,
    # and < 5% of notional (5y forward swap with 50bp spread has
    # bounded value).
    assert npv >= -1e-6, f"NPV must be non-negative for an option, got {npv}"
    assert npv < 50_000.0, f"NPV unbounded relative to notional? {npv}"
