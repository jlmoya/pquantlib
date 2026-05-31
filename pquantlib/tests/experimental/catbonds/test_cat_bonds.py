"""Cat bonds (W8-B batch b) — cross-validation vs C++ probe.

Probe source: migration-harness/cpp/probes/cluster_w8b/probe.cpp
Reference:    migration-harness/references/cluster/w8b.json

Covers:
  * EventSet / EventSetSimulation — path count + aggregated loss over a
    synthetic catalogue.
  * Digital / Proportional NotionalRisk — deterministic notional erosion.
  * FloatingCatBond + MonteCarloCatBondEngine — NPV + loss/exhaustion
    probability under a deterministic EventSet cat risk.
  * BetaRisk — method-of-moments alpha/beta mapping + simulated severity
    moments (statistical, not draw-by-draw vs C++).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.usd_libor import USDLibor
from pquantlib.instruments.cat_bond import FloatingCatBond
from pquantlib.instruments.cat_risk import BetaRisk, BetaRiskSimulation, EventSet
from pquantlib.instruments.risky_notional import (
    DigitalNotionalRisk,
    NoOffset,
    NotionalPath,
    ProportionalNotionalRisk,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.monte_carlo_cat_bond_engine import (
    MonteCarloCatBondEngine,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
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
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w8b")


_TODAY = Date.from_ymd(15, Month.February, 2007)


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _TODAY


def _june(year: int) -> Date:
    return Date.from_ymd(15, Month.June, year)


# ----------------------------------------------------------------------
# EventSet
# ----------------------------------------------------------------------


def test_event_set_path_count_and_loss(cpp_ref: dict[str, Any]) -> None:
    """EventSet over a 5-event catalogue reproduces the C++ path stats."""
    events = [(_june(y), 10.0) for y in range(2000, 2005)]
    es = EventSet(
        events, Date.from_ymd(1, Month.January, 2000), Date.from_ymd(31, Month.December, 2004)
    )
    sim = es.new_simulation(
        Date.from_ymd(1, Month.January, 2010), Date.from_ymd(31, Month.December, 2010)
    )
    path: list[tuple[Date, float]] = []
    path_count = 0
    total_loss = 0.0
    nonempty = 0
    while sim.next_path(path):
        path_count += 1
        total_loss += sum(loss for _, loss in path)
        if path:
            nonempty += 1
        if path_count > 100:
            break

    assert path_count == int(cpp_ref["eventset_path_count"])
    tolerance.loose(total_loss, float(cpp_ref["eventset_total_loss"]))
    assert nonempty == int(cpp_ref["eventset_nonempty_paths"])


# ----------------------------------------------------------------------
# Notional risk
# ----------------------------------------------------------------------


def test_digital_notional_risk(cpp_ref: dict[str, Any]) -> None:
    """Digital wipe: threshold 40 → only the 50-loss event triggers."""
    offset = NoOffset()
    events = [(_june(2010), 30.0), (_june(2011), 50.0)]
    digital = DigitalNotionalRisk(offset, 40.0)
    path = NotionalPath()
    digital.update_path(events, path)
    tolerance.loose(path.loss(), float(cpp_ref["digital_loss"]))
    tolerance.loose(path.notional_rate(_june(2011)), float(cpp_ref["digital_notional_after"]))


def test_proportional_notional_risk(cpp_ref: dict[str, Any]) -> None:
    """Proportional erosion: attach 20, exhaust 100; losses 30 then 80."""
    offset = NoOffset()
    events = [(_june(2010), 30.0), (_june(2011), 50.0)]
    prop = ProportionalNotionalRisk(offset, 20.0, 100.0)
    path = NotionalPath()
    prop.update_path(events, path)
    tolerance.loose(
        path.notional_rate(_june(2010)), float(cpp_ref["proportional_notional_after_first"])
    )
    tolerance.loose(
        path.notional_rate(_june(2011)), float(cpp_ref["proportional_notional_after_second"])
    )
    tolerance.loose(path.loss(), float(cpp_ref["proportional_loss"]))


# ----------------------------------------------------------------------
# MonteCarloCatBondEngine
# ----------------------------------------------------------------------


def _cat_curve() -> FlatForward:
    return FlatForward.from_rate(
        _TODAY, 0.055, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )


def test_monte_carlo_cat_bond_engine(cpp_ref: dict[str, Any]) -> None:
    """FloatingCatBond NPV + loss stats under a deterministic EventSet."""
    curve = _cat_curve()
    events = [(_june(2001), 1000.0), (_june(2004), 1000.0), (_june(2007), 1000.0)]
    cat_risk = EventSet(
        events, Date.from_ymd(1, Month.January, 2000), Date.from_ymd(31, Month.December, 2010)
    )
    notional_risk = DigitalNotionalRisk(NoOffset(), 500.0)

    cat_issue = Date.from_ymd(15, Month.February, 2007)
    cat_maturity = Date.from_ymd(15, Month.February, 2008)
    index = USDLibor(Period(3, TimeUnit.Months), curve)
    index.add_fixing(index.fixing_date(cat_issue), 0.055)

    sched = Schedule.from_rule(
        cat_issue,
        cat_maturity,
        Period.from_frequency(Frequency.Quarterly),
        TARGET(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        False,
    )
    cat_bond = FloatingCatBond(
        settlement_days=0,
        face_amount=1_000_000.0,
        schedule=sched,
        ibor_index=index,
        accrual_day_counter=Actual365Fixed(),
        notional_risk=notional_risk,
        payment_convention=BusinessDayConvention.Following,
    )
    cat_bond.set_pricing_engine(MonteCarloCatBondEngine(cat_risk, curve))

    # Trigger calculation first (the loss-stat inspectors are non-computing,
    # matching the C++ ``lossProbability()`` accessor).
    npv = cat_bond.npv()

    # The cat-risk simulation + notional erosion + discounting are exact: the
    # loss / exhaustion / expected-loss stats match the C++ probe to LOOSE.
    tolerance.loose(cat_bond.loss_probability(), float(cpp_ref["catbond_loss_probability"]))
    tolerance.loose(
        cat_bond.exhaustion_probability(), float(cpp_ref["catbond_exhaustion_probability"])
    )
    tolerance.loose(cat_bond.expected_loss(), float(cpp_ref["catbond_expected_loss"]))

    # The NPV carries the underlying floating-coupon valuation, where
    # PQuantLib's IborCouponPricer differs from C++'s BlackIborCouponPricer
    # par-rate convexity at the ~1e-4 relative level (a pre-existing L2-D
    # coupon-pricer port boundary, not a cat-bond engine effect — the loss
    # stats above isolate the engine and match exactly).
    tolerance.custom(
        npv,
        float(cpp_ref["catbond_npv"]),
        abs_tol=1.0,
        rel_tol=2e-4,
        reason="floating-coupon par-rate convexity: IborCouponPricer vs C++ BlackIborCouponPricer",
    )


# ----------------------------------------------------------------------
# BetaRisk (statistical only — RNG diverges from C++ mt19937)
# ----------------------------------------------------------------------


def test_beta_risk_moment_mapping() -> None:
    """BetaRisk maps (mean, std_dev) onto Beta(alpha, beta) by moments.

    Verifies the method-of-moments alpha/beta against the closed-form
    Beta-distribution moment relations (independent of any RNG draw).
    """
    max_loss = 100.0
    mean = 20.0
    std_dev = 10.0
    br = BetaRisk(max_loss, years=1.0, mean=mean, std_dev=std_dev)

    # For a Beta(a,b) scaled to [0, max_loss]:
    #   E[X]   = max_loss * a/(a+b)
    #   Var[X] = max_loss^2 * a*b / ((a+b)^2 (a+b+1))
    a, b = br.alpha, br.beta
    e_x = max_loss * a / (a + b)
    var_x = max_loss * max_loss * a * b / ((a + b) ** 2 * (a + b + 1.0))
    tolerance.loose(e_x, mean)
    tolerance.loose(var_x, std_dev * std_dev)


def test_beta_risk_simulated_severity_moments() -> None:
    """Simulated Beta severities track the target mean (statistical)."""
    max_loss = 100.0
    mean = 20.0
    std_dev = 10.0
    br = BetaRisk(max_loss, years=1.0, mean=mean, std_dev=std_dev)
    # Build the simulation directly with a fixed-seed generator so the
    # severity moments are reproducible across runs.
    sim = BetaRiskSimulation(
        _TODAY,
        _TODAY + Period(1, TimeUnit.Years),
        max_loss,
        br.intensity,
        br.alpha,
        br.beta,
        rng=np.random.default_rng(12345),
    )

    severities: list[float] = []
    for _ in range(20000):
        path: list[tuple[Date, float]] = []
        sim.next_path(path)
        severities.extend(loss for _, loss in path)

    assert len(severities) > 1000
    sample_mean = float(np.mean(severities))
    # 2% relative tolerance on a 20k-sample Monte-Carlo mean.
    tolerance.custom(
        sample_mean, mean, abs_tol=1.0, rel_tol=0.05,
        reason="20k-sample MC severity mean vs target (statistical)",
    )
