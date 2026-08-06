"""Tests for OvernightIndexedSwap cross-validated vs the cluster_l3c probe.

Reference values: ``migration-harness/references/cluster/l3c.json`` → ``ois_2y``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.instruments.overnight_indexed_swap import OvernightIndexedSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = Path(__file__).resolve().parents[3] / "migration-harness/references/cluster/l3c.json"

_EVAL_DATE = Date.from_ymd(17, Month.January, 2024)


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the global evaluation date to the one the C++ probe used.

    ``Swap.is_expired()`` reads ``ObservableSettings().evaluation_date`` through
    ``CashFlow.has_occurred()`` (ql/instruments/swap.cpp:68-75), so without this
    pin the module was silently wall-clock dependent: after 2026-01 every flow
    of the 2024-2026 swap has occurred, the instrument reports expired and every
    result collapses to 0. The probe sets the same date —
    ``migration-harness/cpp/probes/cluster_l3c/probe.cpp`` builds the curve at
    2024-01-17, which ``_build_2y_ois`` mirrors.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _EVAL_DATE
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _build_2y_ois(fixed_rate: float) -> tuple[OvernightIndexedSwap, YieldTermStructureProtocol]:
    """Build the 2y OIS from the probe: Sofr vs 4% fixed, FlatForward(4%)."""
    eval_date = _EVAL_DATE
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.04, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    idx = Sofr(curve)
    cal = idx.fixing_calendar()
    settle = cal.advance(eval_date, 2, TimeUnit.Days)
    end = settle + Period(2, TimeUnit.Years)
    sched = Schedule.from_rule(
        settle, end, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    swap = OvernightIndexedSwap(
        SwapType.Payer, 1_000_000.0, sched, fixed_rate, Actual360(), idx,
    )
    swap.set_pricing_engine(DiscountingSwapEngine(curve))
    return swap, curve


def test_ois_leg_sizes(cluster_refs: dict[str, dict[str, float]]) -> None:
    swap, _ = _build_2y_ois(0.04)
    expected = cluster_refs["ois_2y"]
    assert len(swap.fixed_leg()) == int(expected["fixed_leg_size"])
    assert len(swap.overnight_leg()) == int(expected["overnight_leg_size"])


def test_ois_npv(cluster_refs: dict[str, dict[str, float]]) -> None:
    swap, _ = _build_2y_ois(0.04)
    expected = cluster_refs["ois_2y"]
    tolerance.loose(swap.npv(), expected["npv"])


def test_ois_fair_rate(cluster_refs: dict[str, dict[str, float]]) -> None:
    swap, _ = _build_2y_ois(0.04)
    expected = cluster_refs["ois_2y"]
    tolerance.loose(swap.fair_rate(), expected["fair_rate"])


def test_ois_fixed_leg_npv(cluster_refs: dict[str, dict[str, float]]) -> None:
    swap, _ = _build_2y_ois(0.04)
    expected = cluster_refs["ois_2y"]
    tolerance.loose(swap.fixed_leg_npv(), expected["fixed_leg_npv"])


def test_ois_overnight_leg_npv(cluster_refs: dict[str, dict[str, float]]) -> None:
    swap, _ = _build_2y_ois(0.04)
    expected = cluster_refs["ois_2y"]
    tolerance.loose(swap.overnight_leg_npv(), expected["overnight_leg_npv"])


def test_ois_inspectors() -> None:
    swap, _ = _build_2y_ois(0.04)
    assert swap.swap_type() == SwapType.Payer
    assert swap.fixed_rate() == 0.04
    assert swap.spread() == 0.0
    assert swap.payer(0)
    assert not swap.payer(1)
    # Helper aliases mirror the C++ overnight*-named accessors.
    assert swap.overnight_nominals() == swap.floating_nominals()
    assert swap.overnight_leg() is swap.floating_leg()
