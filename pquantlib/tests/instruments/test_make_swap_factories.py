"""Tests for make_vanilla_swap and make_ois free-function factories.

Cross-validate that the factories produce swaps numerically identical
(within LOOSE) to the direct-constructor swaps tested in
test_vanilla_swap.py / test_overnight_indexed_swap.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.instruments.make_ois import make_ois
from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = Path(__file__).resolve().parents[3] / "migration-harness/references/cluster/l3c.json"


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


def test_make_vanilla_swap_default_fixed_rate_solves_fair(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    """When fixed_rate is None, make_vanilla_swap solves for fair_rate.

    Fair rate from FF(5%) Continuous/Annual for 5y vs Euribor3M (US-style
    inputs except EUR currency-driven fixed-leg conventions) is ~5.14%
    per the cluster_l3c probe.
    """
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    swap = make_vanilla_swap(
        Period(5, TimeUnit.Years),
        idx,
        fixed_rate=None,
        nominal=1_000_000.0,
        floating_leg_tenor=Period(3, TimeUnit.Months),
        fixed_leg_tenor=Period(6, TimeUnit.Months),
        fixed_leg_day_count=Thirty360(Convention.BondBasis),
        evaluation_date=eval_date,
    )
    expected_fair = cluster_refs["vanilla_swap_5y"]["fair_rate"]
    tolerance.loose(swap.fixed_rate(), expected_fair)


def test_make_vanilla_swap_explicit_rate_npv(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    """When fixed_rate is given, the swap matches direct-ctor NPV."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    swap = make_vanilla_swap(
        Period(5, TimeUnit.Years),
        idx,
        fixed_rate=0.05,
        nominal=1_000_000.0,
        floating_leg_tenor=Period(3, TimeUnit.Months),
        fixed_leg_tenor=Period(6, TimeUnit.Months),
        fixed_leg_day_count=Thirty360(Convention.BondBasis),
        evaluation_date=eval_date,
    )
    expected_npv = cluster_refs["vanilla_swap_5y"]["npv"]
    tolerance.loose(swap.npv(), expected_npv)


def test_make_ois_default_fixed_rate_solves_fair(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.04, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    sofr = Sofr(curve)
    ois = make_ois(
        Period(2, TimeUnit.Years),
        sofr,
        fixed_rate=None,
        nominal=1_000_000.0,
        evaluation_date=eval_date,
        end_of_month=False,
        payment_frequency=Frequency.Annual,
    )
    expected_fair = cluster_refs["ois_2y"]["fair_rate"]
    tolerance.loose(ois.fixed_rate(), expected_fair)


def test_make_ois_explicit_rate_npv(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.04, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    sofr = Sofr(curve)
    ois = make_ois(
        Period(2, TimeUnit.Years),
        sofr,
        fixed_rate=0.04,
        nominal=1_000_000.0,
        evaluation_date=eval_date,
        end_of_month=False,
        payment_frequency=Frequency.Annual,
    )
    expected_npv = cluster_refs["ois_2y"]["npv"]
    tolerance.loose(ois.npv(), expected_npv)


def test_make_vanilla_swap_rejects_double_dates() -> None:
    """Cannot pass both effective_date and settlement_days."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    with pytest.raises(Exception, match="effective date and settlement days"):
        make_vanilla_swap(
            Period(5, TimeUnit.Years),
            idx,
            fixed_rate=0.05,
            effective_date=Date.from_ymd(19, Month.January, 2024),
            settlement_days=2,
            evaluation_date=eval_date,
        )


def test_make_vanilla_swap_with_explicit_effective_date() -> None:
    """Explicit effective_date bypasses the spot-date inference."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    eff = Date.from_ymd(19, Month.January, 2024)
    swap = make_vanilla_swap(
        Period(5, TimeUnit.Years),
        idx,
        fixed_rate=0.05,
        effective_date=eff,
        fixed_leg_tenor=Period(6, TimeUnit.Months),
        fixed_leg_day_count=Thirty360(Convention.BondBasis),
        fixed_leg_convention=BusinessDayConvention.ModifiedFollowing,
        fixed_leg_termination_convention=BusinessDayConvention.ModifiedFollowing,
    )
    assert swap.start_date() == eff
