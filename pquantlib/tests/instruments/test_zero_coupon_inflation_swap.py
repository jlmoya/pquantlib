"""ZeroCouponInflationSwap cross-validation against C++ probe (L7-D cluster).

C++ reference values: ``migration-harness/references/cluster/l7d.json``.

Builds the same setup as the probe:

- evaluation date 2024-01-17,
- flat 3% nominal discount curve (Act/365F, Continuous, Annual),
- flat 2.5% zero-inflation curve from base date (2019-10-15) forward,
- UKRPI index with base-period fixing = 100.0 (so the forecast path
  resolves to ``I(T) = 100 * (1 + 0.025)^t``),
- ZCIIS Payer-type, nominal 1M, start 2020-01-15, maturity 2030-01-15,
  fixed_rate=2.5%, observation_lag=3M.

Cross-validates:

- fair_rate ≈ 0.0250 (probe gives 0.024999734...),
- fixed_leg_npv,
- inflation_leg_npv,
- total NPV (≈ 0 since fair_rate ≈ fixed_rate).

LOOSE tier: the chain of zero-curve interpolation + inflation_year_fraction
+ Act/Act ISDA arithmetic accumulates ~1e-6 rounding, especially the
``base_fixing * (1+Z1)^t`` exponentiation. LOOSE captures.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as ActualActualConvention
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import (
    ZeroInflationIndex,
    inflation_period,
)
from pquantlib.indexes.inflation.uk_rpi import UKRPI
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.zero_coupon_inflation_swap import ZeroCouponInflationSwap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import (
    DiscountingSwapEngine,
)
from pquantlib.termstructures.inflation.zero_inflation_term_structure import (
    ZeroInflationTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = (
    Path(__file__).resolve().parents[3] / "migration-harness/references/cluster/l7d.json"
)
_EVAL_DATE = Date.from_ymd(17, Month.January, 2024)


# ---------------------------------------------------------------------------
# Scaffolding: a flat-rate ZeroInflationTermStructure for the test.
# ---------------------------------------------------------------------------


class _FlatZeroInflationCurve(ZeroInflationTermStructure):
    """Minimal flat-rate ZeroInflationTermStructure.

    L7-D scaffold for ZCIIS NPV tests. Always returns the constructor
    rate from ``_zero_rate_impl``. L7-B's bootstrap curves will subsume.
    """

    def __init__(self, *, base_date: Date, rate: float) -> None:
        super().__init__(
            base_date=base_date,
            frequency=Frequency.Monthly,
            day_counter=ActualActual(ActualActualConvention.ISDA),
            observation_lag=Period(3, TimeUnit.Months),
            reference_date=_EVAL_DATE,
            calendar=TARGET(),
        )
        self._rate = rate

    def _zero_rate_impl(self, t: float) -> float:
        del t
        return self._rate

    def max_date(self) -> Date:
        return Date.max_date()


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = _EVAL_DATE
    yield
    s.evaluation_date = old


def _build_swap(
    *,
    nominal: float,
    fixed_rate: float,
    start_date: Date,
    maturity: Date,
    obs_lag: Period,
) -> tuple[ZeroCouponInflationSwap, ZeroInflationIndex]:
    base_date = start_date - obs_lag
    curve = _FlatZeroInflationCurve(base_date=base_date, rate=0.025)
    ukrpi = UKRPI(ts=curve)
    # Store the base-period fixing = 100.
    base_period_start, _ = inflation_period(base_date, ukrpi.frequency())
    ukrpi.add_fixing(base_period_start, 100.0, True)

    swap = ZeroCouponInflationSwap(
        type_=SwapType.Payer,
        nominal=nominal,
        start_date=start_date,
        maturity=maturity,
        fix_calendar=TARGET(),
        fix_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=ActualActual(ActualActualConvention.ISDA),
        fixed_rate=fixed_rate,
        index=ukrpi,
        observation_lag=obs_lag,
        observation_interpolation=InterpolationType.AsIndex,
    )
    return swap, ukrpi


def test_zciis_fair_rate_and_legs_match_cpp(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    ref = cluster_refs["zero_coupon_inflation_swap"]
    # Probe dates: start_serial=43845 → 2020-01-15; maturity_serial=47498 → 2030-01-15.
    swap, _ = _build_swap(
        nominal=ref["nominal"],
        fixed_rate=ref["fixed_rate"],
        start_date=Date.from_ymd(15, Month.January, 2020),
        maturity=Date.from_ymd(15, Month.January, 2030),
        obs_lag=Period(int(ref["observation_lag_months"]), TimeUnit.Months),
    )
    # Nominal discount curve: flat 3% Act/365F, Continuous, Annual.
    nominal_curve = FlatForward.from_rate(
        _EVAL_DATE, 0.03, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    eng = DiscountingSwapEngine(nominal_curve)
    swap.set_pricing_engine(eng)

    # The swap is constructed at the swap's fair rate (2.5% market rate),
    # so the NPV should be near zero (probe gives ~2.77 EUR off zero).
    loose(swap.fair_rate(), ref["fair_rate"], reason="ZCIIS fair rate")
    loose(swap.fixed_leg_npv(), ref["fixed_leg_npv"], reason="ZCIIS fixed leg NPV")
    loose(
        swap.inflation_leg_npv(),
        ref["inflation_leg_npv"],
        reason="ZCIIS inflation leg NPV (forecast via flat zero curve)",
    )
    loose(swap.npv(), ref["npv"], reason="ZCIIS net NPV")


def test_zciis_inspectors() -> None:
    obs_lag = Period(3, TimeUnit.Months)
    swap, ukrpi = _build_swap(
        nominal=1_000_000.0,
        fixed_rate=0.025,
        start_date=Date.from_ymd(15, Month.January, 2020),
        maturity=Date.from_ymd(15, Month.January, 2030),
        obs_lag=obs_lag,
    )
    assert swap.type() == SwapType.Payer
    assert swap.nominal() == 1_000_000.0
    assert swap.fixed_rate() == 0.025
    assert swap.observation_lag() == obs_lag
    assert swap.inflation_index() is ukrpi
    assert swap.observation_interpolation() == InterpolationType.AsIndex
    assert len(swap.fixed_leg()) == 1
    assert len(swap.inflation_leg()) == 1


def test_zciis_constructor_rejects_underflag_lag() -> None:
    """Observation lag must be >= index availability lag."""
    base_date = Date.from_ymd(15, Month.October, 2019)
    curve = _FlatZeroInflationCurve(base_date=base_date, rate=0.025)
    ukrpi = UKRPI(ts=curve)
    # UKRPI availability = 1M; obs_lag = 0 days < 1M.
    with pytest.raises(Exception, match="availability lag"):
        ZeroCouponInflationSwap(
            type_=SwapType.Payer,
            nominal=1.0,
            start_date=Date.from_ymd(15, Month.January, 2020),
            maturity=Date.from_ymd(15, Month.January, 2030),
            fix_calendar=TARGET(),
            fix_convention=BusinessDayConvention.ModifiedFollowing,
            day_counter=ActualActual(ActualActualConvention.ISDA),
            fixed_rate=0.025,
            index=ukrpi,
            observation_lag=Period(0, TimeUnit.Days),
        )
