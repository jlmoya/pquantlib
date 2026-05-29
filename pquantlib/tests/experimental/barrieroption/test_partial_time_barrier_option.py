"""Tests for PartialTimeBarrierOption + AnalyticPartialTimeBarrierOptionEngine.

# C++ parity:
# ql/instruments/partialtimebarrieroption.{hpp,cpp} +
# ql/pricingengines/barrier/analyticpartialtimebarrieroptionengine.{hpp,cpp}
# @ v1.42.1.

Cross-validates against the ``partial_time_*`` keys of
``migration-harness/references/cluster/w4c.json``.

Test setup:
* T=1y (365 days), T1=0.5y (182 days, cover event), Actual/365Fixed.
* Spot S=100, sigma=25%, r=5%, q=0%.
* For DownOut/DownIn/EndB2 reliance on strike<barrier: K=90, B=100.
* For UpOut/UpIn: K=110, B=120.
* For Put put-call-symmetry: K=110, B=120, mapped to call via
  X_call = S^2/X_put = 90.909..., B_call = S^2/B_put = 83.333... with
  type flipped.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.barrieroption.partial_time_barrier_option import (
    PartialBarrierRange,
    PartialTimeBarrierOption,
)
from pquantlib.instruments.barrier_option import BarrierType
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.analytic_partial_time_barrier_option_engine import (
    AnalyticPartialTimeBarrierOptionEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import custom, loose
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# C++ engine uses BivariateCumulativeNormalDistributionDr78 (Drezner-1978,
# ~6 decimal places). Our port uses scipy's We04/Genz-Bretz which is
# at least as accurate but produces tiny (~1e-5) absolute differences
# in the multi-term sum. NPV-tier differences propagate to ~5e-5
# absolute on order-10 prices. Custom tolerance applied per-test.
_BVN_ABS: float = 5e-5
_BVN_REL: float = 5e-6
_BVN_REASON: str = (
    "C++ uses BivariateCumulativeNormalDistributionDr78 (Drezner-1978, "
    "~6dp) but our port uses scipy We04/Genz-Bretz; the nested M(a, b, "
    "rho) chain in the Heynen-Kat closed form propagates ~1e-5 absolute "
    "differences to the NPV"
)


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w4c")


@pytest.fixture
def today() -> Date:
    return Date.from_ymd(15, Month.January, 2024)


def _make_process(
    today: Date,
    spot: float,
    r: float,
    q: float,
    sigma: float,
) -> GeneralizedBlackScholesProcess:
    """Build the same BSM process as the C++ probe.

    # C++ parity: ``makeBsmProcess`` in the W4-C probe — FlatForward(r),
    # FlatForward(q), BlackConstantVol(sigma), Actual/365Fixed.
    """
    dc = Actual365Fixed()
    cal = NullCalendar()
    spot_q = SimpleQuote(spot)
    q_curve = FlatForward.from_rate(today, q, dc)
    r_curve = FlatForward.from_rate(today, r, dc)
    vol_ts = BlackConstantVol(
        reference_date=today,
        calendar=cal,
        volatility=sigma,
        day_counter=dc,
    )
    return GeneralizedBlackScholesProcess(
        x0=spot_q,
        dividend_ts=q_curve,
        risk_free_ts=r_curve,
        black_vol_ts=vol_ts,
    )


def _build(
    today: Date,
    barrier_type: BarrierType,
    barrier_range: PartialBarrierRange,
    barrier: float,
    strike: float,
    option_type: OptionType,
    *,
    spot: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.25,
    cover_days: int = 182,
    expiry_days: int = 365,
) -> PartialTimeBarrierOption:
    """Build a partial-time barrier option + attach the analytic engine."""
    process = _make_process(today, spot, r, q, sigma)
    cover_date = today + cover_days
    expiry_date = today + expiry_days
    exercise = EuropeanExercise(expiry_date)
    payoff = PlainVanillaPayoff(option_type, strike)
    opt = PartialTimeBarrierOption(
        barrier_type,
        barrier_range,
        barrier,
        0.0,  # rebate
        cover_date,
        payoff,
        exercise,
    )
    opt.set_pricing_engine(
        AnalyticPartialTimeBarrierOptionEngine(process)
    )
    return opt


# ---------------------------------------------------------------------------
# DownOut Call across three ranges (K=90 < B=100 so EndB2 is admissible).
# Tolerance: LOOSE because the bivariate-normal CDF uses scipy's We04
# vs C++ Drezner-1978 — agreement is well below 1e-8 in practice
# but the analytical formula's chain of nested ``M(a, b, rho)`` calls
# can amplify tiny differences.
# ---------------------------------------------------------------------------
def test_partial_time_downout_call_start(
    today: Date, reference_data: dict[str, Any]
) -> None:
    """DownOut, Start range: S=100 == B=100 puts the option on the boundary;
    expected NPV is ~0 (knocked out).
    """
    opt = _build(
        today,
        BarrierType.DownOut,
        PartialBarrierRange.Start,
        barrier=100.0,
        strike=90.0,
        option_type=OptionType.Call,
    )
    loose(opt.npv(), reference_data["partial_time_downout_call_start"])


def test_partial_time_downout_call_endb1(
    today: Date, reference_data: dict[str, Any]
) -> None:
    opt = _build(
        today,
        BarrierType.DownOut,
        PartialBarrierRange.EndB1,
        barrier=100.0,
        strike=90.0,
        option_type=OptionType.Call,
    )
    custom(
        opt.npv(),
        reference_data["partial_time_downout_call_endb1"],
        abs_tol=_BVN_ABS,
        rel_tol=_BVN_REL,
        reason=_BVN_REASON,
    )


def test_partial_time_downout_call_endb2(
    today: Date, reference_data: dict[str, Any]
) -> None:
    opt = _build(
        today,
        BarrierType.DownOut,
        PartialBarrierRange.EndB2,
        barrier=100.0,
        strike=90.0,
        option_type=OptionType.Call,
    )
    custom(
        opt.npv(),
        reference_data["partial_time_downout_call_endb2"],
        abs_tol=_BVN_ABS,
        rel_tol=_BVN_REL,
        reason=_BVN_REASON,
    )


def test_partial_time_upout_call_start(
    today: Date, reference_data: dict[str, Any]
) -> None:
    opt = _build(
        today,
        BarrierType.UpOut,
        PartialBarrierRange.Start,
        barrier=120.0,
        strike=110.0,
        option_type=OptionType.Call,
    )
    custom(
        opt.npv(),
        reference_data["partial_time_upout_call_start"],
        abs_tol=_BVN_ABS,
        rel_tol=_BVN_REL,
        reason=_BVN_REASON,
    )


def test_partial_time_downin_call_start(
    today: Date, reference_data: dict[str, Any]
) -> None:
    """DownIn Start (S=100, B=80, K=90): NPV via CIA = vanilla - CA."""
    opt = _build(
        today,
        BarrierType.DownIn,
        PartialBarrierRange.Start,
        barrier=80.0,
        strike=90.0,
        option_type=OptionType.Call,
    )
    custom(
        opt.npv(),
        reference_data["partial_time_downin_call_start"],
        abs_tol=_BVN_ABS,
        rel_tol=_BVN_REL,
        reason=_BVN_REASON,
    )


def test_partial_time_upin_call_start(
    today: Date, reference_data: dict[str, Any]
) -> None:
    opt = _build(
        today,
        BarrierType.UpIn,
        PartialBarrierRange.Start,
        barrier=120.0,
        strike=110.0,
        option_type=OptionType.Call,
    )
    custom(
        opt.npv(),
        reference_data["partial_time_upin_call_start"],
        abs_tol=_BVN_ABS,
        rel_tol=_BVN_REL,
        reason=_BVN_REASON,
    )


def test_partial_time_upout_put_start(
    today: Date, reference_data: dict[str, Any]
) -> None:
    """Put variant — engine maps to call via FX-style put-call symmetry."""
    opt = _build(
        today,
        BarrierType.UpOut,
        PartialBarrierRange.Start,
        barrier=120.0,
        strike=110.0,
        option_type=OptionType.Put,
    )
    custom(
        opt.npv(),
        reference_data["partial_time_upout_put_start"],
        abs_tol=_BVN_ABS,
        rel_tol=_BVN_REL,
        reason=_BVN_REASON,
    )


# ---------------------------------------------------------------------------
# Unimplemented cases mirror the C++ QL_FAIL behaviour.
# ---------------------------------------------------------------------------
def test_downin_endb1_unimplemented(today: Date) -> None:
    opt = _build(
        today,
        BarrierType.DownIn,
        PartialBarrierRange.EndB1,
        barrier=80.0,
        strike=90.0,
        option_type=OptionType.Call,
    )
    with pytest.raises(LibraryException, match="not implemented"):
        opt.npv()


def test_upin_endb2_unimplemented(today: Date) -> None:
    opt = _build(
        today,
        BarrierType.UpIn,
        PartialBarrierRange.EndB2,
        barrier=120.0,
        strike=110.0,
        option_type=OptionType.Call,
    )
    with pytest.raises(LibraryException, match="not implemented"):
        opt.npv()


def test_downout_endb2_strike_ge_barrier_unimplemented(today: Date) -> None:
    """C++ engine QL_FAILs when strike >= barrier in EndB2 mode."""
    opt = _build(
        today,
        BarrierType.DownOut,
        PartialBarrierRange.EndB2,
        barrier=100.0,
        strike=110.0,  # strike > barrier
        option_type=OptionType.Call,
    )
    with pytest.raises(LibraryException, match="OutEnd B2"):
        opt.npv()
