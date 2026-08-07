"""CounterpartyAdjSwapEngine cross-validation against C++ QuantLib v1.43.

Reference: ``migration-harness/references/v143/pe/bondswap.json`` (probe
``migration-harness/cpp/probes/v143_pe_bondswap/probe.cpp``, "PART 6").

Pinned: both trade directions, the ``Volatility`` and the ``Handle<Quote>``
constructors (which must give identical numbers for the same vol), the
counterparty-only case against the case with both default curves, and the
degenerate zero-hazard / full-recovery cases where the adjustment must vanish.

The ``cva_swaplet_strip`` case pins the engine's *inner* strip — the swaplets it
builds with ``MakeVanillaSwap`` and prices as European swaptions — so that a
mismatch can be localised to the swaplet construction rather than to the CVA
arithmetic. Its first swaplet expires on the pricing date, i.e. the swaption is
already expired and prices at 0 without the engine running at all; that is what
first exposed the ``Swaption.is_expired`` divergence.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exercise import EuropeanExercise
from pquantlib.indexes.ibor.euribor import Euribor6M
from pquantlib.instruments.make_vanilla_swap import MakeVanillaSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import Swaption
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.cva_swap_engine import CounterpartyAdjSwapEngine
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.pricingengines.swaption.black_swaption_engine import BlackSwaptionEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
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

CPP: dict[str, Any] = reference_reader.load("v143/pe/bondswap")

# probe.cpp — `const Date kToday(15, May, 2025);`
TODAY = Date.from_ymd(15, Month.May, 2025)
DC_365 = Actual365Fixed()
DC_360 = Actual360()
DC_30_360 = Thirty360(Convention.BondBasis)

# LOOSE, not TIGHT: the adjusted NPV is a difference of same-magnitude terms —
# a riskless NPV of ~76 minus 0.6 * a swaption strip whose accumulated value is
# ~550, so the result loses about one decimal digit of the ~1e-16 double
# resolution per term. Measured agreement with C++ is <= 2.5e-15 relative.
_CVA_REASON = (
    "CVA NPV is riskless-NPV (~76) minus 0.6 * a ~550 swaption strip; the "
    "cancellation makes the last two bits of the strip visible in the result"
)


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _curve() -> FlatForward:
    return FlatForward.from_rate(TODAY, 0.03, DC_365, Compounding.Compounded, Frequency.Annual)


def _schedule(tenor: Period) -> Schedule:
    return Schedule.from_rule(
        effective_date=Date.from_ymd(19, Month.May, 2025),
        termination_date=Date.from_ymd(19, Month.May, 2030),
        tenor=tenor,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        termination_date_convention=BusinessDayConvention.ModifiedFollowing,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )


def _swap(curve: FlatForward, swap_type: SwapType) -> VanillaSwap:
    return VanillaSwap(
        swap_type,
        1_000_000.0,
        _schedule(Period(1, TimeUnit.Years)),
        0.03,
        DC_30_360,
        _schedule(Period(6, TimeUnit.Months)),
        Euribor6M(curve),
        0.0,
        DC_360,
    )


_CASES = sorted(
    k
    for k in CPP
    if k.startswith("cva_") and k not in ("cva_riskless_reference", "cva_swaplet_strip")
)


@pytest.mark.parametrize("case", _CASES)
def test_cva_adjusted_npv(case: str) -> None:
    inputs = CPP[case]["inputs"]
    expected = CPP[case]["expected"]
    curve = _curve()
    swap_type = SwapType.Payer if inputs["swap_type"] == "Payer" else SwapType.Receiver
    swap = _swap(curve, swap_type)

    ctpty = FlatHazardRate.from_rate(TODAY, float(inputs["ctpty_hazard_rate"]), DC_365)
    invst = (
        FlatHazardRate.from_rate(TODAY, float(inputs["invst_hazard_rate"]), DC_365)
        if inputs["with_investor_curve"]
        else None
    )
    vol: SimpleQuote | float = (
        SimpleQuote(float(inputs["black_vol"]))
        if inputs["use_quote_ctor"]
        else float(inputs["black_vol"])
    )
    swap.set_pricing_engine(
        CounterpartyAdjSwapEngine(
            curve,
            vol,
            ctpty,
            float(inputs["ctpty_recovery_rate"]),
            invst,
            float(inputs["invst_recovery_rate"]),
        )
    )

    tolerance.custom(
        swap.npv(), expected["npv"], abs_tol=1e-9, rel_tol=1e-13, reason=_CVA_REASON
    )
    tolerance.tight(swap.fair_rate(), expected["fair_rate"])


def test_volatility_and_quote_constructors_agree() -> None:
    """C++ ctors #2 and #3 differ only in how the same vol is delivered."""
    tolerance.exact(
        CPP["cva_payer_quote_ctor"]["expected"]["npv"],
        CPP["cva_payer_vol_ctor"]["expected"]["npv"],
    )
    curve = _curve()
    ctpty = FlatHazardRate.from_rate(TODAY, 0.02, DC_365)
    a = _swap(curve, SwapType.Payer)
    a.set_pricing_engine(CounterpartyAdjSwapEngine(curve, 0.20, ctpty, 0.4))
    b = _swap(curve, SwapType.Payer)
    b.set_pricing_engine(CounterpartyAdjSwapEngine(curve, SimpleQuote(0.20), ctpty, 0.4))
    tolerance.exact(a.npv(), b.npv())


def test_investor_curve_changes_the_answer() -> None:
    """Counterparty-only vs both curves must not give the same number."""
    only = CPP["cva_payer_vol_ctor"]["expected"]["npv"]
    both = CPP["cva_both_default_curves"]["expected"]["npv"]
    assert only != both
    # The DVA term is a credit, so supplying an investor curve raises the NPV.
    assert both > only


def test_empty_investor_curve_is_a_1e_minus_12_hazard_not_zero() -> None:
    """C++ substitutes ``FlatHazardRate(0, NullCalendar(), 1e-12, dc)``.

    The DVA term is therefore tiny but not identically zero: pricing with an
    explicit 1e-12 investor curve on the counterparty's day counter must
    reproduce the default-constructed answer bit-for-bit.
    """
    curve = _curve()
    ctpty = FlatHazardRate.from_rate(TODAY, 0.02, DC_365)
    implicit = _swap(curve, SwapType.Payer)
    implicit.set_pricing_engine(CounterpartyAdjSwapEngine(curve, 0.20, ctpty, 0.4))
    explicit = _swap(curve, SwapType.Payer)
    explicit.set_pricing_engine(
        CounterpartyAdjSwapEngine(
            curve, 0.20, ctpty, 0.4, FlatHazardRate.from_rate(TODAY, 1.0e-12, DC_365), 0.999
        )
    )
    tolerance.tight(implicit.npv(), explicit.npv())


def test_zero_hazard_and_full_recovery_remove_the_adjustment() -> None:
    riskless = CPP["cva_riskless_reference"]["expected"]["npv"]
    for case in ("cva_zero_ctpty_hazard", "cva_full_recovery"):
        # Not exactly the riskless NPV: the DVA term survives through the
        # 1e-12 stand-in curve, and C++ itself differs in the last bits.
        tolerance.tight(CPP[case]["expected"]["npv"], riskless)


def test_fair_rate_is_the_documented_approximation() -> None:
    """``results.fairRate`` is cvaswapengine.cpp:211-214, not a re-solve."""
    riskless_rate = CPP["cva_riskless_reference"]["expected"]["fair_rate"]
    adjusted = CPP["cva_payer_vol_ctor"]["expected"]["fair_rate"]
    assert adjusted != riskless_rate
    fixed_npv = CPP["cva_riskless_reference"]["expected"]["fixed_leg_npv"]
    float_npv = CPP["cva_riskless_reference"]["expected"]["floating_leg_npv"]
    # Re-derive it from the pinned strip: fairRate = -r * (floatNPV - CVA + DVA) / fixedNPV,
    # where (-CVA + DVA) is exactly (adjustedNPV - risklessNPV).
    adjustment = (
        CPP["cva_payer_vol_ctor"]["expected"]["npv"] - CPP["cva_riskless_reference"]["expected"]["npv"]
    )
    tolerance.custom(
        -0.03 * (float_npv + adjustment) / fixed_npv,
        adjusted,
        abs_tol=1e-12,
        rel_tol=1e-12,
        reason="re-derivation from pinned leg NPVs re-associates the same sum",
    )


def test_swaplet_strip_matches() -> None:
    """The engine's inner swaplet strip, rebuilt exactly as C++ builds it."""
    expected = CPP["cva_swaplet_strip"]["expected"]
    curve = _curve()
    index = Euribor6M(curve)
    swap = _swap(curve, SwapType.Payer)
    swap.set_pricing_engine(DiscountingSwapEngine(curve))
    base_swap_fair_rate = -0.03 * swap.floating_leg_npv() / swap.fixed_leg_npv()
    tolerance.tight(base_swap_fair_rate, expected["base_swap_fair_rate"])
    tolerance.tight(swap.fair_rate(), expected["base_swap_fair_rate_via_accessor"])

    black_engine = BlackSwaptionEngine(curve, 0.20)
    fixed_pay_dates = [cf.date() for cf in swap.fixed_leg()]
    last_fixed = fixed_pay_dates[-1]
    swaplet_start = TODAY
    for k, next_fd in enumerate(fixed_pay_dates):
        tenor = Period(last_fixed.serial_number() - swaplet_start.serial_number(), TimeUnit.Days)
        assert int(expected[f"swaplet_{k}_tenor_days"]) == (
            last_fixed.serial_number() - swaplet_start.serial_number()
        )
        assert swaplet_start == Date(int(expected[f"swaplet_{k}_start"]))
        assert last_fixed == Date(int(expected[f"swaplet_{k}_end"]))

        call_swap = (
            MakeVanillaSwap(tenor, index, base_swap_fair_rate)
            .with_type(SwapType.Payer)
            .with_nominal(1_000_000.0)
            .with_effective_date(swaplet_start)
            .with_termination_date(last_fixed)
            .build()
        )
        put_swap = (
            MakeVanillaSwap(tenor, index, base_swap_fair_rate)
            .with_type(SwapType.Receiver)
            .with_nominal(1_000_000.0)
            .with_effective_date(swaplet_start)
            .with_termination_date(last_fixed)
            .build()
        )
        assert len(call_swap.floating_leg()) == int(expected[f"swaplet_{k}_n_float_coupons"])

        call_opt = Swaption(call_swap, EuropeanExercise(swaplet_start))
        put_opt = Swaption(put_swap, EuropeanExercise(swaplet_start))
        call_opt.set_pricing_engine(black_engine)
        put_opt.set_pricing_engine(black_engine)
        tolerance.tight(call_opt.npv(), expected[f"swaplet_{k}_call_npv"])
        tolerance.tight(put_opt.npv(), expected[f"swaplet_{k}_put_npv"])

        ctpty = FlatHazardRate.from_rate(TODAY, 0.02, DC_365)
        tolerance.tight(
            ctpty.default_probability(swaplet_start, next_fd),
            expected[f"swaplet_{k}_ctpty_default_prob"],
        )
        swaplet_start = next_fd


def test_first_swaplet_is_expired_and_prices_at_zero() -> None:
    """A swaption exercising on the pricing date is expired: NPV 0, no engine.

    C++ ``Swaption::isExpired`` is
    ``simple_event(exercise->dates().back()).hasOccurred()`` and
    ``Event::hasOccurred`` defaults to ``date() <= referenceDate``. The probe
    records ``swaplet_0_call_npv == swaplet_0_put_npv == 0`` *and*
    ``swaplet_0_atm_forward_throws == true`` — i.e. C++ returned 0 without ever
    entering ``BlackStyleSwaptionEngine``.
    """
    expected = CPP["cva_swaplet_strip"]["expected"]
    tolerance.exact(expected["swaplet_0_call_npv"], 0.0)
    tolerance.exact(expected["swaplet_0_put_npv"], 0.0)
    assert expected["swaplet_0_atm_forward_throws"] is True
    assert expected["swaplet_0_annuity_throws"] is True

    curve = _curve()
    index = Euribor6M(curve)
    swaplet = (
        MakeVanillaSwap(Period(1831, TimeUnit.Days), index, 0.030016678637798003)
        .with_type(SwapType.Payer)
        .with_nominal(1_000_000.0)
        .with_effective_date(TODAY)
        .with_termination_date(Date(int(expected["swaplet_0_end"])))
        .build()
    )
    option = Swaption(swaplet, EuropeanExercise(TODAY))
    assert option.is_expired() is True
    option.set_pricing_engine(BlackSwaptionEngine(curve, 0.20))
    tolerance.exact(option.npv(), 0.0)
