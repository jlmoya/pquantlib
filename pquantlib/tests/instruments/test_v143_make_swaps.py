"""Cross-validate the v1.43 ``Make*`` swap builders against C++ QuantLib.

Probe: ``v143/inst/makeswaps``.

Each case is the same baseline builder with exactly ONE chained setter moved off
its default, and every case is checked against the COMPLETE cashflow listing of
both legs — payment date and amount per flow, plus nominal, accrual start/end,
accrual period and rate per coupon — as well as the fair rate/spread, both legs'
BPS and NPV, and the swap NPV.

That structure is the point. ``FixedVsFloatingSwap`` and ``OvernightIndexedSwap``
each used to accept a ``payment_lag``, store it and never hand it to a leg
builder; every lagged swap paid on its accrual end dates and no test noticed,
because no test set a lag and then looked at a payment date. A builder has a
dozen such arguments, so here each one is set to a non-default value and checked
against a value that only comes out right if the argument actually arrives.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.currencies.america import CADCurrency, USDCurrency
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.europe import CHFCurrency, GBPCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.indexes.ibor.sonia import Sonia
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.claim import Claim
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwap,
    ProtectionSide,
    cds_maturity,
)
from pquantlib.instruments.fixed_vs_floating_swap import FixedVsFloatingSwap
from pquantlib.instruments.make_cds import MakeCDS, MakeCreditDefaultSwap
from pquantlib.instruments.make_multiple_resets_swap import MakeMultipleResetsSwap
from pquantlib.instruments.make_ois import MakeOIS, make_ois
from pquantlib.instruments.make_vanilla_swap import MakeVanillaSwap, make_vanilla_swap
from pquantlib.instruments.multiple_resets_swap import MultipleResetsSwap
from pquantlib.instruments.overnight_indexed_swap import OvernightIndexedSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.credit.midpoint_cds_engine import MidPointCdsEngine
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

# --- fixture (mirrors the probe's) -----------------------------------------

TODAY: Date = Date.from_ymd(15, Month.June, 2026)
# A TARGET end-of-month business day: Feb 2027 ends on a Sunday, so the calendar
# end of month is Friday the 26th. The end-of-month setters only bite from an
# end-of-month start.
EOM_START: Date = Date.from_ymd(26, Month.February, 2027)

_YEARS = TimeUnit.Years
_MONTHS = TimeUnit.Months


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/makeswaps")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the global evaluation date to the probe's."""
    ObservableSettings().evaluation_date = TODAY
    yield
    ObservableSettings().evaluation_date = None


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(TODAY, rate, Actual365Fixed())


def fwd_curve() -> YieldTermStructureProtocol:
    return cast(YieldTermStructureProtocol, _flat(0.03))


def disc_curve() -> YieldTermStructureProtocol:
    return cast(YieldTermStructureProtocol, _flat(0.025))


def euribor6m() -> Euribor:
    return Euribor.six_months(fwd_curve())


def euribor3m() -> Euribor:
    return Euribor.three_months(fwd_curve())


def sofr() -> Sofr:
    return Sofr(fwd_curve())


def generic_index(ccy: Currency, tenor: Period) -> IborIndex:
    """Generic index in a given currency — pins the currency-driven defaults."""
    return IborIndex(
        "Generic",
        tenor,
        2,
        ccy,
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        fwd_curve(),
    )


class HalfNotionalClaim(Claim):
    """Non-default claim — mirrors the probe's ``HalfNotionalClaim``."""

    def amount(self, default_date: Date, notional: float, recovery_rate: float) -> float:
        del default_date
        return 0.5 * notional * (1.0 - recovery_rate)


# --- comparison helpers -----------------------------------------------------


def _check_flow(flow: CashFlow, ref: dict[str, Any]) -> None:
    assert flow.date().serial_number() == ref["pay"]
    tight(flow.amount(), ref["amount"])
    if "nominal" not in ref:
        return
    assert isinstance(flow, Coupon)
    tight(flow.nominal(), ref["nominal"])
    assert flow.accrual_start_date().serial_number() == ref["start"]
    assert flow.accrual_end_date().serial_number() == ref["end"]
    tight(flow.accrual_period(), ref["accrual"])
    tight(flow.rate(), ref["rate"])


def _check_leg(leg: list[CashFlow], ref: list[dict[str, Any]]) -> None:
    assert len(leg) == len(ref)
    for flow, flow_ref in zip(leg, ref, strict=True):
        _check_flow(flow, flow_ref)


def _check_npv(swap: FixedVsFloatingSwap, ref: dict[str, Any]) -> None:
    """Compare the swap NPV with a cancellation-aware absolute bound.

    The NPV is the sum of two leg NPVs of magnitude ``L`` that very nearly
    cancel — these swaps are built at their own fair rate, so the true value is
    zero and everything reported is rounding. Each leg NPV is a sum of a handful
    of discounted flows, so it carries a few ulps of ``L``; adding the two then
    leaves an absolute error of order ``L * 2**-53`` per operation. A bound of
    ``L * 1e-14`` is roughly 90 ulps of ``L``: far above that noise floor, and
    still three orders of magnitude below the smallest structural error worth
    catching (one basis point on either leg is ``1e-4 * L``). At ``L <= 1`` it
    collapses to the plain TIGHT absolute tolerance.
    """
    scale = max(abs(ref["fixed_leg_npv"]), abs(ref["floating_leg_npv"]), 1.0)
    custom(
        swap.npv(),
        ref["npv"],
        abs_tol=scale * 1e-14,
        rel_tol=1e-12,
        reason="swap NPV is a near-total cancellation of two leg NPVs of magnitude "
        f"{scale:g}; bound is ~90 ulps of that magnitude",
    )


def _check_swap(swap: FixedVsFloatingSwap, ref: dict[str, Any]) -> None:
    assert int(swap.swap_type()) == ref["type"]
    assert swap.start_date().serial_number() == ref["start"]
    assert swap.maturity_date().serial_number() == ref["maturity"]
    assert int(swap.payment_convention()) == ref["payment_convention"]
    tight(swap.fixed_rate(), ref["fixed_rate"])
    tight(swap.spread(), ref["spread"])
    tight(swap.fair_rate(), ref["fair_rate"])
    tight(swap.fair_spread(), ref["fair_spread"])
    tight(swap.fixed_leg_bps(), ref["fixed_leg_bps"])
    tight(swap.fixed_leg_npv(), ref["fixed_leg_npv"])
    tight(swap.floating_leg_bps(), ref["floating_leg_bps"])
    tight(swap.floating_leg_npv(), ref["floating_leg_npv"])
    _check_npv(swap, ref)
    _check_leg(swap.fixed_leg(), ref["fixed_leg"])
    _check_leg(swap.floating_leg(), ref["floating_leg"])


def _check_cds(cds: CreditDefaultSwap, ref: dict[str, Any]) -> None:
    assert int(cds.side()) == ref["side"]
    tight(cds.notional(), ref["notional"])
    tight(cds.running_spread(), ref["running_spread"])
    assert (cds.upfront() is not None) == ref["has_upfront"]
    if cds.upfront() is not None:
        tight(cast(float, cds.upfront()), ref["upfront"])
    assert cds.settles_accrual() == ref["settles_accrual"]
    assert cds.pays_at_default_time() == ref["pays_at_default_time"]
    assert cds.rebates_accrual() == ref["rebates_accrual"]
    assert cds.protection_start_date().serial_number() == ref["protection_start"]
    assert cds.protection_end_date().serial_number() == ref["protection_end"]
    assert cds.trade_date().serial_number() == ref["trade_date"]
    assert cds.cash_settlement_days() == ref["cash_settlement_days"]
    assert cds.upfront_payment().date().serial_number() == ref["upfront_pay"]
    tight(cds.upfront_payment().amount(), ref["upfront_amount"])
    rebate = cds.accrual_rebate()
    if "rebate_pay" in ref:
        assert rebate is not None
        assert rebate.date().serial_number() == ref["rebate_pay"]
        tight(rebate.amount(), ref["rebate_amount"])
    else:
        assert rebate is None
    tight(cds.npv(), ref["npv"])
    tight(cds.fair_spread(), ref["fair_spread"])
    tight(cds.coupon_leg_bps(), ref["coupon_leg_bps"])
    tight(cds.coupon_leg_npv(), ref["coupon_leg_npv"])
    tight(cds.default_leg_npv(), ref["default_leg_npv"])
    tight(cds.upfront_npv(), ref["upfront_npv"])
    tight(cds.accrual_rebate_npv(), ref["accrual_rebate_npv"])
    _check_leg(cds.coupons(), ref["coupons"])


# ===========================================================================
# MakeVanillaSwap
# ===========================================================================


def vs_base() -> MakeVanillaSwap:
    return MakeVanillaSwap(Period(5, _YEARS), euribor6m())


VS_CASES: dict[str, Callable[[], MakeVanillaSwap]] = {
    # baseline + the two constructor arguments beyond the required pair
    "vs_base": vs_base,
    "vs_ctor_fixed_rate": lambda: MakeVanillaSwap(Period(5, _YEARS), euribor6m(), 0.02),
    "vs_ctor_forward_start": lambda: MakeVanillaSwap(
        Period(5, _YEARS), euribor6m(), None, Period(3, _MONTHS)
    ),
    # type / nominal
    "vs_receive_fixed": lambda: vs_base().receive_fixed(),
    "vs_with_type_receiver": lambda: vs_base().with_type(SwapType.Receiver),
    "vs_nominal": lambda: vs_base().with_nominal(1.0e6),
    # start / end date resolution
    "vs_settlement_days": lambda: vs_base().with_settlement_days(5),
    "vs_effective_date": lambda: vs_base().with_effective_date(
        Date.from_ymd(1, Month.July, 2026)
    ),
    "vs_termination_date": lambda: vs_base().with_termination_date(
        Date.from_ymd(17, Month.June, 2031)
    ),
    "vs_termination_date_gbp_short": lambda: MakeVanillaSwap(
        Period(5, _YEARS), generic_index(GBPCurrency(), Period(6, _MONTHS))
    )
    .with_effective_date(Date.from_ymd(17, Month.June, 2026))
    .with_termination_date(Date.from_ymd(17, Month.June, 2027)),
    "vs_termination_date_gbp_long": lambda: MakeVanillaSwap(
        Period(5, _YEARS), generic_index(GBPCurrency(), Period(6, _MONTHS))
    )
    .with_effective_date(Date.from_ymd(17, Month.June, 2026))
    .with_termination_date(Date.from_ymd(17, Month.June, 2029)),
    # rules / conventions
    "vs_rule": lambda: vs_base().with_rule(DateGeneration.Forward),
    "vs_payment_convention": lambda: vs_base().with_payment_convention(
        BusinessDayConvention.Preceding
    ),
    # fixed leg
    "vs_fixed_leg_tenor": lambda: vs_base().with_fixed_leg_tenor(Period(6, _MONTHS)),
    "vs_fixed_leg_calendar": lambda: vs_base().with_fixed_leg_calendar(UnitedKingdom()),
    "vs_fixed_leg_convention": lambda: vs_base().with_fixed_leg_convention(
        BusinessDayConvention.Preceding
    ),
    "vs_fixed_leg_termination_convention": lambda: (
        vs_base().with_fixed_leg_termination_date_convention(BusinessDayConvention.Preceding)
    ),
    "vs_fixed_leg_rule": lambda: vs_base().with_fixed_leg_rule(DateGeneration.Forward),
    "vs_fixed_leg_day_count": lambda: vs_base().with_fixed_leg_day_count(Actual360()),
    "vs_fixed_leg_first_date": lambda: vs_base()
    .with_effective_date(Date.from_ymd(17, Month.June, 2026))
    .with_fixed_leg_first_date(Date.from_ymd(17, Month.December, 2026)),
    "vs_fixed_leg_next_to_last_date": lambda: vs_base()
    .with_effective_date(Date.from_ymd(17, Month.June, 2026))
    .with_fixed_leg_next_to_last_date(Date.from_ymd(17, Month.December, 2030)),
    # floating leg
    "vs_float_leg_tenor": lambda: vs_base().with_floating_leg_tenor(Period(3, _MONTHS)),
    "vs_float_leg_calendar": lambda: vs_base().with_floating_leg_calendar(UnitedKingdom()),
    "vs_float_leg_convention": lambda: vs_base().with_floating_leg_convention(
        BusinessDayConvention.Preceding
    ),
    "vs_float_leg_termination_convention": lambda: (
        vs_base().with_floating_leg_termination_date_convention(
            BusinessDayConvention.Preceding
        )
    ),
    "vs_float_leg_rule": lambda: vs_base().with_floating_leg_rule(DateGeneration.Forward),
    "vs_float_leg_day_count": lambda: vs_base().with_floating_leg_day_count(
        Actual365Fixed()
    ),
    "vs_float_leg_spread": lambda: vs_base().with_floating_leg_spread(0.0025),
    "vs_float_leg_first_date": lambda: vs_base()
    .with_effective_date(Date.from_ymd(17, Month.June, 2026))
    .with_floating_leg_first_date(Date.from_ymd(17, Month.September, 2026)),
    "vs_float_leg_next_to_last_date": lambda: vs_base()
    .with_effective_date(Date.from_ymd(17, Month.June, 2026))
    .with_floating_leg_next_to_last_date(Date.from_ymd(17, Month.March, 2031)),
    # end-of-month family
    "vs_eom_none": lambda: vs_base().with_effective_date(EOM_START),
    "vs_eom_fixed": lambda: vs_base()
    .with_effective_date(EOM_START)
    .with_fixed_leg_end_of_month(True),
    "vs_eom_float": lambda: vs_base()
    .with_effective_date(EOM_START)
    .with_floating_leg_end_of_month(True),
    "vs_eom_maturity": lambda: vs_base()
    .with_effective_date(EOM_START)
    .with_maturity_end_of_month(True),
    # discounting / engine
    "vs_discounting_ts": lambda: vs_base().with_discounting_term_structure(disc_curve()),
    "vs_pricing_engine": lambda: vs_base().with_pricing_engine(
        DiscountingSwapEngine(disc_curve(), include_settlement_date_flows=False)
    ),
    # indexed vs at-par coupons
    "vs_coupons_default": lambda: vs_base().with_floating_leg_tenor(Period(3, _MONTHS)),
    "vs_indexed_coupons": lambda: vs_base()
    .with_floating_leg_tenor(Period(3, _MONTHS))
    .with_indexed_coupons(True),
    "vs_at_par_coupons": lambda: vs_base()
    .with_floating_leg_tenor(Period(3, _MONTHS))
    .with_at_par_coupons(True),
    # currency-driven fixed-leg defaults
    "vs_ccy_usd": lambda: MakeVanillaSwap(
        Period(5, _YEARS), generic_index(USDCurrency(), Period(3, _MONTHS))
    ),
    "vs_ccy_gbp_1y": lambda: MakeVanillaSwap(
        Period(1, _YEARS), generic_index(GBPCurrency(), Period(6, _MONTHS))
    ),
    "vs_ccy_gbp_5y": lambda: MakeVanillaSwap(
        Period(5, _YEARS), generic_index(GBPCurrency(), Period(6, _MONTHS))
    ),
    "vs_ccy_chf": lambda: MakeVanillaSwap(
        Period(5, _YEARS), generic_index(CHFCurrency(), Period(6, _MONTHS))
    ),
}


@pytest.mark.parametrize("key", sorted(VS_CASES))
def test_make_vanilla_swap(key: str, cpp: dict[str, Any]) -> None:
    _check_swap(VS_CASES[key]().build(), cpp[key])


def test_make_vanilla_swap_raises_effective_and_settlement(cpp: dict[str, Any]) -> None:
    assert cpp["vs_raises_effective_and_settlement"]["raises"] is True
    with pytest.raises(LibraryException, match="effective date and settlement days"):
        vs_base().with_effective_date(Date.from_ymd(1, Month.July, 2026)).with_settlement_days(
            2
        ).build()


def test_make_vanilla_swap_raises_unknown_currency(cpp: dict[str, Any]) -> None:
    assert cpp["vs_raises_unknown_currency"]["raises"] is True
    with pytest.raises(LibraryException, match="unknown fixed leg default tenor"):
        MakeVanillaSwap(
            Period(5, _YEARS), generic_index(CADCurrency(), Period(3, _MONTHS))
        ).build()


def test_make_vanilla_swap_raises_null_term_structure(cpp: dict[str, Any]) -> None:
    assert cpp["vs_raises_null_term_structure"]["raises"] is True
    with pytest.raises(LibraryException, match="null term structure"):
        MakeVanillaSwap(Period(5, _YEARS), Euribor.six_months()).build()


def test_make_vanilla_swap_call_is_build(cpp: dict[str, Any]) -> None:
    """``__call__`` mirrors C++'s ``operator VanillaSwap()``."""
    _check_swap(vs_base()(), cpp["vs_base"])


# ===========================================================================
# MakeOIS
# ===========================================================================


def ois_base() -> MakeOIS:
    return MakeOIS(Period(2, _YEARS), sofr())


def _ois_eom(setter: Callable[[MakeOIS], MakeOIS]) -> MakeOIS:
    return setter(ois_base().with_effective_date(EOM_START))


OIS_CASES: dict[str, Callable[[], MakeOIS]] = {
    "ois_base": ois_base,
    "ois_ctor_fixed_rate": lambda: MakeOIS(Period(2, _YEARS), sofr(), 0.02),
    "ois_ctor_forward_start": lambda: MakeOIS(
        Period(2, _YEARS), sofr(), None, Period(3, _MONTHS)
    ),
    "ois_receive_fixed": lambda: ois_base().receive_fixed(),
    "ois_with_type_receiver": lambda: ois_base().with_type(SwapType.Receiver),
    "ois_nominal": lambda: ois_base().with_nominal(1.0e6),
    "ois_settlement_days": lambda: ois_base().with_settlement_days(5),
    "ois_effective_date": lambda: ois_base().with_effective_date(
        Date.from_ymd(1, Month.July, 2026)
    ),
    "ois_termination_date": lambda: ois_base().with_termination_date(
        Date.from_ymd(17, Month.June, 2028)
    ),
    "ois_rule": lambda: ois_base().with_rule(DateGeneration.Forward),
    "ois_fixed_leg_rule": lambda: ois_base().with_fixed_leg_rule(DateGeneration.Forward),
    "ois_overnight_leg_rule": lambda: ois_base().with_overnight_leg_rule(
        DateGeneration.Forward
    ),
    "ois_fixed_leg_rule_zero": lambda: ois_base().with_fixed_leg_rule(DateGeneration.Zero),
    "ois_overnight_leg_rule_zero": lambda: ois_base().with_overnight_leg_rule(
        DateGeneration.Zero
    ),
    "ois_payment_frequency": lambda: ois_base().with_payment_frequency(Frequency.Semiannual),
    "ois_fixed_leg_payment_frequency": lambda: ois_base().with_fixed_leg_payment_frequency(
        Frequency.Semiannual
    ),
    "ois_overnight_leg_payment_frequency": lambda: (
        ois_base().with_overnight_leg_payment_frequency(Frequency.Semiannual)
    ),
    "ois_payment_frequency_once": lambda: ois_base().with_payment_frequency(Frequency.Once),
    "ois_payment_adjustment": lambda: ois_base().with_payment_adjustment(
        BusinessDayConvention.Preceding
    ),
    # the defect this cluster exists to prevent
    "ois_payment_lag": lambda: ois_base().with_payment_lag(2),
    "ois_payment_calendar": lambda: ois_base().with_payment_calendar(UnitedKingdom()),
    "ois_calendar": lambda: ois_base().with_calendar(UnitedKingdom()),
    "ois_fixed_leg_calendar": lambda: ois_base().with_fixed_leg_calendar(UnitedKingdom()),
    "ois_overnight_leg_calendar": lambda: ois_base().with_overnight_leg_calendar(
        UnitedKingdom()
    ),
    "ois_convention": lambda: ois_base().with_convention(BusinessDayConvention.Preceding),
    "ois_fixed_leg_convention": lambda: ois_base().with_fixed_leg_convention(
        BusinessDayConvention.Preceding
    ),
    "ois_overnight_leg_convention": lambda: ois_base().with_overnight_leg_convention(
        BusinessDayConvention.Preceding
    ),
    "ois_termination_convention": lambda: ois_base().with_termination_date_convention(
        BusinessDayConvention.Preceding
    ),
    "ois_fixed_leg_termination_convention": lambda: (
        ois_base().with_fixed_leg_termination_date_convention(BusinessDayConvention.Preceding)
    ),
    "ois_overnight_leg_termination_convention": lambda: (
        ois_base().with_overnight_leg_termination_date_convention(
            BusinessDayConvention.Preceding
        )
    ),
    # end-of-month family — MakeOIS's is_default_eom logic
    "ois_eom_default": lambda: _ois_eom(lambda b: b),
    "ois_eom_true": lambda: _ois_eom(lambda b: b.with_end_of_month(True)),
    "ois_eom_false": lambda: _ois_eom(lambda b: b.with_end_of_month(False)),
    "ois_eom_fixed_leg": lambda: _ois_eom(lambda b: b.with_fixed_leg_end_of_month(True)),
    "ois_eom_overnight_leg": lambda: _ois_eom(
        lambda b: b.with_overnight_leg_end_of_month(True)
    ),
    "ois_eom_maturity": lambda: _ois_eom(lambda b: b.with_maturity_end_of_month(True)),
    "ois_fixed_leg_day_count": lambda: ois_base().with_fixed_leg_day_count(Actual365Fixed()),
    "ois_overnight_leg_spread": lambda: ois_base().with_overnight_leg_spread(0.0025),
    "ois_discounting_ts": lambda: ois_base().with_discounting_term_structure(disc_curve()),
    "ois_pricing_engine": lambda: ois_base().with_pricing_engine(
        DiscountingSwapEngine(disc_curve(), include_settlement_date_flows=False)
    ),
    # C++ produces bit-identical results with and without the telescopic
    # optimisation when every fixing is forecast, which is the case here.
    "ois_telescopic_value_dates": lambda: ois_base().with_telescopic_value_dates(True),
    # per-index default settlement days: Sonia 0, everything else 2
    "ois_sonia_default_spot": lambda: MakeOIS(Period(2, _YEARS), Sonia(fwd_curve())),
}


@pytest.mark.parametrize("key", sorted(OIS_CASES))
def test_make_ois(key: str, cpp: dict[str, Any]) -> None:
    _check_swap(OIS_CASES[key]().build(), cpp[key])


def test_make_ois_raises_effective_and_settlement(cpp: dict[str, Any]) -> None:
    assert cpp["ois_raises_effective_and_settlement"]["raises"] is True
    with pytest.raises(LibraryException, match="effective date and settlement days"):
        ois_base().with_effective_date(Date.from_ymd(1, Month.July, 2026)).with_settlement_days(
            2
        ).build()


def test_make_ois_raises_null_term_structure(cpp: dict[str, Any]) -> None:
    assert cpp["ois_raises_null_term_structure"]["raises"] is True
    with pytest.raises(LibraryException, match="null term structure"):
        MakeOIS(Period(2, _YEARS), Sofr()).build()


def test_make_ois_default_settlement_days_is_not_sonia_for_a_clone() -> None:
    """A cloned Sonia takes the 2-day default, as in C++.

    ``OvernightIndex::clone`` returns a plain ``OvernightIndex`` on both sides,
    so the ``dynamic_pointer_cast<Sonia>`` in C++ fails for a clone exactly as
    the ``isinstance`` check does here.
    """
    clone = Sonia(fwd_curve()).clone(fwd_curve())
    assert isinstance(clone, OvernightIndex)
    plain = MakeOIS(Period(2, _YEARS), clone).build()
    sonia = MakeOIS(Period(2, _YEARS), Sonia(fwd_curve())).build()
    assert plain.start_date() != sonia.start_date()


# --- the four MakeOIS setters this port cannot yet honour -------------------
#
# OvernightIndexedSwap builds its leg through ``overnight_leg``, which exposes
# neither an averaging method nor the lookback / lockout / observation-shift
# modifiers. C++ reference values for all four are in the probe
# (ois_averaging_simple, ois_lookback_days, ois_lockout_days,
# ois_observation_shift) and each differs from ois_base, so they cannot be
# silently accepted: MakeOIS rejects them instead.


OIS_UNSUPPORTED: list[tuple[str, Callable[[MakeOIS], MakeOIS], str]] = [
    (
        "ois_averaging_simple",
        lambda b: b.with_averaging_method(RateAveraging.Simple),
        r"only RateAveraging\.Compound is supported",
    ),
    ("ois_lookback_days", lambda b: b.with_lookback_days(2), "lookback days"),
    ("ois_lockout_days", lambda b: b.with_lockout_days(2), "lockout days"),
    (
        "ois_observation_shift",
        lambda b: b.with_observation_shift(True),
        "observation shift",
    ),
]


@pytest.mark.parametrize(("ref_key", "setter", "message"), OIS_UNSUPPORTED)
def test_make_ois_rejects_unsupported_setters(
    ref_key: str, setter: Callable[[MakeOIS], MakeOIS], message: str, cpp: dict[str, Any]
) -> None:
    # The C++ reference really is different from the baseline, i.e. the setter
    # is not a no-op that could safely be ignored.
    assert cpp[ref_key]["npv"] != cpp["ois_base"]["npv"]
    with pytest.raises(LibraryException, match=message):
        setter(ois_base()).build()


def test_make_ois_averaging_method_compound_is_accepted(cpp: dict[str, Any]) -> None:
    """Setting the averaging method back to its default is not rejected."""
    swap = (
        ois_base()
        .with_averaging_method(RateAveraging.Simple)
        .with_averaging_method(RateAveraging.Compound)
        .build()
    )
    _check_swap(swap, cpp["ois_base"])


# ===========================================================================
# MakeCreditDefaultSwap
# ===========================================================================


def cds_engine() -> PricingEngine:
    return MidPointCdsEngine(
        FlatHazardRate.from_rate(TODAY, 0.02, Actual365Fixed()), 0.4, _flat(0.025)
    )


def cds_base() -> MakeCreditDefaultSwap:
    return MakeCreditDefaultSwap(tenor=Period(5, _YEARS), running_spread=0.01).with_pricing_engine(
        cds_engine()
    )


def _cds_schedule() -> Schedule:
    from pquantlib.time.calendars.weekends_only import WeekendsOnly  # noqa: PLC0415

    return Schedule.from_rule(
        Date.from_ymd(20, Month.June, 2026),
        Date.from_ymd(20, Month.June, 2029),
        Period(3, _MONTHS),
        WeekendsOnly(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Unadjusted,
        DateGeneration.CDS,
        False,
    )


CDS_CASES: dict[str, Callable[[], MakeCreditDefaultSwap]] = {
    "cds_base": cds_base,
    # the other two constructor overloads
    "cds_ctor_term_date": lambda: MakeCreditDefaultSwap(
        termination_date=Date.from_ymd(20, Month.June, 2031), running_spread=0.01
    ).with_pricing_engine(cds_engine()),
    "cds_ctor_schedule": lambda: MakeCreditDefaultSwap(
        schedule=_cds_schedule(), running_spread=0.01
    ).with_pricing_engine(cds_engine()),
    "cds_side_seller": lambda: cds_base().with_side(ProtectionSide.Seller),
    "cds_nominal": lambda: cds_base().with_nominal(1.0e7),
    "cds_upfront_rate": lambda: cds_base().with_upfront_rate(0.05),
    "cds_coupon_tenor": lambda: cds_base().with_coupon_tenor(Period(6, _MONTHS)),
    "cds_rule_cds2015": lambda: cds_base().with_date_generation_rule(DateGeneration.CDS2015),
    "cds_rule_old_cds": lambda: cds_base().with_date_generation_rule(DateGeneration.OldCDS),
    "cds_rule_backward": lambda: cds_base().with_date_generation_rule(
        DateGeneration.Backward
    ),
    "cds_convention": lambda: cds_base().with_convention(
        BusinessDayConvention.ModifiedFollowing
    ),
    "cds_day_counter": lambda: cds_base().with_day_counter(Actual365Fixed()),
    "cds_settle_accrual_false": lambda: cds_base().settle_accrual(False),
    "cds_pay_at_default_false": lambda: cds_base().pay_at_default_time(False),
    "cds_protection_start": lambda: cds_base().with_protection_start(
        Date.from_ymd(10, Month.June, 2026)
    ),
    "cds_upfront_date": lambda: cds_base()
    .with_upfront_rate(0.05)
    .with_upfront_date(Date.from_ymd(25, Month.June, 2026)),
    # an upfront DATE with a ZERO upfront rate still moves the accrual rebate
    "cds_upfront_date_zero_rate": lambda: cds_base().with_upfront_date(
        Date.from_ymd(25, Month.June, 2026)
    ),
    "cds_claim": lambda: cds_base().with_claim(HalfNotionalClaim()),
    "cds_last_period_day_counter": lambda: cds_base().with_last_period_day_counter(
        Actual360(include_last_day=False)
    ),
    "cds_rebate_accrual_false": lambda: cds_base().rebate_accrual(False),
    "cds_trade_date": lambda: cds_base().with_trade_date(Date.from_ymd(10, Month.June, 2026)),
    "cds_cash_settlement_days": lambda: cds_base().with_cash_settlement_days(5),
}


@pytest.mark.parametrize("key", sorted(CDS_CASES))
def test_make_credit_default_swap(key: str, cpp: dict[str, Any]) -> None:
    _check_cds(CDS_CASES[key]().build(), cpp[key])


def test_make_cds_alias_is_the_cpp_class() -> None:
    """``MakeCDS`` stays available as an alias for existing call sites."""
    assert MakeCDS is MakeCreditDefaultSwap


@pytest.mark.parametrize(
    ("legacy", "cpp_name"),
    [
        ("with_notional", "with_nominal"),
        ("with_rule", "with_date_generation_rule"),
        ("settles_accrual", "settle_accrual"),
        ("pays_at_default_time", "pay_at_default_time"),
        ("rebates_accrual", "rebate_accrual"),
    ],
)
def test_make_cds_legacy_setter_names_still_work(legacy: str, cpp_name: str) -> None:
    builder = MakeCreditDefaultSwap(tenor=Period(5, _YEARS), running_spread=0.01)
    assert hasattr(builder, legacy)
    assert hasattr(builder, cpp_name)


CDS_MATURITY_CASES: dict[str, tuple[Date, Period, DateGeneration]] = {
    "cds_maturity_5y_cds": (TODAY, Period(5, _YEARS), DateGeneration.CDS),
    "cds_maturity_5y_cds2015": (TODAY, Period(5, _YEARS), DateGeneration.CDS2015),
    "cds_maturity_5y_old_cds": (TODAY, Period(5, _YEARS), DateGeneration.OldCDS),
    "cds_maturity_3m_cds": (TODAY, Period(3, _MONTHS), DateGeneration.CDS),
    "cds_maturity_dec_anchor_cds2015": (
        Date.from_ymd(15, Month.January, 2027),
        Period(5, _YEARS),
        DateGeneration.CDS2015,
    ),
    "cds_maturity_dec_anchor_cds": (
        Date.from_ymd(15, Month.January, 2027),
        Period(5, _YEARS),
        DateGeneration.CDS,
    ),
}


@pytest.mark.parametrize("key", sorted(CDS_MATURITY_CASES))
def test_cds_maturity(key: str, cpp: dict[str, Any]) -> None:
    trade_date, tenor, rule = CDS_MATURITY_CASES[key]
    assert cds_maturity(trade_date, tenor, rule).serial_number() == cpp[key]


@pytest.mark.parametrize(
    ("ref_key", "tenor", "rule", "message"),
    [
        (
            "cds_maturity_raises_bad_rule",
            Period(5, _YEARS),
            DateGeneration.Backward,
            "CDS2015, CDS or OldCDS",
        ),
        (
            "cds_maturity_raises_bad_tenor",
            Period(4, _MONTHS),
            DateGeneration.CDS,
            "multiple of 3 months",
        ),
        (
            "cds_maturity_raises_zero_old_cds",
            Period(0, _MONTHS),
            DateGeneration.OldCDS,
            "0M is not supported",
        ),
    ],
)
def test_cds_maturity_raises(
    ref_key: str, tenor: Period, rule: DateGeneration, message: str, cpp: dict[str, Any]
) -> None:
    assert cpp[ref_key]["raises"] is True
    with pytest.raises(LibraryException, match=message):
        cds_maturity(TODAY, tenor, rule)


# ===========================================================================
# MakeMultipleResetsSwap / MultipleResetsSwap
# ===========================================================================


def mrs_base() -> MakeMultipleResetsSwap:
    return MakeMultipleResetsSwap(Period(1, _YEARS), euribor3m(), 2)


MRS_CASES: dict[str, Callable[[], MakeMultipleResetsSwap]] = {
    "mrs_base": mrs_base,
    "mrs_receive_fixed": lambda: mrs_base().receive_fixed(),
    "mrs_with_type_receiver": lambda: mrs_base().with_type(SwapType.Receiver),
    "mrs_nominal": lambda: mrs_base().with_nominal(1.0e6),
    "mrs_fixed_rate": lambda: mrs_base().with_fixed_rate(0.02),
    "mrs_settlement_days": lambda: mrs_base().with_settlement_days(5),
    "mrs_effective_date": lambda: mrs_base().with_effective_date(
        Date.from_ymd(1, Month.July, 2026)
    ),
    "mrs_termination_date": lambda: mrs_base().with_termination_date(
        Date.from_ymd(17, Month.June, 2027)
    ),
    "mrs_forward_start": lambda: mrs_base().with_forward_start(Period(3, _MONTHS)),
    "mrs_fixed_leg_frequency": lambda: mrs_base().with_fixed_leg_frequency(Frequency.Annual),
    "mrs_fixed_leg_day_count": lambda: mrs_base().with_fixed_leg_day_count(Actual365Fixed()),
    "mrs_fixed_leg_convention": lambda: mrs_base().with_fixed_leg_convention(
        BusinessDayConvention.Preceding
    ),
    "mrs_float_leg_spread": lambda: mrs_base().with_floating_leg_spread(0.0025),
    "mrs_averaging_simple": lambda: mrs_base().with_averaging_method(RateAveraging.Simple),
    "mrs_discounting_ts": lambda: mrs_base().with_discounting_term_structure(disc_curve()),
    "mrs_pricing_engine": lambda: mrs_base().with_pricing_engine(
        DiscountingSwapEngine(disc_curve(), include_settlement_date_flows=False)
    ),
    "mrs_resets_per_coupon_4": lambda: MakeMultipleResetsSwap(
        Period(1, _YEARS), euribor3m(), 4
    ),
}


@pytest.mark.parametrize("key", sorted(MRS_CASES))
def test_make_multiple_resets_swap(key: str, cpp: dict[str, Any]) -> None:
    _check_swap(MRS_CASES[key]().build(), cpp[key])


def test_make_multiple_resets_swap_raises_adjusted_end_stub(cpp: dict[str, Any]) -> None:
    """The end date is adjusted before the reset schedule is generated.

    A 2Y tenor from the 17-Jun-2026 spot ends on a Saturday, so the adjusted end
    seeds the backward generation one roll away from the start and leaves an odd
    number of reset periods.
    """
    assert cpp["mrs_raises_adjusted_end_stub"]["raises"] is True
    with pytest.raises(LibraryException, match="not a multiple of"):
        MakeMultipleResetsSwap(Period(2, _YEARS), euribor3m(), 2).build()


def test_make_multiple_resets_swap_raises_effective_and_settlement(
    cpp: dict[str, Any],
) -> None:
    assert cpp["mrs_raises_effective_and_settlement"]["raises"] is True
    with pytest.raises(LibraryException, match="mutually exclusive"):
        mrs_base().with_effective_date(
            Date.from_ymd(1, Month.July, 2026)
        ).with_settlement_days(2).build()


def test_make_multiple_resets_swap_raises_null_term_structure(cpp: dict[str, Any]) -> None:
    assert cpp["mrs_raises_null_term_structure"]["raises"] is True
    with pytest.raises(LibraryException, match="null term structure"):
        MakeMultipleResetsSwap(Period(1, _YEARS), Euribor.three_months(), 2).build()


def test_make_multiple_resets_swap_inspectors() -> None:
    swap = mrs_base().build()
    assert swap.resets_per_coupon() == 2
    assert swap.averaging_method() == RateAveraging.Compound
    assert swap.full_reset_schedule().size() == 5


# --- MultipleResetsSwap built directly -------------------------------------
#
# paymentConvention / paymentLag / paymentCalendar are constructor arguments the
# builder does not expose, so they are exercised on the instrument itself.


def _mrs_schedules() -> tuple[Schedule, Schedule]:
    start = Date.from_ymd(17, Month.June, 2026)
    end = Date.from_ymd(17, Month.June, 2028)
    cal = TARGET()
    fixed = Schedule.from_rule(
        start,
        end,
        Period(6, _MONTHS),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )
    resets = Schedule.from_rule(
        start,
        end,
        Period(3, _MONTHS),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )
    return fixed, resets


def _mrs_direct(
    payment_convention: BusinessDayConvention | None,
    payment_lag: int,
    payment_calendar: Any,
    averaging: RateAveraging,
) -> MultipleResetsSwap:
    fixed, resets = _mrs_schedules()
    swap = MultipleResetsSwap(
        SwapType.Payer,
        1.0e6,
        fixed,
        0.02,
        Actual360(),
        resets,
        euribor3m(),
        2,
        0.0,
        averaging,
        payment_convention,
        payment_lag,
        payment_calendar,
    )
    swap.set_pricing_engine(
        DiscountingSwapEngine(fwd_curve(), include_settlement_date_flows=False)
    )
    return swap


MRS_DIRECT_CASES: dict[str, Callable[[], MultipleResetsSwap]] = {
    "mrs_direct_base": lambda: _mrs_direct(None, 0, None, RateAveraging.Compound),
    "mrs_direct_payment_convention": lambda: _mrs_direct(
        BusinessDayConvention.Preceding, 0, None, RateAveraging.Compound
    ),
    # the defect this cluster exists to prevent, on the instrument itself
    "mrs_direct_payment_lag": lambda: _mrs_direct(None, 3, None, RateAveraging.Compound),
    "mrs_direct_payment_calendar": lambda: _mrs_direct(
        None, 3, UnitedKingdom(), RateAveraging.Compound
    ),
    "mrs_direct_averaging_simple": lambda: _mrs_direct(None, 0, None, RateAveraging.Simple),
}


@pytest.mark.parametrize("key", sorted(MRS_DIRECT_CASES))
def test_multiple_resets_swap_direct(key: str, cpp: dict[str, Any]) -> None:
    _check_swap(MRS_DIRECT_CASES[key](), cpp[key])


def test_multiple_resets_swap_raises_not_a_multiple(cpp: dict[str, Any]) -> None:
    assert cpp["mrs_direct_raises_not_a_multiple"]["raises"] is True
    fixed, resets = _mrs_schedules()
    with pytest.raises(LibraryException, match="not a multiple of"):
        MultipleResetsSwap(
            SwapType.Payer, 1.0e6, fixed, 0.02, Actual360(), resets, euribor3m(), 3
        )


# ===========================================================================
# The keyword-argument façades and the PQuantLib-only setters
# ===========================================================================
#
# make_vanilla_swap / make_ois are thin wrappers that map one keyword to one
# chained setter. A keyword that never reaches its setter is exactly the defect
# this file exists to catch, so each of the ones whose value is easy to get
# wrong is checked against the reference case the corresponding setter produces.


FACADE_VS_CASES: dict[str, Callable[[], VanillaSwap]] = {
    "vs_float_leg_spread": lambda: make_vanilla_swap(
        Period(5, _YEARS), euribor6m(), floating_leg_spread=0.0025
    ),
    "vs_fixed_leg_first_date": lambda: make_vanilla_swap(
        Period(5, _YEARS),
        euribor6m(),
        effective_date=Date.from_ymd(17, Month.June, 2026),
        fixed_leg_first_date=Date.from_ymd(17, Month.December, 2026),
    ),
    "vs_float_leg_next_to_last_date": lambda: make_vanilla_swap(
        Period(5, _YEARS),
        euribor6m(),
        effective_date=Date.from_ymd(17, Month.June, 2026),
        floating_leg_next_to_last_date=Date.from_ymd(17, Month.March, 2031),
    ),
    "vs_indexed_coupons": lambda: make_vanilla_swap(
        Period(5, _YEARS),
        euribor6m(),
        floating_leg_tenor=Period(3, _MONTHS),
        use_indexed_coupons=True,
    ),
    "vs_eom_maturity": lambda: make_vanilla_swap(
        Period(5, _YEARS), euribor6m(), effective_date=EOM_START, maturity_end_of_month=True
    ),
    "vs_pricing_engine": lambda: make_vanilla_swap(
        Period(5, _YEARS),
        euribor6m(),
        pricing_engine=DiscountingSwapEngine(
            disc_curve(), include_settlement_date_flows=False
        ),
    ),
}


@pytest.mark.parametrize("key", sorted(FACADE_VS_CASES))
def test_make_vanilla_swap_facade(key: str, cpp: dict[str, Any]) -> None:
    _check_swap(FACADE_VS_CASES[key](), cpp[key])


FACADE_OIS_CASES: dict[str, Callable[[], OvernightIndexedSwap]] = {
    "ois_payment_lag": lambda: make_ois(Period(2, _YEARS), sofr(), payment_lag=2),
    # Before this port grew the per-leg end-of-month keywords, the façade read
    # ``maturity_end_of_month`` only when ``end_of_month`` was also given, so
    # this call silently produced the default-end-of-month swap instead.
    "ois_eom_maturity": lambda: make_ois(
        Period(2, _YEARS), sofr(), effective_date=EOM_START, maturity_end_of_month=True
    ),
    "ois_eom_fixed_leg": lambda: make_ois(
        Period(2, _YEARS), sofr(), effective_date=EOM_START, fixed_leg_end_of_month=True
    ),
    "ois_eom_overnight_leg": lambda: make_ois(
        Period(2, _YEARS), sofr(), effective_date=EOM_START, overnight_leg_end_of_month=True
    ),
    "ois_eom_default": lambda: make_ois(
        Period(2, _YEARS), sofr(), effective_date=EOM_START
    ),
    "ois_overnight_leg_spread": lambda: make_ois(
        Period(2, _YEARS), sofr(), overnight_leg_spread=0.0025
    ),
    "ois_fixed_leg_rule_zero": lambda: make_ois(
        Period(2, _YEARS), sofr(), fixed_rule=DateGeneration.Zero
    ),
}


@pytest.mark.parametrize("key", sorted(FACADE_OIS_CASES))
def test_make_ois_facade(key: str, cpp: dict[str, Any]) -> None:
    _check_swap(FACADE_OIS_CASES[key](), cpp[key])


def test_make_ois_facade_rejects_unsupported_averaging_method() -> None:
    """The façade's ``averaging_method`` reaches the builder's guard."""
    with pytest.raises(LibraryException, match=r"only RateAveraging\.Compound is supported"):
        make_ois(Period(2, _YEARS), sofr(), averaging_method=RateAveraging.Simple)


# --- with_evaluation_date (PQuantLib-only) ---------------------------------


def test_with_evaluation_date_overrides_the_global() -> None:
    """The local reference date wins over the pinned global one.

    The autouse fixture pins the global to 15-Jun-2026, whose spot is
    17-Jun-2026; pinning the builder to a week later must move the start date
    without touching global state.
    """
    later = Date.from_ymd(22, Month.June, 2026)
    swap = MakeVanillaSwap(Period(5, _YEARS), euribor6m()).with_evaluation_date(later).build()
    assert swap.start_date() == Date.from_ymd(24, Month.June, 2026)
    assert ObservableSettings().evaluation_date == TODAY

    ois = MakeOIS(Period(2, _YEARS), sofr()).with_evaluation_date(later).build()
    assert ois.start_date() == Date.from_ymd(24, Month.June, 2026)

    mrs = (
        MakeMultipleResetsSwap(Period(1, _YEARS), euribor3m(), 2)
        .with_evaluation_date(later)
        .build()
    )
    assert mrs.start_date() == Date.from_ymd(24, Month.June, 2026)


def test_facade_evaluation_date_keyword_reaches_the_builder() -> None:
    later = Date.from_ymd(22, Month.June, 2026)
    swap = make_vanilla_swap(Period(5, _YEARS), euribor6m(), evaluation_date=later)
    assert swap.start_date() == Date.from_ymd(24, Month.June, 2026)
    ois = make_ois(Period(2, _YEARS), sofr(), evaluation_date=later)
    assert ois.start_date() == Date.from_ymd(24, Month.June, 2026)


# --- MakeCreditDefaultSwap's two PQuantLib-only setters --------------------


def test_make_cds_with_calendar_moves_the_cash_settlement_date() -> None:
    """``with_calendar`` (C++ hard-codes WeekendsOnly) reaches the upfront advance.

    Three business days after Thursday 24-Dec-2026 is Tuesday 29-Dec on
    WeekendsOnly, which does not know Christmas Day is a holiday, and Wednesday
    30-Dec on TARGET, which does.
    """
    trade_date = Date.from_ymd(24, Month.December, 2026)
    default_cds = cds_base().with_trade_date(trade_date).build()
    target_cds = cds_base().with_trade_date(trade_date).with_calendar(TARGET()).build()
    assert default_cds.upfront_payment().date() == Date.from_ymd(29, Month.December, 2026)
    assert target_cds.upfront_payment().date() == Date.from_ymd(30, Month.December, 2026)


def test_make_cds_termination_date_convention_changes_the_maturity() -> None:
    """``with_termination_date_convention`` (C++ hard-codes Unadjusted) applies.

    21-Jun-2031 is a Saturday, so it survives ``Unadjusted`` and rolls back to
    Friday the 20th under ``Preceding``.
    """
    saturday = Date.from_ymd(21, Month.June, 2031)

    def builder() -> MakeCreditDefaultSwap:
        return (
            MakeCreditDefaultSwap(termination_date=saturday, running_spread=0.01)
            .with_date_generation_rule(DateGeneration.Backward)
            .with_pricing_engine(cds_engine())
        )

    assert builder().build().protection_end_date() == saturday
    preceding = (
        builder().with_termination_date_convention(BusinessDayConvention.Preceding).build()
    )
    assert preceding.protection_end_date() == Date.from_ymd(20, Month.June, 2031)


def test_make_cds_without_an_engine_cannot_price() -> None:
    """``with_pricing_engine`` is what makes the CDS priceable at all."""
    cds = MakeCreditDefaultSwap(tenor=Period(5, _YEARS), running_spread=0.01).build()
    with pytest.raises(LibraryException):
        cds.npv()
