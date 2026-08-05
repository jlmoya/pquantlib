"""Cross-validate Stock, CompositeInstrument, FaceValueAccrualClaim and
ImpliedVolatilityHelper against C++ v1.43.

Probe: ``v143/inst/misc``.

Each of the four is small, and each has a way to be silently wrong:

* ``CompositeInstrument``'s multiplier is signed and ``subtract`` is defined
  as ``add(-multiplier)``; the components' own NPVs are pinned alongside the
  total so a cancelling pair of sign errors cannot hide.
* ``CompositeInstrument.is_expired`` is an AND over components, not an OR.
* ``FaceValueAccrualClaim`` divides accrued by ``notional(d)``, which moves
  over an amortising bond's life — the reference bond here amortises so that
  using the initial face amount instead would be visibly wrong.
* ``ImpliedVolatilityHelper.clone`` must carry the spot, dividend curve and
  risk-free curve across and replace only the vol; the clone's own spot and
  discount factors are pinned, not just the recovered volatility.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.simple_cash_flow import AmortizingPayment
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.bond import Bond
from pquantlib.instruments.claim import FaceValueAccrualClaim, FaceValueClaim
from pquantlib.instruments.composite_instrument import CompositeInstrument
from pquantlib.instruments.implied_volatility import ImpliedVolatilityHelper
from pquantlib.instruments.instrument import Instrument
from pquantlib.instruments.stock import Stock
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(15, Month.June, 2026)
_A365 = Actual365Fixed()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/misc")


def test_reference_today(cpp: dict[str, Any]) -> None:
    assert _TODAY.serial_number() == cpp["today_serial"]


# =====================================================================
# Stock
# =====================================================================


def test_stock(cpp: dict[str, Any]) -> None:
    ref = cpp["stock"]
    quote = SimpleQuote(ref["quote"])
    stock = Stock(quote)
    # EXACT would also hold (NPV is the quote verbatim), but TIGHT keeps the
    # tier vocabulary uniform across the file.
    tight(stock.npv(), ref["npv"])
    assert stock.is_expired() is ref["is_expired"]

    # The quote is observed: bumping it must invalidate the cached NPV.
    quote.set_value(ref["npv_after_quote_bump"])
    tight(stock.npv(), ref["npv_after_quote_bump"])


def test_stock_null_quote_raises(cpp: dict[str, Any]) -> None:
    ref = cpp["stock"]
    assert ref["null_quote_raises"] is True
    with pytest.raises(LibraryException) as excinfo:
        Stock(None).npv()
    assert ref["null_quote_error"] in str(excinfo.value)


# =====================================================================
# CompositeInstrument
# =====================================================================


def _process(spot: float, q: float, r: float, vol: float) -> GeneralizedBlackScholesProcess:
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(spot),
        dividend_ts=FlatForward.from_rate(_TODAY, q, _A365),
        risk_free_ts=FlatForward.from_rate(_TODAY, r, _A365),
        black_vol_ts=BlackConstantVol(
            reference_date=_TODAY,
            calendar=TARGET(),
            day_counter=_A365,
            volatility=vol,
        ),
    )


def _option(option_type: OptionType, strike: float, expiry: Date) -> VanillaOption:
    return VanillaOption(PlainVanillaPayoff(option_type, strike), EuropeanExercise(expiry))


class _ExpiredInstrument(Instrument):
    """A stand-in for an expired component.

    C++ uses an expired ``VanillaOption`` here. This port's
    ``VanillaOption.is_expired()`` is hard-coded ``False`` (see
    ``instruments/vanilla_option.py``), so an expired option would still be
    handed to the engine and raise on a past exercise date. The class under
    test is ``CompositeInstrument``, whose ``is_expired`` is an AND over its
    components, so any expired component serves — and C++ prices an expired
    component at 0.0, which is exactly what ``setup_expired`` gives here.
    """

    def is_expired(self) -> bool:
        return True

    def _perform_calculations(self) -> None:  # pragma: no cover - never reached
        raise AssertionError("an expired instrument must not be priced")


@pytest.fixture
def composite_fixture() -> tuple[VanillaOption, VanillaOption, VanillaOption, Instrument]:
    process = _process(100.0, 0.03, 0.05, 0.20)
    engine = AnalyticEuropeanEngine(process)
    expiry = _TODAY + Period(1, TimeUnit.Years)
    call90 = _option(OptionType.Call, 90.0, expiry)
    call110 = _option(OptionType.Call, 110.0, expiry)
    put100 = _option(OptionType.Put, 100.0, expiry)
    for opt in (call90, call110, put100):
        opt.set_pricing_engine(engine)
    return call90, call110, put100, _ExpiredInstrument()


def test_composite_components_price_as_cpp(
    cpp: dict[str, Any],
    composite_fixture: tuple[VanillaOption, VanillaOption, VanillaOption, Instrument],
) -> None:
    """The parts, before the whole — so a cancelling pair cannot hide."""
    ref = cpp["composite"]
    call90, call110, put100, expired = composite_fixture
    # LOOSE: closed-form Black-Scholes, i.e. a cumulative-normal evaluation,
    # whose last bits differ between libm implementations.
    loose(call90.npv(), ref["call90_npv"])
    loose(call110.npv(), ref["call110_npv"])
    loose(put100.npv(), ref["put100_npv"])
    tight(expired.npv(), ref["expired_component_npv"])


def test_composite_signed_multipliers(
    cpp: dict[str, Any],
    composite_fixture: tuple[VanillaOption, VanillaOption, VanillaOption, Instrument],
) -> None:
    ref = cpp["composite"]
    call90, call110, put100, _ = composite_fixture
    composite = CompositeInstrument()
    composite.add(call90, 2.5)
    composite.subtract(call110, 1.75)
    composite.add(put100, 0.5)

    # The multipliers reached the slots they configure.
    assert [m for _, m in composite.components()] == [2.5, -1.75, 0.5]
    loose(composite.npv(), ref["npv"])
    assert composite.is_expired() is ref["is_expired"]


def test_composite_default_multiplier_is_one(
    cpp: dict[str, Any],
    composite_fixture: tuple[VanillaOption, VanillaOption, VanillaOption, Instrument],
) -> None:
    ref = cpp["composite"]
    call90, call110, _, _ = composite_fixture
    composite = CompositeInstrument()
    composite.add(call90)
    composite.subtract(call110)
    assert [m for _, m in composite.components()] == [1.0, -1.0]
    loose(composite.npv(), ref["npv_default_multipliers"])


def test_composite_empty_is_expired_and_worthless(cpp: dict[str, Any]) -> None:
    ref = cpp["composite"]
    composite = CompositeInstrument()
    assert composite.is_expired() is ref["empty_is_expired"]
    tight(composite.npv(), ref["empty_npv"])


def test_composite_is_expired_is_an_and(
    cpp: dict[str, Any],
    composite_fixture: tuple[VanillaOption, VanillaOption, VanillaOption, Instrument],
) -> None:
    """One live component keeps the composite live — AND, not OR."""
    ref = cpp["composite"]
    call90, _, _, expired = composite_fixture

    mixed = CompositeInstrument()
    mixed.add(expired, 1.0)
    mixed.add(call90, 1.0)
    assert mixed.is_expired() is ref["mixed_is_expired"]
    loose(mixed.npv(), ref["mixed_npv"])

    all_expired = CompositeInstrument()
    all_expired.add(expired, 1.0)
    assert all_expired.is_expired() is ref["all_expired_is_expired"]
    tight(all_expired.npv(), ref["all_expired_npv"])


def test_composite_deep_update_cascades(
    composite_fixture: tuple[VanillaOption, VanillaOption, VanillaOption, Instrument],
) -> None:
    """``deep_update`` reaches the components, not only the composite.

    C++ parity: compositeinstrument.cpp:60-65. Without the cascade the
    components keep their own stale caches.
    """
    call90, _, _, _ = composite_fixture
    composite = CompositeInstrument()
    composite.add(call90, 1.0)
    composite.npv()
    call90.calculate()

    composite.deep_update()
    # Both caches invalidated, not just the composite's.
    # is_calculated() is the observable: a non-cascading deep_update would
    # leave the component still calculated.
    assert composite.is_calculated() is False
    assert call90.is_calculated() is False


# =====================================================================
# FaceValueAccrualClaim
# =====================================================================


def _amortising_bond() -> Bond:
    schedule = (
        MakeSchedule()
        .from_date(Date.from_ymd(15, Month.June, 2025))
        .to(Date.from_ymd(15, Month.June, 2030))
        .with_frequency(Frequency.Annual)
        .with_calendar(TARGET())
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )
    notionals = [100.0, 90.0, 80.0, 70.0, 60.0]
    dc = Thirty360(Thirty360Convention.BondBasis)
    coupons: list[CashFlow] = []
    for i in range(len(schedule) - 1):
        coupons.append(
            FixedRateCoupon.from_rate(
                schedule.date(i + 1),
                notionals[i],
                0.04,
                dc,
                schedule.date(i),
                schedule.date(i + 1),
            )
        )
        amortised = notionals[i] - (notionals[i + 1] if i + 1 < len(notionals) else 0.0)
        coupons.append(AmortizingPayment(amortised, schedule.date(i + 1)))
    return Bond(2, TARGET(), Date.from_ymd(15, Month.June, 2025), coupons)


def test_face_value_accrual_claim(cpp: dict[str, Any]) -> None:
    bond = _amortising_bond()
    face_value = FaceValueClaim()
    accrual_claim = FaceValueAccrualClaim(bond)

    for case in cpp["claim"]["cases"]:
        d = Date(int(case["date_serial"]))
        # The divisor: an amortising notional, so a claim that used the
        # initial face amount would be wrong by the amortisation factor.
        tight(bond.notional(d), case["bond_notional"])
        tight(bond.accrued_amount(d), case["bond_accrued"])
        tight(face_value.amount(d, 1000.0, 0.4), case["face_value_claim"])
        tight(accrual_claim.amount(d, 1000.0, 0.4), case["face_value_accrual_claim"])
        # A second recovery rate, so a claim that dropped it still fails.
        tight(accrual_claim.amount(d, 1000.0, 0.0), case["face_value_accrual_claim_rr0"])


def test_face_value_accrual_claim_differs_from_face_value(cpp: dict[str, Any]) -> None:
    """The accrual term is load-bearing at least somewhere in the grid."""
    cases = cpp["claim"]["cases"]
    assert any(c["face_value_accrual_claim"] != c["face_value_claim"] for c in cases), (
        "probe grid no longer distinguishes the two claim conventions"
    )


def test_face_value_accrual_claim_keeps_its_reference_security() -> None:
    bond = _amortising_bond()
    assert FaceValueAccrualClaim(bond).reference_security() is bond


# =====================================================================
# ImpliedVolatilityHelper
# =====================================================================


def test_implied_volatility_helper(cpp: dict[str, Any]) -> None:
    for case in cpp["implied_vol"]["cases"]:
        expiry = Date(int(case["expiry_serial"]))
        process = _process(case["spot"], case["dividend_yield"], case["risk_free"], case["true_vol"])
        option = _option(OptionType(int(case["option_type"])), case["strike"], expiry)
        option.set_pricing_engine(AnalyticEuropeanEngine(process))
        # LOOSE: closed-form Black-Scholes through a cumulative normal.
        loose(option.npv(), case["target_value"])

        vol_quote = SimpleQuote(0.0)
        cloned = ImpliedVolatilityHelper.clone(process, vol_quote)

        # The clone carried spot / dividend / risk-free across untouched.
        tight(cloned.state_variable().value(), case["cloned_spot"])
        loose(cloned.dividend_yield().discount(expiry), case["cloned_dividend_discount"])
        loose(cloned.risk_free_rate().discount(expiry), case["cloned_riskfree_discount"])

        recovered = ImpliedVolatilityHelper.calculate(
            option,
            AnalyticEuropeanEngine(cloned),
            vol_quote,
            case["target_value"],
            1.0e-8,
            200,
            1.0e-7,
            4.0,
        )
        # LOOSE: the output of a Brent solve stopped at accuracy 1e-8 in vol.
        loose(recovered, case["implied_vol"])
        # ... and the quote the helper drove now holds it, which is what
        # makes the cloned vol surface report it.
        loose(cloned.black_volatility().black_vol(expiry, case["strike"]), case["cloned_vol_at_recovered"])


def test_implied_volatility_helper_recovers_the_true_vol(cpp: dict[str, Any]) -> None:
    """Sanity check with a different oracle: the vol that priced the target.

    The probe's `true_vol` is the vol the target price was computed at, so
    the helper must return it. Distinct grid points use distinct vols, so a
    helper that ignored `target_value` would fail.
    """
    for case in cpp["implied_vol"]["cases"]:
        loose(case["implied_vol"], case["true_vol"])


def test_vanilla_option_implied_volatility_uses_the_helper(cpp: dict[str, Any]) -> None:
    """``VanillaOption.implied_volatility`` must agree with the helper.

    It is implemented on top of it, so this is a wiring test: a
    ``VanillaOption`` that grew its own private copy would drift.
    """
    case = cpp["implied_vol"]["cases"][0]
    expiry = Date(int(case["expiry_serial"]))
    process = _process(case["spot"], case["dividend_yield"], case["risk_free"], case["true_vol"])
    option = _option(OptionType(int(case["option_type"])), case["strike"], expiry)
    option.set_pricing_engine(AnalyticEuropeanEngine(process))
    recovered = option.implied_volatility(case["target_value"], process, accuracy=1.0e-8, max_evaluations=200)
    loose(recovered, case["implied_vol"])
