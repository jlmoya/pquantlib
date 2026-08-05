"""Cross-validate ``EquityTotalReturnSwap`` against C++ v1.43.

Probe: ``v143/inst/etrs``.

Both C++ constructors are covered — the Ibor flavour and the overnight
flavour — and each is probed at ``payment_delay = 0`` and at
``payment_delay = 2`` with a non-default payment calendar and convention.
That pair is the point of the file: ``payment_delay`` is threaded to
``withPaymentLag`` on the interest leg **and** to the equity cash flow's
payment date, and an accepted-then-dropped payment lag is the defect class
this port is prone to. The reference itself shows the delay moving dates
(interest coupon 46371 -> 46373, equity flow 46919 -> 46924), so a port
that dropped it cannot pass.

Every flow of both legs is asserted, not only the NPV: an equity-leg error
and an interest-leg error would otherwise be free to cancel.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.equity_index import EquityIndex
from pquantlib.indexes.ibor.estr import Estr
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.instruments.equity_total_return_swap import EquityTotalReturnSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule, Schedule
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(15, Month.June, 2026)
_A365 = Actual365Fixed()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/etrs")


def _flat(r: float) -> FlatForward:
    return FlatForward.from_rate(_TODAY, r, _A365)


# Forward-starting, exactly as the probe builds it: every Euribor6M fixing then
# falls after the evaluation date, so no coupon depends on a historical fixing.
_START = _TODAY + Period(6, TimeUnit.Months)


def _schedule() -> Schedule:
    return (
        MakeSchedule()
        .from_date(_START)
        .to(_START + Period(2, TimeUnit.Years))
        .with_frequency(Frequency.Semiannual)
        .with_calendar(TARGET())
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )


@pytest.fixture
def market() -> tuple[EquityIndex, Euribor, Estr, DiscountingSwapEngine]:
    IndexManager().clear_histories()
    forecast = _flat(0.035)
    equity_index = EquityIndex(
        "eqIndex",
        TARGET(),
        EURCurrency(),
        _flat(0.025),
        _flat(0.015),
        SimpleQuote(8700.0),
    )
    euribor = Euribor.six_months(forecast)
    estr = Estr(forecast)
    # Same past fixings the probe adds: the first Euribor coupon fixes two
    # business days before the schedule start, i.e. before the evaluation
    # date, so it must come from history.
    euribor.add_fixing(Date.from_ymd(11, Month.June, 2026), 0.0341)
    equity_index.add_fixing(_TODAY, 8695.5)
    return equity_index, euribor, estr, DiscountingSwapEngine(_flat(0.03))


def _build(
    key: str, market: tuple[EquityIndex, Euribor, Estr, DiscountingSwapEngine]
) -> EquityTotalReturnSwap:
    equity_index, euribor, estr, engine = market
    schedule = _schedule()
    us = UnitedStates(UnitedStates.Market.GovernmentBond)
    specs: dict[str, EquityTotalReturnSwap] = {
        "ibor_plain": EquityTotalReturnSwap.with_ibor_index(
            SwapType.Payer, 1.0e7, schedule, equity_index, euribor, _A365, 0.0035
        ),
        "ibor_lagged": EquityTotalReturnSwap.with_ibor_index(
            SwapType.Payer,
            1.0e7,
            schedule,
            equity_index,
            euribor,
            _A365,
            0.0035,
            1.0,
            us,
            BusinessDayConvention.ModifiedFollowing,
            2,
        ),
        "ibor_geared_receiver": EquityTotalReturnSwap.with_ibor_index(
            SwapType.Receiver,
            5.0e6,
            schedule,
            equity_index,
            euribor,
            _A365,
            -0.0012,
            1.35,
            TARGET(),
            BusinessDayConvention.Following,
            1,
        ),
        "overnight_plain": EquityTotalReturnSwap.with_overnight_index(
            SwapType.Payer, 1.0e7, schedule, equity_index, estr, _A365, 0.0035
        ),
        "overnight_lagged": EquityTotalReturnSwap.with_overnight_index(
            SwapType.Payer,
            1.0e7,
            schedule,
            equity_index,
            estr,
            _A365,
            0.0035,
            1.0,
            us,
            BusinessDayConvention.ModifiedFollowing,
            2,
        ),
    }
    swap = specs[key]
    swap.set_pricing_engine(engine)
    return swap


_KEYS = (
    "ibor_plain",
    "ibor_lagged",
    "ibor_geared_receiver",
    "overnight_plain",
    "overnight_lagged",
)


def test_reference_dates(cpp: dict[str, Any]) -> None:
    assert _TODAY.serial_number() == cpp["today_serial"]
    assert _START.serial_number() == cpp["start_serial"]


@pytest.mark.parametrize("key", _KEYS)
def test_wiring(
    cpp: dict[str, Any],
    key: str,
    market: tuple[EquityIndex, Euribor, Estr, DiscountingSwapEngine],
) -> None:
    """Every constructor argument is readable back off the swap."""
    ref = cpp[key]
    swap = _build(key, market)
    assert int(swap.type()) == ref["type"]
    tight(swap.nominal(), ref["nominal"])
    tight(swap.margin(), ref["margin"])
    tight(swap.gearing(), ref["gearing"])
    assert swap.payment_delay() == ref["payment_delay"]
    assert int(swap.payment_convention()) == ref["payment_convention"]
    cal = swap.payment_calendar()
    assert (cal.name() if cal is not None else "") == ref["payment_calendar"]
    assert swap.day_counter().name() == ref["day_counter"]


@pytest.mark.parametrize("key", _KEYS)
def test_cashflows(
    cpp: dict[str, Any],
    key: str,
    market: tuple[EquityIndex, Euribor, Estr, DiscountingSwapEngine],
) -> None:
    """The full structure of both legs — flow by flow.

    This is where a dropped ``payment_delay`` / ``payment_calendar`` /
    ``payment_convention`` / ``gearing`` / ``margin`` shows up, and it is
    checked before any aggregate so a cancelling pair cannot hide.
    """
    ref = cpp[key]
    swap = _build(key, market)
    for leg_name, leg in (
        ("equity", swap.equity_leg()),
        ("interest", swap.interest_rate_leg()),
    ):
        expected = ref["legs"][leg_name]
        assert len(leg) == len(expected), leg_name
        for cf, flow in zip(leg, expected, strict=True):
            assert cf.date().serial_number() == flow["date_serial"], leg_name
            # LOOSE: the amounts run through the curves' exp/log, and the
            # overnight coupon compounds a product of daily factors.
            loose(cf.amount(), flow["amount"])
            if "nominal" in flow:
                assert isinstance(cf, Coupon)
                tight(cf.nominal(), flow["nominal"])
                assert cf.accrual_start_date().serial_number() == flow["accrual_start_serial"]
                assert cf.accrual_end_date().serial_number() == flow["accrual_end_serial"]
                tight(cf.accrual_period(), flow["accrual_period"])
                loose(cf.rate(), flow["rate"])


@pytest.mark.parametrize("key", _KEYS)
def test_results(
    cpp: dict[str, Any],
    key: str,
    market: tuple[EquityIndex, Euribor, Estr, DiscountingSwapEngine],
) -> None:
    ref = cpp[key]
    swap = _build(key, market)
    # LOOSE throughout: discounted sums over exp()-based discount factors.
    loose(swap.npv(), ref["npv"])
    loose(swap.equity_leg_npv(), ref["equity_leg_npv"])
    loose(swap.interest_rate_leg_npv(), ref["interest_rate_leg_npv"])
    loose(swap.leg_bps(1), ref["interest_leg_bps"])
    loose(swap.fair_margin(), ref["fair_margin"])


def test_payment_delay_actually_moves_the_dates(cpp: dict[str, Any]) -> None:
    """The reference itself must discriminate the delay.

    If C++ produced the same dates at delay 0 and delay 2, the tests above
    would pass for a port that dropped the argument.
    """
    for plain, lagged in (("ibor_plain", "ibor_lagged"), ("overnight_plain", "overnight_lagged")):
        p = cpp[plain]["legs"]
        q = cpp[lagged]["legs"]
        assert [f["date_serial"] for f in p["equity"]] != [f["date_serial"] for f in q["equity"]], plain
        assert [f["date_serial"] for f in p["interest"]] != [f["date_serial"] for f in q["interest"]], plain


def test_ibor_and_overnight_constructors_build_different_coupons(
    market: tuple[EquityIndex, Euribor, Estr, DiscountingSwapEngine],
) -> None:
    """The two constructors must not collapse onto the same leg builder.

    Deliberately a type assertion, not a value one: under the flat curve this
    probe uses, the compounded overnight rate and the simple Ibor forward are
    the *same number* — both are ``(P(s)/P(e) - 1) / tau`` — so C++'s own
    reference has identical rates for the two flavours. What distinguishes
    them is the coupon class, and hence which fixings they would consume off a
    non-flat curve.
    """
    ibor_leg = _build("ibor_plain", market).interest_rate_leg()
    overnight_leg = _build("overnight_plain", market).interest_rate_leg()
    assert all(isinstance(cf, IborCoupon) for cf in ibor_leg)
    assert all(isinstance(cf, OvernightIndexedCoupon) for cf in overnight_leg)


def test_negative_nominal_raises(
    cpp: dict[str, Any],
    market: tuple[EquityIndex, Euribor, Estr, DiscountingSwapEngine],
) -> None:
    ref = cpp["negative_nominal"]
    assert ref["raises"] is True
    equity_index, euribor, _, _ = market
    with pytest.raises(LibraryException) as excinfo:
        EquityTotalReturnSwap.with_ibor_index(
            SwapType.Payer, -1.0, _schedule(), equity_index, euribor, _A365, 0.0
        )
    assert ref["error"] in str(excinfo.value)
