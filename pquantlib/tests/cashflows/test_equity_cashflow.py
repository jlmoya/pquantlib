"""Cross-validate the equity cash-flow family against C++ QuantLib v1.43.

Probe: ``v143/cf/equitycashflow`` (sections ``equity_cash_flow``,
``quanto_grid``, ``pricer_raises`` and ``set_coupon_pricer``).

Market: the one used by the C++ test suite (test-suite/equitycashflow.cpp) —
TARGET / Actual365Fixed, evaluation date 27 January 2023, notional 1e7, flat
3.75% local interest, 0.5% dividend, 0.1% quanto currency, 40% equity vol, 20%
FX vol, correlation 0.4, spot 8700, historical fixings 9010 (5 Jan) and 8690
(today).

The payment date is deliberately a week LATER than the fixing date, so
``date()`` cannot silently return the fixing date; and every scenario pins the
full state that produced the amount — payment/base/fixing serials, notional,
``growth_only``, the two index fixings and the pricer's ``price()`` — because
an amount can match while two errors cancel.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.equity_cashflow import (
    EquityCashFlow,
    EquityCashFlowPricer,
    EquityQuantoCashFlowPricer,
    set_coupon_pricer,
)
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.equity_index import EquityIndex
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_TODAY = Date.from_ymd(27, Month.January, 2023)
_BASE_DATE = Date.from_ymd(5, Month.January, 2023)
_FIXING_DATE = Date.from_ymd(5, Month.April, 2023)
_PAYMENT_DATE = Date.from_ymd(12, Month.April, 2023)

_NOTIONAL = 1.0e7
_SPOT = 8700.0
_CORRELATION = 0.4
_INTEREST_RATE = 0.0375
_DIVIDEND_RATE = 0.005
_QUANTO_RATE = 0.001
_EQUITY_VOL = 0.4
_FX_VOL = 0.2
_BASE_FIXING = 9010.0
_TODAYS_FIXING = 8690.0


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/equitycashflow")


@pytest.fixture(autouse=True)
def _global_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    previous = ObservableSettings().evaluation_date
    ObservableSettings().evaluation_date = _TODAY
    IndexManager().clear_histories()
    yield
    IndexManager().clear_histories()
    ObservableSettings().evaluation_date = previous


class _Market:
    """The C++ ``CommonVars`` fixture, rebuilt per test."""

    def __init__(self) -> None:
        self.calendar = TARGET()
        self.day_count = Actual365Fixed()
        self.interest = FlatForward.from_rate(_TODAY, _INTEREST_RATE, self.day_count)
        self.dividend = FlatForward.from_rate(_TODAY, _DIVIDEND_RATE, self.day_count)
        self.quanto_interest = FlatForward.from_rate(_TODAY, _QUANTO_RATE, self.day_count)
        self.equity_vol = self.flat_vol(_EQUITY_VOL)
        self.fx_vol = self.flat_vol(_FX_VOL)
        self.spot = SimpleQuote(_SPOT)
        self.correlation = SimpleQuote(_CORRELATION)
        self.index = EquityIndex(
            "eqIndex", self.calendar, EURCurrency(), self.interest, self.dividend, self.spot
        )
        self.index.add_fixing(_BASE_DATE, _BASE_FIXING)
        self.index.add_fixing(_TODAY, _TODAYS_FIXING)

    def flat_vol(self, vol: float) -> BlackConstantVol:
        return BlackConstantVol(
            reference_date=_TODAY,
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            volatility=vol,
        )

    def index_without_dividend(self) -> EquityIndex:
        return self.index.clone(self.interest, None, self.spot)

    def quanto_pricer(self) -> EquityQuantoCashFlowPricer:
        return EquityQuantoCashFlowPricer(
            self.quanto_interest, self.equity_vol, self.fx_vol, self.correlation
        )

    def cash_flow(
        self,
        *,
        index: EquityIndex | None = None,
        base_date: Date = _BASE_DATE,
        fixing_date: Date = _FIXING_DATE,
        payment_date: Date = _PAYMENT_DATE,
        growth_only: bool = True,
        pricer: EquityCashFlowPricer | None = None,
    ) -> EquityCashFlow:
        cf = EquityCashFlow(
            _NOTIONAL,
            index if index is not None else self.index,
            base_date,
            fixing_date,
            payment_date,
            growth_only,
        )
        if pricer is not None:
            cf.set_pricer(pricer)
        return cf


def _check_cash_flow(cf: EquityCashFlow, ref: dict[str, Any]) -> None:
    """Every observable of the flow, not just the amount."""
    assert cf.date().serial_number() == ref["date_serial"]
    assert cf.base_date().serial_number() == ref["base_date_serial"]
    assert cf.fixing_date().serial_number() == ref["fixing_date_serial"]
    assert cf.growth_only() is ref["growth_only"]
    assert (cf.pricer() is not None) is ref["has_pricer"]
    tolerance.exact(cf.notional(), ref["notional"])
    tolerance.exact(cf.base_fixing(), ref["base_fixing"]["value"])
    tolerance.tight(cf.index_fixing(), ref["index_fixing"]["value"])
    tolerance.tight(cf.amount(), ref["amount"]["value"])


# --- EquityCashFlow without a pricer ------------------------------------------


def test_payment_date_is_not_the_fixing_date(cpp: dict[str, Any]) -> None:
    ref = cpp["equity_cash_flow"]["no_pricer_growth_only"]
    assert ref["date_serial"] != ref["fixing_date_serial"]
    cf = _Market().cash_flow()
    assert cf.date().serial_number() == ref["date_serial"]
    assert cf.date() == _PAYMENT_DATE


def test_growth_only_default_is_true() -> None:
    """C++ ``EquityCashFlow`` defaults ``growthOnly`` to true, unlike its
    ``IndexedCashFlow`` base, which defaults it to false."""
    assert _Market().cash_flow().growth_only() is True


def test_no_pricer_growth_only(cpp: dict[str, Any]) -> None:
    market = _Market()
    _check_cash_flow(market.cash_flow(growth_only=True), cpp["equity_cash_flow"]["no_pricer_growth_only"])


def test_no_pricer_total_return(cpp: dict[str, Any]) -> None:
    """``growth_only=False`` (non-default) pays I1/I0 rather than I1/I0 - 1."""
    market = _Market()
    ref = cpp["equity_cash_flow"]["no_pricer_total_return"]
    _check_cash_flow(market.cash_flow(growth_only=False), ref)
    growth = cpp["equity_cash_flow"]["no_pricer_growth_only"]["amount"]["value"]
    tolerance.tight(ref["amount"]["value"] - growth, _NOTIONAL)


# --- EquityCashFlow with the quanto pricer ------------------------------------


def test_quanto_growth_only(cpp: dict[str, Any]) -> None:
    market = _Market()
    ref = cpp["equity_cash_flow"]["quanto_growth_only"]
    pricer = market.quanto_pricer()
    cf = market.cash_flow(growth_only=True, pricer=pricer)
    assert cf.pricer() is pricer
    pricer.initialize(cf)
    tolerance.tight(pricer.price(), ref["price"]["value"])
    _check_cash_flow(cf, ref)


def test_quanto_total_return(cpp: dict[str, Any]) -> None:
    """``growth_only`` must reach the *pricer*, not only the fallback path."""
    market = _Market()
    ref = cpp["equity_cash_flow"]["quanto_total_return"]
    pricer = market.quanto_pricer()
    cf = market.cash_flow(growth_only=False, pricer=pricer)
    pricer.initialize(cf)
    tolerance.tight(pricer.price(), ref["price"]["value"])
    _check_cash_flow(cf, ref)
    growth = cpp["equity_cash_flow"]["quanto_growth_only"]["price"]["value"]
    tolerance.tight(ref["price"]["value"] - growth, 1.0)


def test_quanto_without_dividend_curve(cpp: dict[str, Any]) -> None:
    market = _Market()
    ref = cpp["equity_cash_flow"]["quanto_no_dividend"]
    pricer = market.quanto_pricer()
    cf = market.cash_flow(index=market.index_without_dividend(), pricer=pricer)
    pricer.initialize(cf)
    tolerance.tight(pricer.price(), ref["price"]["value"])
    _check_cash_flow(cf, ref)
    # The empty-dividend branch of the pricer substitutes a flat 0% curve, so
    # the answer must differ from the with-dividend one.
    assert ref["price"]["value"] != cpp["equity_cash_flow"]["quanto_growth_only"]["price"]["value"]


def test_amount_falls_back_when_the_pricer_is_detached(cpp: dict[str, Any]) -> None:
    """``set_pricer(None)`` restores the plain ``IndexedCashFlow`` payoff."""
    market = _Market()
    pricer = market.quanto_pricer()
    cf = market.cash_flow(pricer=pricer)
    tolerance.tight(cf.amount(), cpp["equity_cash_flow"]["quanto_growth_only"]["amount"]["value"])
    cf.set_pricer(None)
    assert cf.pricer() is None
    tolerance.tight(cf.amount(), cpp["equity_cash_flow"]["no_pricer_growth_only"]["amount"]["value"])


# --- the (equity vol, fx vol, correlation) grid --------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "base",
        "zero_correlation",
        "negative_correlation",
        "unit_correlation",
        "swapped_vols",
        "zero_equity_vol",
        "zero_fx_vol",
        "total_return",
        "no_dividend",
    ],
)
def test_quanto_grid(cpp: dict[str, Any], key: str) -> None:
    ref = cpp["quanto_grid"][key]
    market = _Market()
    index = market.index_without_dividend() if key == "no_dividend" else market.index
    pricer = EquityQuantoCashFlowPricer(
        market.quanto_interest,
        market.flat_vol(ref["equity_vol"]),
        market.flat_vol(ref["fx_vol"]),
        SimpleQuote(ref["correlation"]),
    )
    cf = market.cash_flow(index=index, growth_only=ref["growth_only"], pricer=pricer)
    pricer.initialize(cf)
    tolerance.tight(pricer.price(), ref["price"])
    tolerance.tight(cf.amount(), ref["amount"])


def test_correlation_sign_is_not_lost(cpp: dict[str, Any]) -> None:
    """A dropped or sign-flipped correlation collapses these three onto one."""
    grid = cpp["quanto_grid"]
    negative = grid["negative_correlation"]["price"]
    zero = grid["zero_correlation"]["price"]
    positive = grid["base"]["price"]
    unit = grid["unit_correlation"]["price"]
    # Higher correlation drags the quanto forward down, monotonically.
    assert unit < positive < zero < negative
    # ... and rho == 0 collapses onto the unadjusted forward.
    tolerance.tight(zero, cpp["equity_cash_flow"]["no_pricer_growth_only"]["amount"]["value"] / _NOTIONAL)


# --- failure branches ---------------------------------------------------------


def test_raises_when_fixing_date_before_base_date(cpp: dict[str, Any]) -> None:
    assert cpp["pricer_raises"]["fixing_before_base_date"] == {"raises": True}
    market = _Market()
    cf = market.cash_flow(base_date=_FIXING_DATE, fixing_date=_BASE_DATE, pricer=market.quanto_pricer())
    with pytest.raises(LibraryException, match="Fixing date cannot fall before base date"):
        cf.amount()


def test_raises_when_quanto_curve_is_empty(cpp: dict[str, Any]) -> None:
    assert cpp["pricer_raises"]["empty_quanto_curve"] == {"raises": True}
    market = _Market()
    pricer = EquityQuantoCashFlowPricer(None, market.equity_vol, market.fx_vol, market.correlation)
    cf = market.cash_flow(pricer=pricer)
    with pytest.raises(LibraryException, match="Quanto currency term structure handle"):
        cf.amount()


def test_raises_when_equity_vol_is_empty(cpp: dict[str, Any]) -> None:
    assert cpp["pricer_raises"]["empty_equity_vol"] == {"raises": True}
    market = _Market()
    pricer = EquityQuantoCashFlowPricer(market.quanto_interest, None, market.fx_vol, market.correlation)
    cf = market.cash_flow(pricer=pricer)
    with pytest.raises(LibraryException, match="Equity volatility term structure handle"):
        cf.amount()


def test_raises_when_fx_vol_is_empty(cpp: dict[str, Any]) -> None:
    assert cpp["pricer_raises"]["empty_fx_vol"] == {"raises": True}
    market = _Market()
    pricer = EquityQuantoCashFlowPricer(market.quanto_interest, market.equity_vol, None, market.correlation)
    cf = market.cash_flow(pricer=pricer)
    with pytest.raises(LibraryException, match="FX volatility term structure handle"):
        cf.amount()


def test_raises_when_correlation_is_empty(cpp: dict[str, Any]) -> None:
    assert cpp["pricer_raises"]["empty_correlation"] == {"raises": True}
    market = _Market()
    pricer = EquityQuantoCashFlowPricer(market.quanto_interest, market.equity_vol, market.fx_vol, None)
    cf = market.cash_flow(pricer=pricer)
    with pytest.raises(LibraryException, match="Correlation handle cannot be empty"):
        cf.amount()


def test_raises_on_inconsistent_reference_dates(cpp: dict[str, Any]) -> None:
    assert cpp["pricer_raises"]["inconsistent_reference_dates"] == {"raises": True}
    market = _Market()
    off_by_one = FlatForward.from_rate(Date.from_ymd(26, Month.January, 2023), 0.02, market.day_count)
    pricer = EquityQuantoCashFlowPricer(off_by_one, market.equity_vol, market.fx_vol, market.correlation)
    cf = market.cash_flow(pricer=pricer)
    with pytest.raises(LibraryException, match="need to have the same reference date"):
        cf.amount()


# --- observability ------------------------------------------------------------


class _Flag:
    """Minimal Observer that records that it was notified."""

    def __init__(self) -> None:
        self.up = False

    def update(self) -> None:
        self.up = True


def test_market_data_change_propagates_through_the_pricer() -> None:
    """The pricer's four inputs are registered with, not just stored."""
    market = _Market()
    pricer = market.quanto_pricer()
    cf = market.cash_flow(pricer=pricer)
    flag = _Flag()
    cf.register_with(flag)
    assert flag.up is False
    market.correlation.set_value(0.9)
    assert flag.up is True


def test_correlation_change_moves_the_amount(cpp: dict[str, Any]) -> None:
    market = _Market()
    cf = market.cash_flow(pricer=market.quanto_pricer())
    tolerance.tight(cf.amount(), cpp["quanto_grid"]["base"]["amount"])
    market.correlation.set_value(0.0)
    tolerance.tight(cf.amount(), cpp["quanto_grid"]["zero_correlation"]["amount"])


# --- set_coupon_pricer --------------------------------------------------------


def test_set_coupon_pricer(cpp: dict[str, Any]) -> None:
    ref = cpp["set_coupon_pricer"]
    market = _Market()
    cf1 = market.cash_flow()
    second_date = Date(int(cpp["equity_index"]["forecast_with_dividend"]["date_serial_2"]))
    cf2 = market.cash_flow(fixing_date=second_date, payment_date=second_date)
    simple = SimpleCashFlow(ref["simple_amount"], _PAYMENT_DATE)

    tolerance.tight(cf1.amount(), ref["amount_before_1"])
    tolerance.tight(cf2.amount(), ref["amount_before_2"])

    leg: list[CashFlow] = [cf1, cf2, simple]
    pricer = market.quanto_pricer()
    set_coupon_pricer(leg, pricer)

    assert (cf1.pricer() is not None) is ref["cf1_has_pricer"]
    assert (cf2.pricer() is not None) is ref["cf2_has_pricer"]
    assert cf1.pricer() is pricer
    assert cf2.pricer() is pricer
    tolerance.tight(cf1.amount(), ref["amount_after_1"])
    tolerance.tight(cf2.amount(), ref["amount_after_2"])

    # The non-equity flow must be left alone.
    tolerance.exact(simple.amount(), ref["simple_amount"])
    assert simple.date().serial_number() == ref["simple_date_serial"]
