"""Tests for FxForward + DiscountingFwdEngine.

Reference values come from ``cluster/l3e.json`` emitted by
``migration-harness/cpp/probes/cluster_l3e/probe.cpp``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.fx_forward import FxForward
from pquantlib.pricingengines.forward.discounting_fwd_engine import (
    DiscountingFwdEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from tests.indexes._mock_curves import FlatForwardMock


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l3e")


# --- FxForward construction validation -------------------------------------


def test_fx_forward_empty_source_currency_rejected() -> None:
    with pytest.raises(LibraryException, match="source currency"):
        FxForward(
            source_nominal=1.0,
            source_currency=Currency(),  # empty
            target_nominal=1.0,
            target_currency=USDCurrency(),
            maturity_date=Date.from_ymd(17, Month.January, 2025),
            pay_source_currency=True,
        )


def test_fx_forward_same_currency_rejected() -> None:
    with pytest.raises(LibraryException, match="must be different"):
        FxForward(
            source_nominal=1.0,
            source_currency=USDCurrency(),
            target_nominal=1.0,
            target_currency=USDCurrency(),
            maturity_date=Date.from_ymd(17, Month.January, 2025),
            pay_source_currency=True,
        )


def test_fx_forward_negative_nominal_rejected() -> None:
    with pytest.raises(LibraryException, match="source nominal"):
        FxForward(
            source_nominal=-1.0,
            source_currency=EURCurrency(),
            target_nominal=1.0,
            target_currency=USDCurrency(),
            maturity_date=Date.from_ymd(17, Month.January, 2025),
            pay_source_currency=True,
        )


def test_fx_forward_from_forward_rate_constructor() -> None:
    fx = FxForward.from_forward_rate(
        source_nominal=1_000_000.0,
        source_currency=EURCurrency(),
        target_currency=USDCurrency(),
        forward_rate=1.10,
        maturity_date=Date.from_ymd(17, Month.January, 2025),
        pay_source_currency=True,
    )
    assert fx.source_nominal() == 1_000_000.0
    assert fx.target_nominal() == 1_100_000.0
    assert fx.forward_rate() == 1.10


def test_fx_forward_inspectors() -> None:
    maturity = Date.from_ymd(17, Month.January, 2025)
    fx = FxForward(
        source_nominal=1_000_000.0,
        source_currency=EURCurrency(),
        target_nominal=1_100_000.0,
        target_currency=USDCurrency(),
        maturity_date=maturity,
        pay_source_currency=True,
    )
    assert fx.source_currency().code == "EUR"
    assert fx.target_currency().code == "USD"
    assert fx.maturity_date() == maturity
    assert fx.pay_source_currency() is True
    assert fx.settlement_days() == 2


# --- FxForward NPV via DiscountingFwdEngine --------------------------------


def test_fx_forward_npv_and_fair_rate(ref: dict[str, Any]) -> None:
    """1M EUR / 1.10M USD, paySource=true, 1Y maturity.

    EUR FlatForward(3%) Actual360 continuous; USD FlatForward(5%) ditto;
    spot FX 1.10 (USD per EUR).
    """
    eval_date = Date.from_ymd(17, Month.January, 2024)
    maturity = Date(eval_date.serial + 365)
    # Sanity-check: the probe used eval+365 too.
    assert maturity.serial_number() == ref["fx_forward"]["maturity_serial"]

    ts_eur = FlatForwardMock(eval_date, 0.03, Actual360())
    ts_usd = FlatForwardMock(eval_date, 0.05, Actual360())
    spot = SimpleQuote(1.10)

    fx = FxForward(
        source_nominal=1_000_000.0,
        source_currency=EURCurrency(),
        target_nominal=1_100_000.0,
        target_currency=USDCurrency(),
        maturity_date=maturity,
        pay_source_currency=True,
        settlement_days=2,
        payment_calendar=TARGET(),
    )
    engine = DiscountingFwdEngine(ts_eur, ts_usd, spot)
    fx.set_pricing_engine(engine)

    expected = ref["fx_forward"]
    tight(fx.npv(), float(expected["npv"]))
    tight(fx.fair_forward_rate(), float(expected["fair_forward_rate"]))
    tight(fx.npv_source_currency(), float(expected["npv_source_ccy"]))
    tight(fx.npv_target_currency(), float(expected["npv_target_ccy"]))


def test_fx_forward_pay_target_currency_flips_sign() -> None:
    """When pay_source_currency=False, NPV flips sign."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    maturity = Date(eval_date.serial + 365)

    ts_eur = FlatForwardMock(eval_date, 0.03, Actual360())
    ts_usd = FlatForwardMock(eval_date, 0.05, Actual360())
    spot = SimpleQuote(1.10)

    fx_pay_src = FxForward(
        source_nominal=1_000_000.0,
        source_currency=EURCurrency(),
        target_nominal=1_100_000.0,
        target_currency=USDCurrency(),
        maturity_date=maturity,
        pay_source_currency=True,
        settlement_days=2,
        payment_calendar=TARGET(),
    )
    fx_pay_tgt = FxForward(
        source_nominal=1_000_000.0,
        source_currency=EURCurrency(),
        target_nominal=1_100_000.0,
        target_currency=USDCurrency(),
        maturity_date=maturity,
        pay_source_currency=False,
        settlement_days=2,
        payment_calendar=TARGET(),
    )
    engine1 = DiscountingFwdEngine(ts_eur, ts_usd, spot)
    engine2 = DiscountingFwdEngine(ts_eur, ts_usd, SimpleQuote(1.10))
    fx_pay_src.set_pricing_engine(engine1)
    fx_pay_tgt.set_pricing_engine(engine2)
    # The two NPVs must be negatives of one another.
    tight(fx_pay_src.npv(), -fx_pay_tgt.npv())
