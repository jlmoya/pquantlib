"""DiscountingFxForwardEngine cross-validation against C++ QuantLib v1.43.

Reference: ``migration-harness/references/v143/pe/bondswap.json`` (probe
``migration-harness/cpp/probes/v143_pe_bondswap/probe.cpp``, "PART 4").

Both trade directions are pinned, both settlement-day conventions (including
``settlementDays == 0``, which puts the settlement date exactly on the curve
reference date — the boundary of the two ``QL_REQUIRE``s), the negative-spot
throw, and the seven ``additionalResults`` by name and by count.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.fx_forward import FxForward
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.forward.discounting_fwd_engine import DiscountingFwdEngine
from pquantlib.pricingengines.forward.discounting_fx_forward_engine import (
    DiscountingFxForwardEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

CPP: dict[str, Any] = reference_reader.load("v143/pe/bondswap")

# probe.cpp — `const Date kToday(15, May, 2025);`
TODAY = Date.from_ymd(15, Month.May, 2025)
DC_365 = Actual365Fixed()

_AR_KEYS = {
    "ar_spot_fx": "spotFx",
    "ar_source_df": "sourceCurrencyDiscountFactor",
    "ar_target_df": "targetCurrencyDiscountFactor",
    "ar_source_settlement_df": "sourceCurrencySettlementDiscountFactor",
    "ar_target_settlement_df": "targetCurrencySettlementDiscountFactor",
    "ar_source_pv": "sourceCurrencyPV",
    "ar_target_pv": "targetCurrencyPV",
}


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _curves() -> tuple[FlatForward, FlatForward]:
    """probe.cpp `runFxForwardEngine` — EUR 2%, USD 4.5%, Compounded/Annual."""
    eur = FlatForward.from_rate(TODAY, 0.02, DC_365, Compounding.Compounded, Frequency.Annual)
    usd = FlatForward.from_rate(TODAY, 0.045, DC_365, Compounding.Compounded, Frequency.Annual)
    return eur, usd


def _build(case: str) -> tuple[FxForward, DiscountingFxForwardEngine]:
    inputs = CPP[case]["inputs"]
    eur, usd = _curves()
    fx = FxForward(
        source_nominal=float(inputs["source_nominal"]),
        source_currency=EURCurrency(),
        target_nominal=float(inputs["target_nominal"]),
        target_currency=USDCurrency(),
        maturity_date=Date(int(inputs["maturity_date"])),
        pay_source_currency=bool(inputs["pay_source_currency"]),
        settlement_days=int(inputs["settlement_days"]),
        payment_calendar=TARGET(),
    )
    engine = DiscountingFxForwardEngine(eur, usd, SimpleQuote(float(inputs["spot_fx"])))
    fx.set_pricing_engine(engine)
    return fx, engine


_CASES = sorted(k for k in CPP if k.startswith("fxfwd_"))


@pytest.mark.parametrize("case", _CASES)
def test_fx_forward_prices(case: str) -> None:
    expected = CPP[case]["expected"]
    fx, _ = _build(case)

    if expected.get("throws") is True:
        with pytest.raises(LibraryException):
            fx.npv()
        return

    tolerance.tight(fx.npv(), expected["npv"])
    tolerance.tight(fx.fair_forward_rate(), expected["fair_forward_rate"])
    tolerance.tight(fx.npv_source_currency(), expected["npv_source_currency"])
    tolerance.tight(fx.npv_target_currency(), expected["npv_target_currency"])
    assert fx.settlement_date(TODAY) == Date(int(expected["settlement_date"]))

    additional = fx.additional_results()
    assert len(additional) == int(expected["n_additional_results"])
    for ref_key, cpp_key in _AR_KEYS.items():
        tolerance.tight(float(additional[cpp_key]), expected[ref_key])


def test_direction_flips_the_sign_only() -> None:
    pay = CPP["fxfwd_pay_source"]["expected"]
    receive = CPP["fxfwd_receive_source"]["expected"]
    tolerance.exact(receive["npv"], -pay["npv"])
    # The fair forward rate does not depend on the direction.
    tolerance.exact(receive["fair_forward_rate"], pay["fair_forward_rate"])


def test_target_npv_is_not_the_source_npv_times_spot() -> None:
    """The two NPVs use different settlement discount factors (cpp:106-109)."""
    expected = CPP["fxfwd_pay_source"]["expected"]
    naive = expected["npv_source_currency"] * expected["ar_spot_fx"]
    assert abs(naive - expected["npv_target_currency"]) > 1.0
    # ... and the actual relation is spot * dfTarget(tau) / dfSource(tau).
    tolerance.tight(
        expected["npv_source_currency"]
        / expected["ar_source_settlement_df"]
        * expected["ar_spot_fx"]
        * expected["ar_target_settlement_df"],
        expected["npv_target_currency"],
    )


def test_maturity_before_settlement_is_not_guarded() -> None:
    """C++ has no guard; the settlement-normalised discount factors exceed 1."""
    expected = CPP["fxfwd_maturity_before_settlement"]["expected"]
    assert expected["ar_source_df"] > 1.0
    assert expected["ar_target_df"] > 1.0
    fx, _ = _build("fxfwd_maturity_before_settlement")
    tolerance.tight(fx.npv(), expected["npv"])


def test_curve_reference_date_after_settlement_raises() -> None:
    """C++ QL_REQUIREs both curve reference dates <= settlement date."""
    eur, usd = _curves()
    late = FlatForward.from_rate(
        Date.from_ymd(20, Month.May, 2025), 0.02, DC_365, Compounding.Compounded, Frequency.Annual
    )
    fx = FxForward(
        source_nominal=1_000_000.0,
        source_currency=EURCurrency(),
        target_nominal=1_100_000.0,
        target_currency=USDCurrency(),
        maturity_date=Date.from_ymd(15, Month.May, 2026),
        pay_source_currency=True,
        settlement_days=0,
        payment_calendar=TARGET(),
    )
    fx.set_pricing_engine(DiscountingFxForwardEngine(late, usd, SimpleQuote(1.08)))
    with pytest.raises(LibraryException, match="source currency discount curve reference date"):
        fx.npv()

    fx2 = FxForward(
        source_nominal=1_000_000.0,
        source_currency=EURCurrency(),
        target_nominal=1_100_000.0,
        target_currency=USDCurrency(),
        maturity_date=Date.from_ymd(15, Month.May, 2026),
        pay_source_currency=True,
        settlement_days=0,
        payment_calendar=TARGET(),
    )
    fx2.set_pricing_engine(DiscountingFxForwardEngine(eur, late, SimpleQuote(1.08)))
    with pytest.raises(LibraryException, match="target currency discount curve reference date"):
        fx2.npv()


def test_legacy_alias_is_the_same_class() -> None:
    """``DiscountingFwdEngine`` is the pre-rename name, kept as a plain alias."""
    assert DiscountingFwdEngine is DiscountingFxForwardEngine
