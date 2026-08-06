"""Cross-validate the Italian government-bond family against C++ v1.43.

Probe: ``v143/inst/bondsbtp`` (see
``migration-harness/cpp/probes/v143_inst_bondsbtp/probe.cpp`` for what is
pinned and why).

Covers :class:`CCTEU`, :class:`BTP`, :class:`RendistatoBasket`,
:class:`RendistatoCalculator` and the two Rendistato-derived quotes.

These classes are pure wiring over ``FloatingRateBond`` / ``FixedRateBond`` /
``MakeVanillaSwap``, so every test compares the FULL cashflow listing, not just
a headline NPV — an NPV can match while two convention errors cancel. Every
optional constructor argument gets a test at a NON-default value asserting an
observable consequence.

Tolerance tiers:

* TIGHT for cashflow amounts / accruals / rates / NPV / prices / swap fair
  rates — short arithmetic chains over identical inputs.
* LOOSE for anything downstream of a yield solve: C++ ``BondFunctions::yield``
  uses ``NewtonSafe`` while :meth:`Bond.yield_from_price` uses ``Brent``, so the
  two roots agree only to the requested solver accuracy (1e-10 here). Observed
  worst case is 7.7e-12 absolute.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, NamedTuple, cast

import pytest

from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as AAConvention
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.bond import Bond
from pquantlib.instruments.bonds.btp import (
    BTP,
    CCTEU,
    RendistatoBasket,
    RendistatoCalculator,
    RendistatoEquivalentSwapLengthQuote,
    RendistatoEquivalentSwapSpreadQuote,
)
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.instruments.bonds.floating_rate_bond import FloatingRateBond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

# --- scenario constants, mirroring the probe -----------------------------

CCTEU_MATURITY = Date.from_ymd(15, Month.October, 2031)
CCTEU_SPREAD = 0.0075
CCTEU_START = Date.from_ymd(15, Month.April, 2024)

BTP_MATURITY = Date.from_ymd(1, Month.August, 2033)
BTP_RATE = 0.0450
BTP_START = Date.from_ymd(1, Month.August, 2023)

BTP_RED_MATURITY = Date.from_ymd(15, Month.May, 2037)
BTP_RED_RATE = 0.0300
BTP_REDEMPTION = 99.999
BTP_RED_START = Date.from_ymd(15, Month.May, 2017)

OUTSTANDINGS = [25.0e9, 30.0e9, 20.0e9, 15.0e9]
CLEAN_PRICES = [101.25, 103.10, 105.40, 92.75]


class Market(NamedTuple):
    """The forecast / discount curves and the seeded Euribor6M index."""

    forecast: InterpolatedZeroCurve
    discount: InterpolatedZeroCurve
    euribor: Euribor
    today: Date


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/bondsbtp")


@pytest.fixture(scope="module")
def market(cpp: dict[str, Any]) -> Iterator[Market]:
    """Replay the probe's evaluation date, curves and fixing history."""
    mk = cpp["market"]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    today = Date(int(mk["evaluation_date"]))
    settings.evaluation_date = today

    dates = [Date(int(s)) for s in mk["curve_dates"]]
    dc = Actual365Fixed()
    forecast = InterpolatedZeroCurve(dates, list(mk["forecast_rates"]), dc)
    discount = InterpolatedZeroCurve(dates, list(mk["discount_rates"]), dc)

    index = Euribor.six_months(cast("YieldTermStructureProtocol", forecast))
    index.clear_fixings()
    for f in mk["euribor6m_fixings"]:
        index.add_fixing(Date(int(f["date"])), float(f["value"]))

    yield Market(forecast, discount, index, today)

    index.clear_fixings()
    settings.evaluation_date = saved


# --- helpers -------------------------------------------------------------


def _check_leg(bond: Bond, ref_flows: list[dict[str, Any]]) -> None:
    """Compare the bond's complete cashflow listing against the reference."""
    flows = bond.cashflows()
    assert len(flows) == len(ref_flows)
    for cf, r in zip(flows, ref_flows, strict=True):
        assert cf.date().serial_number() == r["date"]
        tolerance.tight(cf.amount(), r["amount"])
        assert cf.ex_coupon_date().serial_number() == r["ex_coupon"]
        assert isinstance(cf, Coupon) == r["is_coupon"]
        if isinstance(cf, Coupon):
            tolerance.tight(cf.nominal(), r["nominal"])
            assert cf.accrual_start_date().serial_number() == r["accrual_start"]
            assert cf.accrual_end_date().serial_number() == r["accrual_end"]
            tolerance.tight(cf.accrual_period(), r["accrual_period"])
            tolerance.tight(cf.rate(), r["rate"])


def _check_core(bond: Bond, ref: dict[str, Any]) -> None:
    assert bond.settlement_date().serial_number() == ref["settlement_date"]
    assert bond.issue_date().serial_number() == ref["issue_date"]
    assert bond.start_date().serial_number() == ref["start_date"]
    assert bond.maturity_date().serial_number() == ref["maturity_date"]
    assert len(bond.cashflows()) == ref["n_cashflows"]
    tolerance.tight(bond.notional(bond.settlement_date()), ref["notional"])
    tolerance.tight(bond.accrued_amount(), ref["accrued"])
    tolerance.tight(bond.npv(), ref["npv"])
    tolerance.tight(bond.clean_price(), ref["clean_price"])
    tolerance.tight(bond.dirty_price(), ref["dirty_price"])
    tolerance.tight(bond.settlement_value(), ref["settlement_value"])
    _check_leg(bond, ref["cashflows"])


def _priced[B: Bond](bond: B, market: Market) -> B:
    bond.set_pricing_engine(DiscountingBondEngine(market.discount))
    return bond


def _ccteu(market: Market, **kwargs: Any) -> CCTEU:
    args: dict[str, Any] = {
        "maturity_date": CCTEU_MATURITY,
        "spread": CCTEU_SPREAD,
        "fwd_curve": cast("YieldTermStructureProtocol", market.forecast),
        "start_date": CCTEU_START,
        "issue_date": CCTEU_START,
    }
    args.update(kwargs)
    return CCTEU(**args)


def _btp(market: Market, **kwargs: Any) -> BTP:
    del market
    args: dict[str, Any] = {
        "maturity_date": BTP_MATURITY,
        "fixed_rate": BTP_RATE,
        "start_date": BTP_START,
        "issue_date": BTP_START,
    }
    args.update(kwargs)
    return BTP(**args)


def _basket_bonds() -> list[BTP]:
    return [
        BTP(
            Date.from_ymd(1, Month.August, 2029),
            0.0350,
            start_date=Date.from_ymd(1, Month.August, 2019),
            issue_date=Date.from_ymd(1, Month.August, 2019),
        ),
        BTP(
            Date.from_ymd(1, Month.February, 2032),
            0.0400,
            start_date=Date.from_ymd(1, Month.February, 2022),
            issue_date=Date.from_ymd(1, Month.February, 2022),
        ),
        BTP(
            Date.from_ymd(1, Month.August, 2033),
            0.0450,
            start_date=Date.from_ymd(1, Month.August, 2023),
            issue_date=Date.from_ymd(1, Month.August, 2023),
        ),
        BTP(
            BTP_RED_MATURITY,
            BTP_RED_RATE,
            BTP_REDEMPTION,
            BTP_RED_START,
            BTP_RED_START,
        ),
    ]


def _basket() -> tuple[RendistatoBasket, list[SimpleQuote]]:
    quotes = [SimpleQuote(p) for p in CLEAN_PRICES]
    return RendistatoBasket(_basket_bonds(), OUTSTANDINGS, quotes), quotes


# =========================================================================
# CCTEU
# =========================================================================


def test_ccteu_matches_cpp(market: Market, cpp: dict[str, Any]) -> None:
    _check_core(_priced(_ccteu(market), market), cpp["ccteu"])


def test_ccteu_yield(market: Market, cpp: dict[str, Any]) -> None:
    bond = _priced(_ccteu(market), market)
    tolerance.loose(
        bond.yield_rate(ActualActual(AAConvention.ISMA), Compounding.Compounded, Frequency.Annual),
        cpp["ccteu"]["yield"],
    )


def test_ccteu_accrued_amount_is_rounded_to_five_decimals(market: Market, cpp: dict[str, Any]) -> None:
    """``CCTEU::accruedAmount`` wraps the base in ``ClosestRounding(5)``."""
    bond = _priced(_ccteu(market), market)
    ref = cpp["ccteu"]
    raw = FloatingRateBond.accrued_amount(bond, bond.settlement_date())
    tolerance.tight(raw, ref["accrued_unrounded"])
    tolerance.exact(bond.accrued_amount(), ref["accrued"])
    # The scenario is chosen so the rounding is NOT a no-op: an unrounded
    # accrual would fail the assertion above.
    assert raw != ref["accrued"]


def test_ccteu_spread_is_not_dropped(market: Market, cpp: dict[str, Any]) -> None:
    """Non-default ``spread`` — every coupon rate, the accrual and the NPV move."""
    bond = _priced(_ccteu(market, spread=0.0250), market)
    _check_core(bond, cpp["ccteu_spread"])
    base = _priced(_ccteu(market), market)
    assert bond.npv() != base.npv()


def test_ccteu_start_date_is_not_dropped(market: Market, cpp: dict[str, Any]) -> None:
    """Non-default ``start_date`` — the schedule (hence the leg) is shorter."""
    later = Date.from_ymd(15, Month.October, 2025)
    bond = _priced(_ccteu(market, start_date=later, issue_date=later), market)
    _check_core(bond, cpp["ccteu_start"])
    assert len(bond.cashflows()) < cpp["ccteu"]["n_cashflows"]


def test_ccteu_default_start_date_anchors_on_the_evaluation_date(
    market: Market,
) -> None:
    """Omitting ``start_date`` must NOT silently reuse some other anchor."""
    bond = _ccteu(market, start_date=None, issue_date=None)
    assert bond.start_date().serial_number() == market.today.serial_number()


def test_ccteu_issue_date_is_not_dropped(market: Market, cpp: dict[str, Any]) -> None:
    """Non-default ``issue_date`` in the future moves ``settlement_date``."""
    future = Date.from_ymd(15, Month.April, 2027)
    bond = _priced(
        _ccteu(
            market,
            maturity_date=Date.from_ymd(15, Month.October, 2032),
            start_date=future,
            issue_date=future,
        ),
        market,
    )
    _check_core(bond, cpp["ccteu_future_issue"])
    assert bond.settlement_date().serial_number() == future.serial_number()


def test_ccteu_without_forecast_curve_cannot_forecast(market: Market) -> None:
    """``fwd_curve`` defaults to None — the index then has no forecast curve."""
    bond = _ccteu(market, fwd_curve=None)
    with pytest.raises(LibraryException, match="null term structure"):
        bond.cashflows()[-2].amount()


# =========================================================================
# BTP
# =========================================================================


def test_btp_matches_cpp(market: Market, cpp: dict[str, Any]) -> None:
    _check_core(_priced(_btp(market), market), cpp["btp"])


def test_btp_conventions(market: Market, cpp: dict[str, Any]) -> None:
    bond = _btp(market)
    assert int(bond.frequency()) == cpp["btp"]["frequency"]
    assert bond.day_counter().name() == cpp["btp"]["day_counter"]


def test_btp_accrued_amount_is_rounded_to_five_decimals(market: Market, cpp: dict[str, Any]) -> None:
    bond = _priced(_btp(market), market)
    ref = cpp["btp"]
    raw = FixedRateBond.accrued_amount(bond, bond.settlement_date())
    tolerance.tight(raw, ref["accrued_unrounded"])
    tolerance.exact(bond.accrued_amount(), ref["accrued"])
    assert raw != ref["accrued"]


def test_btp_engine_yield(market: Market, cpp: dict[str, Any]) -> None:
    """The inherited ``Bond::yield`` overload (C++ reaches it by qualification)."""
    bond = _priced(_btp(market), market)
    tolerance.loose(
        bond.yield_rate(ActualActual(AAConvention.ISMA), Compounding.Compounded, Frequency.Annual),
        cpp["btp"]["yield_engine"],
    )


def test_btp_yield_from_clean_price(market: Market, cpp: dict[str, Any]) -> None:
    bond = _btp(market)
    tolerance.loose(bond.yield_(101.75), cpp["btp"]["btp_yield_default"])
    tolerance.loose(bond.yield_(97.25), cpp["btp"]["btp_yield_other_price"])


def test_btp_yield_settlement_date_is_not_dropped(market: Market, cpp: dict[str, Any]) -> None:
    """Non-default ``settlement_date`` / ``accuracy`` / ``max_evaluations``."""
    bond = _btp(market)
    at_date = bond.yield_(
        101.75,
        Date.from_ymd(3, Month.August, 2026),
        1.0e-12,  # non-default accuracy
        200,  # non-default max_evaluations
    )
    tolerance.loose(at_date, cpp["btp"]["btp_yield_at_date"])
    # A different settlement date really is a different yield.
    assert abs(at_date - cpp["btp"]["btp_yield_default"]) > 1e-6


def test_btp_yield_accuracy_must_be_positive(market: Market) -> None:
    bond = _btp(market)
    with pytest.raises(LibraryException, match="accuracy"):
        bond.yield_(101.75, None, 0.0)


def test_btp_yield_max_evaluations_is_not_dropped(market: Market) -> None:
    bond = _btp(market)
    with pytest.raises(LibraryException, match="1 function evaluations"):
        bond.yield_(101.75, None, 1.0e-8, 1)


def test_btp_redemption_is_not_dropped(market: Market, cpp: dict[str, Any]) -> None:
    """Non-default ``redemption`` — the legacy 99.999 BTP."""
    bond = _priced(
        BTP(
            BTP_RED_MATURITY,
            BTP_RED_RATE,
            BTP_REDEMPTION,
            BTP_RED_START,
            BTP_RED_START,
        ),
        market,
    )
    _check_core(bond, cpp["btp_redemption"])
    tolerance.exact(bond.redemption().amount(), BTP_REDEMPTION)
    tolerance.loose(bond.yield_(93.50), cpp["btp_redemption"]["btp_yield_default"])


# =========================================================================
# RendistatoBasket
# =========================================================================


def test_rendistato_basket(cpp: dict[str, Any], market: Market) -> None:
    del market  # evaluation date must be pinned before the BTPs are built
    basket, _ = _basket()
    ref = cpp["basket"]
    assert basket.size() == ref["size"]
    assert len(basket.btps()) == ref["n_btps"]
    assert len(basket.clean_price_quotes()) == ref["n_quotes"]
    tolerance.tight(basket.outstanding(), ref["outstanding"])
    for got, want in zip(basket.weights(), ref["weights"], strict=True):
        tolerance.tight(got, want)
    for got, want in zip(basket.outstandings(), ref["outstandings"], strict=True):
        tolerance.tight(got, want)


def test_rendistato_basket_reweighted(cpp: dict[str, Any], market: Market) -> None:
    del market
    ref = cpp["basket_reweighted"]
    quotes: list[Quote] = [SimpleQuote(p) for p in CLEAN_PRICES]
    basket = RendistatoBasket(_basket_bonds(), [10.0e9, 10.0e9, 40.0e9, 40.0e9], quotes)
    assert basket.size() == ref["size"]
    tolerance.tight(basket.outstanding(), ref["outstanding"])
    for got, want in zip(basket.weights(), ref["weights"], strict=True):
        tolerance.tight(got, want)


def test_rendistato_basket_rejects_empty(cpp: dict[str, Any], market: Market) -> None:
    del market
    assert cpp["basket_raises"]["empty_basket"]["raises"]
    with pytest.raises(LibraryException, match="empty RendistatoCalculator Basket"):
        RendistatoBasket([], [], [])


def test_rendistato_basket_rejects_outstandings_mismatch(cpp: dict[str, Any], market: Market) -> None:
    del market
    assert cpp["basket_raises"]["outstandings_size_mismatch"]["raises"]
    quotes: list[Quote] = [SimpleQuote(p) for p in CLEAN_PRICES]
    with pytest.raises(LibraryException, match="number of outstandings"):
        RendistatoBasket(_basket_bonds(), OUTSTANDINGS[:3], quotes)


def test_rendistato_basket_rejects_quotes_mismatch(cpp: dict[str, Any], market: Market) -> None:
    del market
    assert cpp["basket_raises"]["quotes_size_mismatch"]["raises"]
    quotes: list[Quote] = [SimpleQuote(p) for p in CLEAN_PRICES[:3]]
    with pytest.raises(LibraryException, match="clean prices quotes"):
        RendistatoBasket(_basket_bonds(), OUTSTANDINGS, quotes)


def test_rendistato_basket_rejects_negative_outstanding(cpp: dict[str, Any], market: Market) -> None:
    del market
    assert cpp["basket_raises"]["negative_outstanding"]["raises"]
    quotes: list[Quote] = [SimpleQuote(p) for p in CLEAN_PRICES]
    with pytest.raises(LibraryException, match="negative outstanding"):
        RendistatoBasket(_basket_bonds(), [25.0e9, -1.0, 20.0e9, 15.0e9], quotes)


# =========================================================================
# RendistatoCalculator
# =========================================================================


def _check_vector(got: list[float], want: list[float | None], loose: bool) -> None:
    """Compare a calculator vector; ``None`` in the reference is C++ Null<Real>."""
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        if w is None:
            assert math.isnan(g), f"expected the Null<Real> tail, got {g!r}"
        elif loose:
            tolerance.loose(g, w)
        else:
            tolerance.tight(g, w)


def test_rendistato_calculator(cpp: dict[str, Any], market: Market) -> None:
    basket, _ = _basket()
    calc = RendistatoCalculator(basket, market.euribor, market.discount)
    ref = cpp["rendistato"]

    tolerance.loose(calc.yield_(), ref["yield"])
    tolerance.loose(calc.duration(), ref["duration"])
    tolerance.loose(calc.equivalent_swap_length(), ref["equivalent_swap_length"])
    tolerance.loose(calc.equivalent_swap_spread(), ref["equivalent_swap_spread"])
    tolerance.tight(calc.equivalent_swap_rate(), ref["equivalent_swap_rate"])
    tolerance.loose(calc.equivalent_swap_yield(), ref["equivalent_swap_yield"])
    tolerance.loose(calc.equivalent_swap_duration(), ref["equivalent_swap_duration"])

    _check_vector(calc.yields(), ref["yields"], loose=True)
    _check_vector(calc.durations(), ref["durations"], loose=True)
    _check_vector(calc.swap_lengths(), ref["swap_lengths"], loose=False)
    _check_vector(calc.swap_rates(), ref["swap_rates"], loose=False)
    _check_vector(calc.swap_yields(), ref["swap_yields"], loose=True)
    _check_vector(calc.swap_durations(), ref["swap_durations"], loose=True)


def test_rendistato_equivalent_swap(cpp: dict[str, Any], market: Market) -> None:
    basket, _ = _basket()
    calc = RendistatoCalculator(basket, market.euribor, market.discount)
    ref = cpp["rendistato"]
    swap = calc.equivalent_swap()
    tolerance.exact(swap.fixed_rate(), ref["equivalent_swap_fixed_rate"])
    tolerance.tight(swap.fair_rate(), ref["equivalent_swap_fair_rate"])
    assert len(swap.fixed_schedule()) == ref["equivalent_swap_fixed_dates"]


def test_rendistato_calculator_recalculates_on_quote_change(cpp: dict[str, Any], market: Market) -> None:
    """Basket observes the quotes, the LazyObject observes the basket."""
    basket, quotes = _basket()
    calc = RendistatoCalculator(basket, market.euribor, market.discount)
    before = calc.yield_()
    tolerance.loose(before, cpp["rendistato"]["yield"])

    quotes[0].set_value(95.00)
    quotes[3].set_value(88.00)

    ref = cpp["rendistato_after_bump"]
    tolerance.loose(calc.yield_(), ref["yield"])
    tolerance.loose(calc.duration(), ref["duration"])
    _check_vector(calc.yields(), ref["yields"], loose=True)
    assert calc.yield_() != before


def test_rendistato_calculator_uses_the_discount_curve(market: Market) -> None:
    """``discount_curve`` must reach ``MakeVanillaSwap`` — not be dropped."""
    basket, _ = _basket()
    on_discount = RendistatoCalculator(basket, market.euribor, market.discount)
    basket2, _ = _basket()
    on_forecast = RendistatoCalculator(basket2, market.euribor, market.forecast)
    assert on_discount.swap_rates()[0] != on_forecast.swap_rates()[0]


def test_rendistato_calculator_uses_the_euribor_index(market: Market) -> None:
    """``euribor_index`` drives the swaps' floating leg — not be dropped."""
    basket, _ = _basket()
    on_6m = RendistatoCalculator(basket, market.euribor, market.discount)
    basket2, _ = _basket()
    euribor3m = Euribor.three_months(cast("YieldTermStructureProtocol", market.forecast))
    on_3m = RendistatoCalculator(basket2, euribor3m, market.discount)
    assert on_6m.swap_rates()[0] != on_3m.swap_rates()[0]


# =========================================================================
# The two Quote adapters
# =========================================================================


def test_rendistato_quotes(cpp: dict[str, Any], market: Market) -> None:
    basket, quotes = _basket()
    calc = RendistatoCalculator(basket, market.euribor, market.discount)
    length_quote = RendistatoEquivalentSwapLengthQuote(calc)
    spread_quote = RendistatoEquivalentSwapSpreadQuote(calc)

    before = cpp["quotes_before"]
    assert length_quote.is_valid() is before["length_is_valid"]
    assert spread_quote.is_valid() is before["spread_is_valid"]
    tolerance.loose(length_quote.value(), before["length_value"])
    tolerance.loose(spread_quote.value(), before["spread_value"])

    quotes[0].set_value(95.00)
    quotes[3].set_value(88.00)

    after = cpp["quotes_after"]
    tolerance.loose(length_quote.value(), after["length_value"])
    tolerance.loose(spread_quote.value(), after["spread_value"])
    assert spread_quote.value() != before["spread_value"]


def test_rendistato_quotes_are_invalid_when_the_calculation_throws(
    cpp: dict[str, Any], market: Market
) -> None:
    """An invalid clean-price quote makes ``performCalculations`` raise."""
    ref = cpp["quotes_invalid"]
    broken: list[Quote] = [SimpleQuote(), *(SimpleQuote(p) for p in CLEAN_PRICES[1:])]
    basket = RendistatoBasket(_basket_bonds(), OUTSTANDINGS, broken)
    calc = RendistatoCalculator(basket, market.euribor, market.discount)
    assert RendistatoEquivalentSwapLengthQuote(calc).is_valid() is ref["length_is_valid"]
    assert RendistatoEquivalentSwapSpreadQuote(calc).is_valid() is ref["spread_is_valid"]
