"""Black76Spec / BachelierSpec + Black-style swaption engine vs C++ v1.43.

Reference: ``migration-harness/references/v143/pe/swaption.json``, produced by
``migration-harness/cpp/probes/v143_pe_swaption/probe.cpp``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import BermudanExercise, EuropeanExercise
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    Swaption,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.pricingengines.swaption.black_swaption_engine import (
    BachelierSpec,
    BachelierSwaptionEngine,
    Black76Spec,
    BlackSwaptionEngine,
    CashAnnuityModel,
)
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    ConstantSwaptionVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit

from . import _v143_swaption_market as mkt

_SETTLEMENT_TYPES = {
    "Physical": SettlementType.Physical,
    "Cash": SettlementType.Cash,
}
_SETTLEMENT_METHODS = {
    "PhysicalOTC": SettlementMethod.PhysicalOTC,
    "PhysicalCleared": SettlementMethod.PhysicalCleared,
    "CollateralizedCashPrice": SettlementMethod.CollateralizedCashPrice,
    "ParYieldCurve": SettlementMethod.ParYieldCurve,
}
_SWAP_TYPES = {"Payer": SwapType.Payer, "Receiver": SwapType.Receiver}
_OPTION_TYPES = {"Call": OptionType.Call, "Put": OptionType.Put}
_VOL_TYPES = {
    "ShiftedLognormal": VolatilityType.ShiftedLognormal,
    "Normal": VolatilityType.Normal,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/swaption")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:293 — Settings::instance().evaluationDate() = Date(3, March, 2025)
    settings.evaluation_date = mkt.TODAY
    yield
    settings.evaluation_date = saved


# --------------------------------------------------------------------------
# Market reconstruction
# --------------------------------------------------------------------------


def test_market_setup_matches_probe(cpp: dict[str, Any]) -> None:
    """The dates and swap diagnostics the engine is built on."""
    case = cpp["swap_derived_inputs"]
    inputs, expected = case["inputs"], case["expected"]
    assert str(mkt.TODAY) == inputs["today"]
    assert str(mkt.exercise_date()) == inputs["exercise_date"]
    assert str(mkt.swap_start()) == inputs["swap_start"]
    assert str(mkt.swap_end()) == inputs["swap_end"]
    assert [str(d) for d in mkt.fixed_schedule().dates] == expected["fixed_dates"]
    assert [str(d) for d in mkt.float_schedule().dates] == expected["float_dates"]

    disc = mkt.discount_curve()
    swap = mkt.make_swap(mkt.forwarding_curve(), SwapType.Payer, 0.03)
    swap.set_pricing_engine(DiscountingSwapEngine(disc))
    tolerance.tight(swap.fair_rate(), expected["fair_rate"])
    tolerance.tight(swap.fixed_leg_bps(), expected["fixed_leg_bps"])
    tolerance.tight(swap.floating_leg_bps(), expected["floating_leg_bps"])
    tolerance.tight(swap.npv(), expected["npv"])
    tolerance.tight(
        disc.discount(mkt.exercise_date()), expected["discount_at_exercise"]
    )
    assert str(swap.valuation_date()) == expected["valuation_date"]


# --------------------------------------------------------------------------
# Black76Spec / BachelierSpec, standalone
# --------------------------------------------------------------------------


def _spec_case_names(cpp: dict[str, Any]) -> list[str]:
    return sorted(k for k in cpp if k.startswith(("spec_black76_", "spec_bachelier_")))


def test_spec_cases_are_present(cpp: dict[str, Any]) -> None:
    """Guard against a silently empty parametrisation."""
    assert len(_spec_case_names(cpp)) == 44


@pytest.mark.parametrize(
    "name",
    sorted(
        k
        for k in reference_reader.load("v143/pe/swaption")
        if k.startswith(("spec_black76_", "spec_bachelier_"))
    ),
)
def test_spec_value_vega_delta(cpp: dict[str, Any], name: str) -> None:
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    spec = Black76Spec() if inputs["spec"] == "Black76Spec" else BachelierSpec()

    assert spec.type.name == expected["volatility_type"]

    option_type = _OPTION_TYPES[inputs["option_type"]]
    strike = inputs["strike"]
    fwd = inputs["atm_forward"]
    sd = inputs["std_dev"]
    annuity = inputs["annuity"]
    disp = inputs["displacement"]
    t = inputs["exercise_time"]

    if expected["throws"]:
        with pytest.raises(LibraryException):
            spec.value(option_type, strike, fwd, sd, annuity, disp)
        return

    # TIGHT: closed-form Black / Bachelier evaluations of the same
    # expression tree; nothing iterative in the path.
    tolerance.tight(
        spec.value(option_type, strike, fwd, sd, annuity, disp), expected["value"]
    )
    tolerance.tight(
        spec.vega(strike, fwd, sd, t, annuity, disp), expected["vega"]
    )
    tolerance.tight(
        spec.delta(option_type, strike, fwd, sd, annuity, disp), expected["delta"]
    )


def test_bachelier_spec_ignores_displacement(cpp: dict[str, Any]) -> None:
    """C++ leaves the trailing displacement parameter UNNAMED in BachelierSpec.

    blackswaptionengine.hpp:109, :117, :121 — the argument is accepted and
    discarded. A port that forwards it into ``bachelierBlackFormula`` would
    make these two cases differ.
    """
    disp0 = cpp["spec_bachelier_call_itm"]["expected"]
    disp01 = cpp["spec_bachelier_call_disp01"]["expected"]
    assert disp0["value"] == disp01["value"]
    spec = BachelierSpec()
    inputs = cpp["spec_bachelier_call_disp01"]["inputs"]
    with_disp = spec.value(
        OptionType.Call,
        inputs["strike"],
        inputs["atm_forward"],
        inputs["std_dev"],
        inputs["annuity"],
        inputs["displacement"],
    )
    without_disp = spec.value(
        OptionType.Call,
        inputs["strike"],
        inputs["atm_forward"],
        inputs["std_dev"],
        inputs["annuity"],
        0.0,
    )
    assert with_disp == without_disp
    tolerance.tight(with_disp, disp01["value"])


# --------------------------------------------------------------------------
# BlackSwaptionEngine / BachelierSwaptionEngine
# --------------------------------------------------------------------------

_ENGINE_CASES = sorted(
    k
    for k, v in reference_reader.load("v143/pe/swaption").items()
    if v["inputs"].get("engine")
    in ("BlackSwaptionEngine", "BachelierSwaptionEngine")
    and "ctor" not in v["inputs"]
    and "exercise" not in v["inputs"]
    and "exercise_date_override" not in v["inputs"]
)


def _build_swaption(inputs: dict[str, Any]) -> Swaption:
    swap = mkt.make_swap(
        mkt.forwarding_curve(),
        _SWAP_TYPES[inputs["swap_type"]],
        inputs["fixed_rate"],
        inputs["float_spread"],
    )
    return Swaption(
        swap,
        EuropeanExercise(mkt.exercise_date()),
        _SETTLEMENT_TYPES[inputs["settlement_type"]],
        _SETTLEMENT_METHODS[inputs["settlement_method"]],
    )


_EXTRA_KEYS = (
    ("spreadCorrection", "spread_correction"),
    ("strike", "strike"),
    ("atmForward", "atm_forward"),
    ("annuity", "annuity"),
    ("swapLength", "swap_length"),
    ("stdDev", "std_dev"),
    ("vega", "vega"),
    ("delta", "delta"),
    ("timeToExpiry", "time_to_expiry"),
    ("impliedVolatility", "implied_volatility"),
    ("forwardPrice", "forward_price"),
)


def test_engine_cases_are_present() -> None:
    assert len(_ENGINE_CASES) == 25


@pytest.mark.parametrize("name", _ENGINE_CASES)
def test_black_style_swaption_engine(cpp: dict[str, Any], name: str) -> None:
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    swaption = _build_swaption(inputs)
    disc = mkt.discount_curve()
    model = (
        CashAnnuityModel.DiscountCurve
        if inputs["cash_annuity_model"] == "DiscountCurve"
        else CashAnnuityModel.SwapRate
    )
    if inputs["engine"] == "BlackSwaptionEngine":
        engine = BlackSwaptionEngine(
            disc, inputs["volatility"], Actual365Fixed(), inputs["displacement"], model
        )
    else:
        engine = BachelierSwaptionEngine(
            disc, inputs["volatility"], Actual365Fixed(), model
        )
    swaption.set_pricing_engine(engine)

    if expected["throws"]:
        with pytest.raises(LibraryException):
            swaption.npv()
        return

    # TIGHT: every step is closed form (schedule -> discount factors ->
    # Black/Bachelier). The observed drift across all 25 cases is < 1e-15 rel.
    tolerance.tight(swaption.npv(), expected["npv"])
    extras = swaption.additional_results()
    assert len(extras) == expected["additional_results_count"]
    for py_key, cpp_key in _EXTRA_KEYS:
        tolerance.tight(extras[py_key], expected[cpp_key])
    assert str(swaption.valuation_date()) == expected["valuation_date"]


# --------------------------------------------------------------------------
# SwaptionVolatilityStructure-driven constructors
# --------------------------------------------------------------------------

_VOLTS_CASES = sorted(
    k
    for k, v in reference_reader.load("v143/pe/swaption").items()
    if v["inputs"].get("ctor") == "SwaptionVolatilityStructure"
)


def test_volts_cases_are_present() -> None:
    assert len(_VOLTS_CASES) == 6


@pytest.mark.parametrize("name", _VOLTS_CASES)
def test_vol_structure_constructor(cpp: dict[str, Any], name: str) -> None:
    """The vol structure's OWN reference date drives stdDev and timeToExpiry.

    ``vol_settlement_days`` 0 vs 2 moves that reference date. A port that
    reads the discount curve's reference date instead passes the settle-0
    case and fails the settle-2 one.
    """
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    vol_ts = ConstantSwaptionVolatility(
        settlement_days=inputs["vol_settlement_days"],
        calendar=mkt.calendar(),
        business_day_convention=BusinessDayConvention.Following,
        volatility=inputs["volatility"],
        day_counter=Actual365Fixed(),
        volatility_type=_VOL_TYPES[inputs["vol_type"]],
        shift=inputs["shift"],
    )
    disc = mkt.discount_curve()

    if expected["throws"]:
        build = (
            (lambda: BlackSwaptionEngine(disc, vol_ts))
            if inputs["engine"] == "BlackSwaptionEngine"
            else (lambda: BachelierSwaptionEngine(disc, vol_ts))
        )
        with pytest.raises(LibraryException):
            build()
        return

    assert str(vol_ts.reference_date()) == expected["vol_reference_date"]
    tolerance.tight(
        vol_ts.time_from_reference(mkt.exercise_date()),
        expected["vol_reference_time_to_expiry"],
    )

    swap = mkt.make_swap(
        mkt.forwarding_curve(), _SWAP_TYPES[inputs["swap_type"]], inputs["fixed_rate"]
    )
    swaption = Swaption(swap, EuropeanExercise(mkt.exercise_date()))
    engine = (
        BlackSwaptionEngine(disc, vol_ts)
        if inputs["engine"] == "BlackSwaptionEngine"
        else BachelierSwaptionEngine(disc, vol_ts)
    )
    swaption.set_pricing_engine(engine)
    tolerance.tight(swaption.npv(), expected["npv"])
    extras = swaption.additional_results()
    for py_key, cpp_key in _EXTRA_KEYS[1:]:
        tolerance.tight(extras[py_key], expected[cpp_key])


def test_settle0_and_settle2_differ(cpp: dict[str, Any]) -> None:
    """Regression pin: the two vol reference dates must NOT give the same NPV."""
    a = cpp["black_volts_settle0_payer"]["expected"]
    b = cpp["black_volts_settle2_payer"]["expected"]
    assert a["npv"] != b["npv"]
    assert a["time_to_expiry"] != b["time_to_expiry"]


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_bermudan_exercise_rejected(cpp: dict[str, Any]) -> None:
    expected = cpp["black_bermudan_throws"]["expected"]
    assert expected["throws"]
    swap = mkt.make_swap(mkt.forwarding_curve(), SwapType.Payer, 0.03)
    swaption = Swaption(swap, BermudanExercise(mkt.bermudan_dates()))
    swaption.set_pricing_engine(
        BlackSwaptionEngine(mkt.discount_curve(), 0.20, Actual365Fixed(), 0.0)
    )
    with pytest.raises(LibraryException, match="not a European option"):
        swaption.npv()


def test_swap_starting_before_exercise_rejected(cpp: dict[str, Any]) -> None:
    case = cpp["black_swap_starts_before_exercise_throws"]
    assert case["expected"]["throws"]
    late = mkt.calendar().advance(mkt.swap_start(), 1, TimeUnit.Years)
    assert str(late) == case["inputs"]["exercise_date_override"]
    swap = mkt.make_swap(mkt.forwarding_curve(), SwapType.Payer, 0.03)
    swaption = Swaption(swap, EuropeanExercise(late))
    swaption.set_pricing_engine(
        BlackSwaptionEngine(mkt.discount_curve(), 0.20, Actual365Fixed(), 0.0)
    )
    with pytest.raises(LibraryException, match="before exercise date"):
        swaption.npv()


def test_vol_structure_ctor_rejects_day_counter() -> None:
    """A vol structure carries its own day counter; passing one is a config error."""
    vol_ts = ConstantSwaptionVolatility(
        settlement_days=0,
        calendar=mkt.calendar(),
        business_day_convention=BusinessDayConvention.Following,
        volatility=0.20,
        day_counter=Actual365Fixed(),
    )
    with pytest.raises(LibraryException):
        BlackSwaptionEngine(mkt.discount_curve(), vol_ts, Actual365Fixed())


def test_probe_evaluation_date_is_pinned() -> None:
    """The autouse fixture really did move the clock."""
    assert ObservableSettings().evaluation_date == Date.from_ymd(3, Month.March, 2025)
