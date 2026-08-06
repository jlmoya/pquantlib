"""Cross-validate ComplexChooserOption + AnalyticComplexChooserEngine.

Probe source: migration-harness/cpp/probes/v143_inst_margrabechooser/probe.cpp
Reference:    migration-harness/references/v143/inst/margrabechooser.json

The reference grid varies the call strike, the put strike, the choosing
date, the call expiry and the put expiry **independently**, and includes a
strikes-swapped and a put-expires-before-call case, so a port that swaps
or drops either pair fails.

Tolerance: see ``_BVN_*`` below. The gap is entirely the shared
``BivariateCumulativeNormalDistributionDr78`` alias, not this engine.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.complex_chooser_option import (
    ComplexChooserOption,
    ComplexChooserOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.exotic.analytic_complex_chooser_engine import (
    AnalyticComplexChooserEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date

# C++ AnalyticComplexChooserEngine calls BivariateCumulativeNormalDistributionDr78
# (Drezner 1978, a 5x5 Gauss quadrature good to ~6 decimal places). pquantlib
# aliases that name onto scipy's Genz-Bretz implementation, which is *more*
# accurate but does not reproduce Drezner's values. Four bivariate CDF
# evaluations enter the price scaled by spot/strike (order 50), so the ~1e-6
# per-call difference reaches ~1.5e-5 absolute / ~2.4e-6 relative here.
#
# Evidence that this is the *only* source of divergence: substituting a
# faithful Drezner-1978 implementation into this engine drops the residual
# against the same reference to <= 4.9e-15 relative across all 12 cases.
# Bounds below match the tier already used by the other Dr78-dependent
# engines in this port (see tests/experimental/barrieroption/
# test_partial_time_barrier_option.py).
_BVN_ABS = 5e-5
_BVN_REL = 5e-6
_BVN_REASON = (
    "C++ uses BivariateCumulativeNormalDistributionDr78 (Drezner 1978, ~6dp); "
    "pquantlib aliases that name onto scipy Genz-Bretz. Swapping in a faithful "
    "Dr78 reduces the residual to 5e-15 relative, so the gap is the shared "
    "math module, not the chooser engine"
)

_UNFILLED_GREEKS = ("delta", "gamma", "theta", "vega", "rho", "dividend_rho")


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/inst/margrabechooser")


@pytest.fixture(scope="module")
def day_counter() -> DayCounter:
    return Actual365Fixed()


@pytest.fixture(scope="module")
def today(cpp_ref: dict[str, Any]) -> Date:
    return Date(int(cpp_ref["meta"]["today_serial"]))


def _process(today: Date, day_counter: DayCounter, inputs: dict[str, Any]) -> GeneralizedBlackScholesProcess:
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(inputs["spot"]),
        dividend_ts=FlatForward.from_rate(today, inputs["q"], day_counter),
        risk_free_ts=FlatForward.from_rate(today, inputs["r"], day_counter),
        black_vol_ts=BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=day_counter,
            volatility=inputs["vol"],
        ),
    )


def _option(inputs: dict[str, Any]) -> ComplexChooserOption:
    return ComplexChooserOption(
        Date(int(inputs["choosing_serial"])),
        inputs["strike_call"],
        inputs["strike_put"],
        EuropeanExercise(Date(int(inputs["call_expiry_serial"]))),
        EuropeanExercise(Date(int(inputs["put_expiry_serial"]))),
    )


# ---------------------------------------------------------------- instrument --


def test_complex_chooser_base_is_built_from_the_call_leg(today: Date) -> None:
    """C++ builds the OneAssetOption from the CALL strike and CALL exercise."""
    call_exercise = EuropeanExercise(today + 180)
    put_exercise = EuropeanExercise(today + 210)
    opt = ComplexChooserOption(today + 90, 55.0, 48.0, call_exercise, put_exercise)
    assert isinstance(opt, OneAssetOption)
    payoff = opt.payoff()
    assert isinstance(payoff, PlainVanillaPayoff)
    assert payoff.option_type() == OptionType.Call
    assert payoff.strike() == 55.0
    assert opt.exercise() is call_exercise


def test_complex_chooser_setup_arguments_carries_all_five_fields(today: Date) -> None:
    """All five constructor arguments must reach the engine arguments."""
    choosing = today + 90
    call_exercise = EuropeanExercise(today + 180)
    put_exercise = EuropeanExercise(today + 210)
    opt = ComplexChooserOption(choosing, 55.0, 48.0, call_exercise, put_exercise)
    args = ComplexChooserOptionArguments()
    opt.setup_arguments(args)
    assert args.choosing_date == choosing
    assert args.strike_call == 55.0
    assert args.strike_put == 48.0
    assert args.exercise_call is call_exercise
    assert args.exercise_put is put_exercise
    args.validate()  # no raise


def test_complex_chooser_arguments_reject_null_choosing_date(cpp_ref: dict[str, Any], today: Date) -> None:
    """# C++ parity: ``QL_REQUIRE(choosingDate != Date(), ...)``."""
    assert cpp_ref["complex_chooser_errors"]["null_choosing_date"] == {"raises": True}
    args = ComplexChooserOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 55.0)
    args.exercise = EuropeanExercise(today + 180)
    args.exercise_call = EuropeanExercise(today + 180)
    args.exercise_put = EuropeanExercise(today + 210)
    args.choosing_date = Date()
    with pytest.raises(LibraryException, match="no choosing date given"):
        args.validate()


@pytest.mark.parametrize(
    ("probe_key", "choosing_offset", "call_offset", "put_offset", "message"),
    [
        (
            "choosing_after_call_expiry",
            365,
            270,
            300,
            "choosing date later than or equal to Call maturity date",
        ),
        (
            "choosing_after_put_expiry",
            240,
            270,
            210,
            "choosing date later than or equal to Put maturity date",
        ),
    ],
)
def test_complex_chooser_arguments_reject_out_of_order_dates(
    cpp_ref: dict[str, Any],
    today: Date,
    probe_key: str,
    choosing_offset: int,
    call_offset: int,
    put_offset: int,
    message: str,
) -> None:
    """The call and the put maturity checks are independent."""
    assert cpp_ref["complex_chooser_errors"][probe_key] == {"raises": True}
    opt = ComplexChooserOption(
        today + choosing_offset,
        55.0,
        48.0,
        EuropeanExercise(today + call_offset),
        EuropeanExercise(today + put_offset),
    )
    args = ComplexChooserOptionArguments()
    opt.setup_arguments(args)
    with pytest.raises(LibraryException, match=message):
        args.validate()


# ------------------------------------------------------------------- pricing --


def test_complex_chooser_npv_matches_cpp(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter
) -> None:
    """NPV matches C++ across the whole independently-varied grid."""
    for name, case in cpp_ref["complex_chooser"].items():
        inputs = case["inputs"]
        opt = _option(inputs)
        opt.set_pricing_engine(AnalyticComplexChooserEngine(_process(today, day_counter, inputs)))
        expected = case["value"]
        assert not isinstance(expected, dict), f"{name} unexpectedly raises"
        tolerance.custom(
            opt.npv(),
            expected,
            abs_tol=_BVN_ABS,
            rel_tol=_BVN_REL,
            reason=f"{name}: {_BVN_REASON}",
        )


def test_complex_chooser_every_argument_moves_the_price(
    cpp_ref: dict[str, Any],
) -> None:
    """Each perturbation case must differ from the base case.

    This is the guard against an argument that is accepted and dropped:
    every non-base case in the grid changes exactly one input, so equality
    with the base price would mean the input never reached the engine.
    """
    grid = cpp_ref["complex_chooser"]
    base = grid["base"]["value"]
    for name, case in grid.items():
        if name == "base":
            continue
        assert abs(case["value"] - base) > 1e-3, name


def test_complex_chooser_strikes_are_not_interchangeable(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter
) -> None:
    """Swapping the call and put strikes changes the price."""
    grid = cpp_ref["complex_chooser"]
    base_inputs = grid["base"]["inputs"]
    swapped_inputs = grid["strikes_swapped"]["inputs"]
    assert swapped_inputs["strike_call"] == base_inputs["strike_put"]
    assert swapped_inputs["strike_put"] == base_inputs["strike_call"]

    prices: list[float] = []
    for inputs in (base_inputs, swapped_inputs):
        opt = _option(inputs)
        opt.set_pricing_engine(AnalyticComplexChooserEngine(_process(today, day_counter, inputs)))
        prices.append(opt.npv())
    assert abs(prices[0] - prices[1]) > 1.0


def test_complex_chooser_expiries_are_not_interchangeable(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter
) -> None:
    """A put expiring before the call prices differently from the reverse."""
    grid = cpp_ref["complex_chooser"]
    inputs = grid["put_expiry_before_call"]["inputs"]
    assert inputs["put_expiry_serial"] < inputs["call_expiry_serial"]

    opt = _option(inputs)
    opt.set_pricing_engine(AnalyticComplexChooserEngine(_process(today, day_counter, inputs)))
    tolerance.custom(
        opt.npv(),
        grid["put_expiry_before_call"]["value"],
        abs_tol=_BVN_ABS,
        rel_tol=_BVN_REL,
        reason=_BVN_REASON,
    )

    # Same option with the two exercises exchanged must price differently.
    swapped = ComplexChooserOption(
        Date(int(inputs["choosing_serial"])),
        inputs["strike_call"],
        inputs["strike_put"],
        EuropeanExercise(Date(int(inputs["put_expiry_serial"]))),
        EuropeanExercise(Date(int(inputs["call_expiry_serial"]))),
    )
    swapped.set_pricing_engine(AnalyticComplexChooserEngine(_process(today, day_counter, inputs)))
    assert abs(swapped.npv() - opt.npv()) > 1e-2


@pytest.mark.parametrize("field", _UNFILLED_GREEKS)
def test_complex_chooser_engine_fills_value_only(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter, field: str
) -> None:
    """The engine sets ``value`` only, so every Greek accessor raises."""
    assert cpp_ref["complex_chooser"]["base"][field] == {"raises": True}
    inputs = cpp_ref["complex_chooser"]["base"]["inputs"]
    opt = _option(inputs)
    opt.set_pricing_engine(AnalyticComplexChooserEngine(_process(today, day_counter, inputs)))
    with pytest.raises(LibraryException, match="not provided"):
        getattr(opt, field)()
