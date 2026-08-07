"""Cross-validate the v1.43 finite-difference Black-Scholes engines against C++.

Reference: ``migration-harness/references/v143/pe/fdbs`` produced by
``migration-harness/cpp/probes/v143_pe_fdbs/probe.cpp``.

Covers :class:`FdBlackScholesVanillaEngine` / :class:`MakeFdBlackScholesVanillaEngine`,
:class:`FdBlackScholesShoutEngine`, :class:`FdCEVVanillaEngine`,
:class:`FdCIRVanillaEngine` / :class:`MakeFdCIRVanillaEngine`,
:class:`FdSimpleBSSwingEngine`, :class:`FdBlackScholesAsianEngine`,
:class:`FdBlackScholesBarrierEngine`, :class:`FdBlackScholesRebateEngine`,
:class:`Fd2dBlackScholesVanillaEngine` and
:class:`FdndimBlackScholesVanillaEngine`.

Every case carries its whole market description in ``inputs`` — evaluation
date, curves, vols, dividends, grids, scheme, payoff — so the sweeps below
rebuild the market rather than restating constants.

Tolerances
----------
TIGHT (``1e-14`` abs / ``1e-12`` rel) is the default and covers every NPV and
almost every Greek. Nothing here is a statistical band: a finite-difference
rollback is deterministic, so the only slack is summation order.

Three families need a derived bound, each justified inline at the assertion:

* ``gamma`` on the CEV engine and on the 2-D basket engine, where the reported
  number is a *difference of spline derivatives* and loses digits to
  cancellation;
* ``theta``, which every solver computes as ``(V(t_snapshot) - V(0)) / t``
  with ``t = 0.99/365``, i.e. multiplied by ~369; and
* the ``rho = 0.4`` CIR case, which sits on the edge of a genuine upstream
  instability — see :func:`test_fd_cir_engine_is_unstable_for_large_rho`,
  where the C++ engine itself diverges to 3e37 under time-grid refinement.

The bound in each case is derived from that arithmetic, never widened to
force green.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pytest

from pquantlib.cashflows.dividend import Dividend, dividend_vector
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import (
    AmericanExercise,
    BermudanExercise,
    EuropeanExercise,
    Exercise,
)
from pquantlib.experimental.finitedifferences.swing_exercise import SwingExercise
from pquantlib.experimental.finitedifferences.vanilla_swing_option import (
    VanillaSwingOption,
)
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOption
from pquantlib.instruments.average_type import AverageType
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.instruments.basket_option import (
    AverageBasketPayoff,
    BasketOption,
    BasketPayoff,
    MaxBasketPayoff,
    MinBasketPayoff,
    SpreadBasketPayoff,
)
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.matrix import Matrix
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.fdm_simple_process_1d_mesher import (
    FdmSimpleProcess1dMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_cir_op import FdmCIROp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import (
    FdmQuantoHelper,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.fd_black_scholes_asian_engine import (
    FdBlackScholesAsianEngine,
)
from pquantlib.pricingengines.barrier.fd_black_scholes_barrier_engine import (
    FdBlackScholesBarrierEngine,
)
from pquantlib.pricingengines.barrier.fd_black_scholes_rebate_engine import (
    FdBlackScholesRebateEngine,
)
from pquantlib.pricingengines.basket.fd_2d_black_scholes_vanilla_engine import (
    Fd2dBlackScholesVanillaEngine,
)
from pquantlib.pricingengines.basket.fd_ndim_black_scholes_vanilla_engine import (
    PDE_MAX_SUPPORTED_DIM,
    FdndimBlackScholesVanillaEngine,
)
from pquantlib.pricingengines.vanilla.cash_dividend_european_engine import (
    CashDividendModel,
)
from pquantlib.pricingengines.vanilla.fd_black_scholes_shout_engine import (
    FdBlackScholesShoutEngine,
)
from pquantlib.pricingengines.vanilla.fd_black_scholes_vanilla_engine import (
    FdBlackScholesVanillaEngine,
    MakeFdBlackScholesVanillaEngine,
)
from pquantlib.pricingengines.vanilla.fd_cev_vanilla_engine import FdCEVVanillaEngine
from pquantlib.pricingengines.vanilla.fd_cir_vanilla_engine import (
    FdCIRVanillaEngine,
    MakeFdCIRVanillaEngine,
)
from pquantlib.pricingengines.vanilla.fd_simple_bs_swing_engine import (
    FdSimpleBSSwingEngine,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess
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
from pquantlib.time.month import Month

# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/fdbs")


@pytest.fixture(autouse=True)
def _pin_evaluation_date(cpp: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin ``Settings::evaluationDate()`` to the probe's own date.

    The probe sets it in ``main()``
    (migration-harness/cpp/probes/v143_pe_fdbs/probe.cpp, first line of
    ``int main()``) and records it as ``inputs.evaluation_date``.
    """
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = _date(cpp["vanilla_european_call_douglas"]["inputs"]["evaluation_date"])
    yield
    settings.evaluation_date = saved


# --- helpers -----------------------------------------------------------------


def _date(iso: str) -> Date:
    y, m, d = (int(p) for p in iso.split("-"))
    return Date.from_ymd(d, Month(m), y)


def _dates(isos: Sequence[str]) -> list[Date]:
    return [_date(s) for s in isos]


_SCHEMES: dict[str, Callable[[], FdmSchemeDesc]] = {
    "Douglas": FdmSchemeDesc.douglas,
    "CrankNicolson": FdmSchemeDesc.crank_nicolson,
    "ImplicitEuler": FdmSchemeDesc.implicit_euler,
    "Hundsdorfer": FdmSchemeDesc.hundsdorfer,
    "ModifiedHundsdorfer": FdmSchemeDesc.modified_hundsdorfer,
}


def _scheme(name: str) -> FdmSchemeDesc:
    return _SCHEMES[name]()


def _today(inputs: dict[str, Any]) -> Date:
    return _date(inputs["evaluation_date"])


def _process(
    inputs: dict[str, Any],
    *,
    spot_key: str = "spot",
    q_key: str = "dividend_yield",
    vol_key: str = "volatility",
) -> GeneralizedBlackScholesProcess:
    """Rebuild the probe's ``BlackScholesMertonProcess`` from ``inputs``."""
    today = _today(inputs)
    dc = Actual365Fixed()
    return BlackScholesMertonProcess(
        x0=SimpleQuote(float(inputs[spot_key])),
        dividend_ts=FlatForward.from_rate(
            reference_date=today, forward_rate=float(inputs[q_key]), day_counter=dc
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=today,
            forward_rate=float(inputs["risk_free_rate"]),
            day_counter=dc,
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=dc,
            volatility=float(inputs[vol_key]),
        ),
    )


def _payoff(inputs: dict[str, Any]) -> PlainVanillaPayoff:
    option_type = OptionType.Call if inputs["option_type"] == "Call" else OptionType.Put
    return PlainVanillaPayoff(option_type, float(inputs["strike"]))


def _exercise(inputs: dict[str, Any]) -> Exercise:
    kind = inputs["exercise"]
    dates = _dates(inputs["exercise_dates"])
    if kind == "European":
        return EuropeanExercise(dates[0])
    if kind == "American":
        return AmericanExercise(dates[0], dates[1])
    if kind == "Bermudan":
        return BermudanExercise(dates)
    if kind == "Swing":
        return SwingExercise(dates)
    raise AssertionError(f"unknown exercise kind {kind}")


def _dividends(inputs: dict[str, Any]) -> list[Dividend]:
    dates = _dates(inputs.get("dividend_dates", []))
    amounts = [float(a) for a in inputs.get("dividend_amounts", [])]
    if not dates:
        return []
    return list(dividend_vector(dates, amounts))


def _cash_dividend_model(inputs: dict[str, Any]) -> CashDividendModel:
    return (
        CashDividendModel.Escrowed
        if inputs.get("cash_dividend_model") == "Escrowed"
        else CashDividendModel.Spot
    )


def _basket_payoff(kind: str, option_type: OptionType, strike: float, n: int) -> BasketPayoff:
    base = PlainVanillaPayoff(option_type, strike)
    if kind == "Max":
        return MaxBasketPayoff(base)
    if kind == "Min":
        return MinBasketPayoff(base)
    if kind == "Spread":
        return SpreadBasketPayoff(base)
    return AverageBasketPayoff(base, n=n)


#: ``theta`` is not an independent quantity: every FD solver computes it as
#: ``(V(t_snap) - V(0)) / t_snap`` with ``t_snap = 0.99 * min(1/365, first
#: stopping time)`` — i.e. a *difference of two spline evaluations divided by
#: ~1/369*, so any relative error in V is amplified by ~369/|theta| and any
#: absolute error by 369. With V ~ 10 and TIGHT rel 1e-12 on V, the induced
#: bound on theta is 10 * 1e-12 * 369 / |theta| ~ 4e-9 / |theta| relative,
#: hence 1e-8 relative is the *derived* bound, not a relaxed one.
_THETA_REL: float = 1e-8
_THETA_ABS: float = 1e-8


def _assert_greeks(
    expected: dict[str, Any],
    *,
    npv: float,
    delta: float | None = None,
    gamma: float | None = None,
    theta: float | None = None,
    name: str,
    gamma_tol: tuple[float, float] | None = None,
    gamma_reason: str | None = None,
) -> None:
    tolerance.tight(npv, float(expected["npv"]), reason=f"{name}: npv")
    if delta is not None:
        tolerance.tight(delta, float(expected["delta"]), reason=f"{name}: delta")
    if gamma is not None:
        if gamma_tol is None:
            tolerance.tight(gamma, float(expected["gamma"]), reason=f"{name}: gamma")
        else:
            assert gamma_reason is not None
            tolerance.custom(
                gamma,
                float(expected["gamma"]),
                abs_tol=gamma_tol[0],
                rel_tol=gamma_tol[1],
                reason=f"{name}: gamma — {gamma_reason}",
            )
    if theta is not None:
        tolerance.custom(
            theta,
            float(expected["theta"]),
            abs_tol=_THETA_ABS,
            rel_tol=_THETA_REL,
            reason=f"{name}: theta — backward difference over t_snap = 0.99/365 "
            "amplifies any error in the value by ~369x; see _THETA_REL",
        )


def _throws(f: Callable[[], object]) -> tuple[bool, str]:
    try:
        f()
    except LibraryException as e:
        return True, str(e)
    return False, ""


# --- 1. FdBlackScholesVanillaEngine -------------------------------------------

_VANILLA_CASES = [
    "vanilla_european_call_douglas",
    "vanilla_european_put_douglas",
    "vanilla_european_call_crank_nicolson",
    "vanilla_european_call_implicit_euler",
    "vanilla_american_put_douglas",
    "vanilla_american_call_damped",
    "vanilla_bermudan_put",
    "vanilla_local_vol",
    "vanilla_european_dividends_spot",
    "vanilla_european_dividends_escrowed",
    "vanilla_american_dividends_spot",
    "vanilla_american_dividends_escrowed",
]


def _vanilla_option(case: dict[str, Any]) -> VanillaOption:
    inputs = case["inputs"]
    process = _process(inputs)
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(
        FdBlackScholesVanillaEngine(
            process,
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            int(inputs["damping_steps"]),
            _scheme(inputs["scheme"]),
            bool(inputs["local_vol"]),
            cash_dividend_model=_cash_dividend_model(inputs),
            dividends=_dividends(inputs),
        )
    )
    return option


@pytest.mark.parametrize("name", _VANILLA_CASES)
def test_fd_black_scholes_vanilla_engine(cpp: dict[str, Any], name: str) -> None:
    """The engine prices, and fills value, delta, gamma and theta."""
    case = cpp[name]
    option = _vanilla_option(case)
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name=name,
    )


def test_vanilla_default_scheme_is_douglas(cpp: dict[str, Any]) -> None:
    """The engine's default ``scheme_desc`` is Douglas, not Crank-Nicolson.

    In one dimension those two coincide bit for bit (the probe pins that as
    ``vanilla_european_call_crank_nicolson``), so the discriminating comparison
    is against ``ImplicitEuler``, which C++ prices ~0.011 lower.
    """
    douglas = cpp["vanilla_european_call_douglas"]
    cn = cpp["vanilla_european_call_crank_nicolson"]
    implicit = cpp["vanilla_european_call_implicit_euler"]
    tolerance.exact(
        float(douglas["expected"]["npv"]),
        float(cn["expected"]["npv"]),
        reason="C++ Douglas and Crank-Nicolson agree bit-for-bit in 1-D",
    )
    assert float(implicit["expected"]["npv"]) != float(douglas["expected"]["npv"])

    inputs = douglas["inputs"]
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(
        FdBlackScholesVanillaEngine(
            _process(inputs), int(inputs["t_grid"]), int(inputs["x_grid"])
        )
    )
    tolerance.tight(
        option.npv(),
        float(douglas["expected"]["npv"]),
        reason="default scheme must be Douglas",
    )


def test_vanilla_escrowed_differs_from_spot(cpp: dict[str, Any]) -> None:
    """Escrowed and Spot cash-dividend models are genuinely different prices.

    Guards against a port that accepts ``cash_dividend_model`` and ignores it:
    the two European prices differ by ~0.26 in C++.
    """
    spot = float(cpp["vanilla_european_dividends_spot"]["expected"]["npv"])
    escrowed = float(cpp["vanilla_european_dividends_escrowed"]["expected"]["npv"])
    assert abs(spot - escrowed) > 0.2


def test_vanilla_quanto(cpp: dict[str, Any]) -> None:
    """The quanto branch: mesher swaps in a QuantoTermStructure, op adds drift."""
    case = cpp["vanilla_quanto"]
    inputs = case["inputs"]
    today = _today(inputs)
    dc = Actual365Fixed()
    process = _process(inputs)
    quanto = FdmQuantoHelper(
        FlatForward.from_rate(
            reference_date=today,
            forward_rate=float(inputs["risk_free_rate"]),
            day_counter=dc,
        ),
        FlatForward.from_rate(
            reference_date=today,
            forward_rate=float(inputs["foreign_rate"]),
            day_counter=dc,
        ),
        BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=dc,
            volatility=float(inputs["fx_volatility"]),
        ),
        float(inputs["equity_fx_correlation"]),
        float(inputs["exch_rate_atm_level"]),
    )
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(
        FdBlackScholesVanillaEngine(
            process,
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            int(inputs["damping_steps"]),
            _scheme(inputs["scheme"]),
            quanto_helper=quanto,
        )
    )
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name="vanilla_quanto",
    )


def test_make_fd_black_scholes_vanilla_engine_defaults(cpp: dict[str, Any]) -> None:
    """Builder defaults: tGrid=100, xGrid=100, dampingSteps=0, Douglas, Spot."""
    case = cpp["vanilla_make_defaults"]
    inputs = case["inputs"]
    expected = case["expected"]
    assert int(expected["default_t_grid"]) == 100
    assert int(expected["default_x_grid"]) == 100
    assert int(expected["default_damping_steps"]) == 0
    assert expected["default_scheme"] == "Douglas"

    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(MakeFdBlackScholesVanillaEngine(_process(inputs)).engine())
    _assert_greeks(
        expected,
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name="vanilla_make_defaults",
    )

    # The explicitly-parameterised twin must agree bit for bit in C++...
    twin = cpp["vanilla_make_defaults_explicit_twin"]["expected"]
    tolerance.exact(float(expected["npv"]), float(twin["npv"]))
    # ... and in the port.
    explicit = VanillaOption(_payoff(inputs), _exercise(inputs))
    explicit.set_pricing_engine(
        FdBlackScholesVanillaEngine(_process(inputs), 100, 100, 0, FdmSchemeDesc.douglas())
    )
    tolerance.exact(explicit.npv(), option.npv(), reason="builder == explicit engine")


def test_make_fd_black_scholes_vanilla_engine_all_knobs(cpp: dict[str, Any]) -> None:
    """Every ``with*`` setter is wired through to the engine."""
    case = cpp["vanilla_make_all_knobs"]
    inputs = case["inputs"]
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(
        MakeFdBlackScholesVanillaEngine(_process(inputs))
        .with_t_grid(int(inputs["t_grid"]))
        .with_x_grid(int(inputs["x_grid"]))
        .with_damping_steps(int(inputs["damping_steps"]))
        .with_fdm_scheme_desc(_scheme(inputs["scheme"]))
        .with_cash_dividends(
            _dates(inputs["dividend_dates"]),
            [float(a) for a in inputs["dividend_amounts"]],
        )
        .with_cash_dividend_model(CashDividendModel.Escrowed)
        .engine()
    )
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name="vanilla_make_all_knobs",
    )


def test_vanilla_escrowed_rejects_quanto(cpp: dict[str, Any]) -> None:
    """C++: "Escrowed dividend model is not supported for Quanto-Options"."""
    case = cpp["vanilla_escrowed_quanto_rejected"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["throws"] is True

    today = _today(inputs)
    dc = Actual365Fixed()
    quanto = FdmQuantoHelper(
        FlatForward.from_rate(reference_date=today, forward_rate=0.05, day_counter=dc),
        FlatForward.from_rate(reference_date=today, forward_rate=0.03, day_counter=dc),
        BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=dc,
            volatility=0.15,
        ),
        0.4,
        1.25,
    )
    option = VanillaOption(
        _payoff(inputs), EuropeanExercise(_date("2026-05-15"))
    )
    option.set_pricing_engine(
        FdBlackScholesVanillaEngine(
            _process(inputs),
            40,
            40,
            0,
            FdmSchemeDesc.douglas(),
            cash_dividend_model=CashDividendModel.Escrowed,
            dividends=_dividends(inputs),
            quanto_helper=quanto,
        )
    )
    threw, what = _throws(option.npv)
    assert threw
    assert what == expected["what"]


def test_vanilla_escrowed_negative_spot(cpp: dict[str, Any]) -> None:
    """C++: "spot minus dividends becomes negative"."""
    case = cpp["vanilla_escrowed_negative_spot"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["throws"] is True

    option = VanillaOption(_payoff(inputs), EuropeanExercise(_date("2026-05-15")))
    option.set_pricing_engine(
        FdBlackScholesVanillaEngine(
            _process(inputs),
            40,
            40,
            0,
            FdmSchemeDesc.douglas(),
            cash_dividend_model=CashDividendModel.Escrowed,
            dividends=_dividends(inputs),
        )
    )
    threw, what = _throws(option.npv)
    assert threw
    assert what == expected["what"]


# --- 2. FdBlackScholesShoutEngine ---------------------------------------------

_SHOUT_CASES = [
    "shout_european_call",
    "shout_european_put",
    "shout_american_put",
    "shout_american_put_dividends",
]


@pytest.mark.parametrize("name", _SHOUT_CASES)
def test_fd_black_scholes_shout_engine(cpp: dict[str, Any], name: str) -> None:
    """The shout engine prices, with the escrowed dividend adjustment applied."""
    case = cpp[name]
    inputs = case["inputs"]
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(
        FdBlackScholesShoutEngine(
            _process(inputs),
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            int(inputs["damping_steps"]),
            _scheme(inputs["scheme"]),
            dividends=_dividends(inputs),
        )
    )
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name=name,
    )


def _vanilla_twin_npv(inputs: dict[str, Any]) -> float:
    """The plain-vanilla FD price on the same market, grids and scheme."""
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(
        FdBlackScholesVanillaEngine(
            _process(inputs),
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            int(inputs["damping_steps"]),
            _scheme(inputs["scheme"]),
        )
    )
    return option.npv()


def test_shout_right_bites_only_with_early_exercise(cpp: dict[str, Any]) -> None:
    """Where the shout right can be used it is worth ~3.0; where it cannot, ~0.

    Under a *European* exercise the composite carries no step condition, so the
    shout inner-value calculator is only ever evaluated as the terminal
    condition and the price must land on the vanilla one to within the
    difference between ``FdmShoutLogInnerValueCalculator`` and
    ``FdmLogInnerValue`` at t = T (both cell-averaged, ~0.1%).

    Under an *American* exercise the same calculator floors every time step and
    the price jumps from 11.83 to 14.84. A port that wired the plain
    ``FdmLogInnerValue`` into the step conditions would pass the European check
    and fail this one; a port that never applied the step condition at all
    would do the reverse.
    """
    for name in ("shout_european_call", "shout_european_put"):
        inputs = cpp[name]["inputs"]
        shout_npv = float(cpp[name]["expected"]["npv"])
        vanilla_npv = _vanilla_twin_npv(inputs)
        assert abs(shout_npv - vanilla_npv) / vanilla_npv < 1e-3, name

    american = cpp["shout_american_put"]
    shout_npv = float(american["expected"]["npv"])
    vanilla_npv = _vanilla_twin_npv(american["inputs"])
    assert shout_npv > vanilla_npv + 2.5

    # ... and the payoff type is honoured: call and put shout values differ.
    assert float(cpp["shout_european_call"]["expected"]["npv"]) != float(
        cpp["shout_european_put"]["expected"]["npv"]
    )


# --- 3. FdCEVVanillaEngine ----------------------------------------------------

_CEV_CASES = [
    "cev_european_call_beta06",
    "cev_european_put_beta06",
    "cev_european_call_beta14",
    "cev_american_put_beta06",
    "cev_european_call_scaled",
    "cev_european_call_at_kink",
]


@pytest.mark.parametrize("name", _CEV_CASES)
def test_fd_cev_vanilla_engine(cpp: dict[str, Any], name: str) -> None:
    """The CEV engine prices for beta below and above 1, European and American."""
    case = cpp[name]
    inputs = case["inputs"]
    today = _today(inputs)
    discount = FlatForward.from_rate(
        reference_date=today,
        forward_rate=float(inputs["risk_free_rate"]),
        day_counter=Actual365Fixed(),
    )
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(
        FdCEVVanillaEngine(
            float(inputs["f0"]),
            float(inputs["alpha"]),
            float(inputs["beta"]),
            discount,
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            int(inputs["damping_steps"]),
            float(inputs["scaling_factor"]),
            float(inputs["eps"]),
            _scheme(inputs["scheme"]),
        )
    )
    # gamma on this engine is the RAW second derivative of a
    # MonotonicCubicNaturalSpline in forward space. At the ``*_at_kink`` case
    # the read-out point coincides with the payoff kink, where that second
    # derivative is discontinuous, so the value is a difference of two
    # O(1) spline coefficients divided by h^2 ~ (mesh spacing)^2. With
    # xGrid = 200 over a forward range of ~O(300), h ~ 1.5, so a 1e-12
    # relative error in the coefficients becomes ~1e-12/h^2 ~ 4e-13 absolute
    # in gamma. 1e-11 absolute is that bound with one decade of headroom.
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name=name,
        gamma_tol=(1e-11, 1e-9),
        gamma_reason=(
            "raw spline second derivative in forward space; a 1e-12 relative "
            "error in the spline coefficients is 1e-12/h^2 ~ 4e-13 absolute "
            "in gamma at h ~ 1.5"
        ),
    )


def test_cev_lower_boundary_only_when_delta_below_two(cpp: dict[str, Any]) -> None:
    """``delta = (1-2 beta)/(1-beta) < 2`` decides the lower Dirichlet boundary.

    beta = 0.6 gives delta = -0.5 (boundary present); beta = 1.4 gives
    delta = 4.5 (absent). The probe records both, so the branch is not a
    matter of opinion.
    """
    low = cpp["cev_european_call_beta06"]["inputs"]
    high = cpp["cev_european_call_beta14"]["inputs"]
    assert low["lower_boundary_present"] is True
    assert high["lower_boundary_present"] is False
    for inputs in (low, high):
        beta = float(inputs["beta"])
        tolerance.tight(
            (1 - 2 * beta) / (1 - beta), float(inputs["cev_delta_exponent"])
        )


def test_cev_gamma_at_the_kink_is_the_fragile_one(cpp: dict[str, Any]) -> None:
    """Documents why the CEV call strike is moved off ``f0`` in most cases.

    At K == f0 the spline's second derivative is evaluated exactly on the
    payoff kink, and C++ reports gamma = -0.1335 for the call while the
    corresponding put (K = 110, away from the kink) reports +0.0116. That is
    a property of the read-out point, not of the port.
    """
    at_kink = float(cpp["cev_european_call_at_kink"]["expected"]["gamma"])
    off_kink = float(cpp["cev_european_call_beta06"]["expected"]["gamma"])
    assert at_kink < 0.0 < off_kink


# --- 4. FdCIRVanillaEngine ----------------------------------------------------

_CIR_CASES = [
    "cir_european_put_rho_small",
    "cir_european_call_rho0",
    "cir_european_call_rho_neg",
    "cir_european_put_rho_pos",
]


def _cir_engine(inputs: dict[str, Any]) -> FdCIRVanillaEngine:
    return FdCIRVanillaEngine(
        CoxIngersollRossProcess(
            speed=float(inputs["cir_speed"]),
            vol=float(inputs["cir_sigma"]),
            x0=float(inputs["cir_x0"]),
            level=float(inputs["cir_mean"]),
        ),
        _process(inputs),
        int(inputs["t_grid"]),
        int(inputs["x_grid"]),
        int(inputs["r_grid"]),
        int(inputs["damping_steps"]),
        float(inputs["rho"]),
        _scheme(inputs["scheme"]),
    )


@pytest.mark.parametrize("name", _CIR_CASES)
def test_fd_cir_vanilla_engine(cpp: dict[str, Any], name: str) -> None:
    """Stochastic CIR short rate coupled to the equity through rho."""
    case = cpp[name]
    inputs = case["inputs"]
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(_cir_engine(inputs))
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name=name,
    )


def test_fd_cir_vanilla_engine_american(cpp: dict[str, Any]) -> None:
    """American exercise on the 2-D (log-spot, short-rate) grid."""
    case = cpp["cir_american_put_rho_neg"]
    inputs = case["inputs"]
    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(_cir_engine(inputs))
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name="cir_american_put_rho_neg",
    )


def test_fd_cir_operator_matches_cpp_piece_by_piece(cpp: dict[str, Any]) -> None:
    """``FdmCIROp`` is bit-faithful: both mesh axes and all five operator calls.

    This is the control experiment behind
    :func:`test_fd_cir_engine_is_unstable_for_large_rho`. The engine's price at
    ``|rho| = 0.4`` drifts from C++ by ~1e-9 while the operator that produces it
    agrees to ~1e-14, which is what identifies the drift as amplification
    through an unstable rollback rather than a defect in the port.
    """
    case = cpp["cir_operator_diagnostics"]
    inputs, expected = case["inputs"], case["expected"]
    today = _today(inputs)
    dc = Actual365Fixed()
    bs = _process(inputs)
    cir = CoxIngersollRossProcess(
        speed=float(inputs["cir_speed"]),
        vol=float(inputs["cir_sigma"]),
        x0=float(inputs["cir_x0"]),
        level=float(inputs["cir_mean"]),
    )
    assert today == _date(inputs["evaluation_date"])
    assert inputs["day_counter"] == "Actual365Fixed"
    assert isinstance(dc, Actual365Fixed)

    maturity = bs.time(_date(inputs["maturity_date"]))
    short_rate_mesher = FdmSimpleProcess1dMesher(
        int(inputs["r_grid"]), cir, maturity, int(inputs["t_grid"])
    )
    equity_mesher = FdmBlackScholesMesher(
        int(inputs["x_grid"]),
        bs,
        maturity,
        float(inputs["strike"]),
        None,
        None,
        0.0001,
        1.5,
        (float(inputs["strike"]), 0.1),
    )
    mesher = FdmMesherComposite(equity_mesher, short_rate_mesher)
    op = FdmCIROp(mesher, cir, bs, float(inputs["rho"]), float(inputs["strike"]))
    op.set_time(float(inputs["t1"]), float(inputs["t2"]))

    u = np.array(expected["u"], dtype=np.float64)

    def each(got: Any, key: str) -> None:
        want = np.asarray(expected[key], dtype=np.float64)
        got_arr = np.asarray(got, dtype=np.float64)
        assert got_arr.shape == want.shape, key
        for i in range(want.size):
            tolerance.tight(float(got_arr[i]), float(want[i]), reason=f"{key}[{i}]")

    each(equity_mesher.locations(), "equity_locations")
    each(short_rate_mesher.locations(), "rate_locations")
    each(op.apply(u), "apply")
    each(op.apply_mixed(u), "apply_mixed")
    each(op.apply_direction(0, u), "apply_direction0")
    each(op.apply_direction(1, u), "apply_direction1")
    each(op.solve_splitting(0, u, -0.1), "solve_splitting0")
    each(op.solve_splitting(1, u, -0.1), "solve_splitting1")


def test_fd_cir_engine_is_unstable_for_large_rho(cpp: dict[str, Any]) -> None:
    """UPSTREAM FINDING: the engine diverges as the time grid is refined.

    With ``rho = 0.4`` and the engine's own default ``ModifiedHundsdorfer``
    scheme, C++ v1.43 prices 4.30 at tGrid = 10, -6.3e5 at tGrid = 25, 9.0e19
    at tGrid = 50 and 3.4e37 at tGrid = 100. Refinement making the answer worse
    is the signature of an unstable splitting, not of discretisation error.

    Asserted as an ORDER OF MAGNITUDE, not a value: a correct port must blow up
    the same way, but pinning 3.4e37 bit-for-bit would be pinning amplified
    round-off. That is also why every priced CIR case above stays at
    ``|rho| <= 0.25``.
    """
    case = cpp["cir_unstable_large_rho"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["diverges_with_refinement"] is True
    cpp_npvs = [float(v) for v in expected["npv_by_t_grid"]]
    assert abs(cpp_npvs[0]) < 10.0
    assert abs(cpp_npvs[-1]) > 1e30

    got: list[float] = []
    for t_grid in (int(t) for t in expected["t_grids"]):
        option = VanillaOption(_payoff(inputs), _exercise(inputs))
        option.set_pricing_engine(
            FdCIRVanillaEngine(
                CoxIngersollRossProcess(
                    speed=float(inputs["cir_speed"]),
                    vol=float(inputs["cir_sigma"]),
                    x0=float(inputs["cir_x0"]),
                    level=float(inputs["cir_mean"]),
                ),
                _process(inputs),
                t_grid,
                int(inputs["x_grid"]),
                int(inputs["r_grid"]),
                0,
                float(inputs["rho"]),
                _scheme(inputs["scheme"]),
            )
        )
        got.append(option.npv())

    # tGrid = 10 is still on the plausible side of the edge, so it is pinned as
    # a value -- but only to the 1e-9 the amplification permits (a 1e-14
    # operator perturbation grows by ~1e5 over ten steps here; see
    # test_fd_cir_operator_matches_cpp_piece_by_piece for the 1e-14).
    tolerance.custom(
        got[0],
        cpp_npvs[0],
        abs_tol=1e-8,
        rel_tol=1e-8,
        reason="on the edge of the instability: the operator agrees to 1e-14 "
        "but ten Hundsdorfer steps amplify that to ~1e-9 in the price",
    )
    # Beyond the edge only the divergence itself is reproducible.
    for i in range(1, len(got)):
        assert abs(got[i]) > abs(cpp_npvs[i]) / 1e3
        assert abs(got[i]) < abs(cpp_npvs[i]) * 1e3
    assert abs(got[-1]) > 1e30


def test_fd_cir_correlation_moves_the_price(cpp: dict[str, Any]) -> None:
    """rho = 0, -0.15 and +0.15 give three different prices.

    A port that builds the CIR operator without ``FdmCIRMixedPart`` would
    return the rho = 0 number for all three; the gap is ~0.035, four orders of
    magnitude above TIGHT.
    """
    zero = float(cpp["cir_european_call_rho0"]["expected"]["npv"])
    neg = float(cpp["cir_european_call_rho_neg"]["expected"]["npv"])
    pos = float(cpp["cir_european_put_rho_pos"]["expected"]["npv"])
    # The rho0 case is a call and the rho_neg case is a put on a different
    # strike, so the direct comparison that isolates rho is the signed pair
    # below, taken from the probe's own scan.
    assert neg != zero
    assert pos != zero
    assert abs(neg - pos) > 0.05


def test_make_fd_cir_vanilla_engine_defaults(cpp: dict[str, Any]) -> None:
    """Builder defaults: tGrid=10, xGrid=100, rGrid=100, ModifiedHundsdorfer."""
    case = cpp["cir_make_defaults"]
    inputs, expected = case["inputs"], case["expected"]
    assert int(expected["default_t_grid"]) == 10
    assert int(expected["default_x_grid"]) == 100
    assert int(expected["default_r_grid"]) == 100
    assert int(expected["default_damping_steps"]) == 0
    assert expected["default_scheme"] == "ModifiedHundsdorfer"

    option = VanillaOption(_payoff(inputs), _exercise(inputs))
    option.set_pricing_engine(
        MakeFdCIRVanillaEngine(
            CoxIngersollRossProcess(
                speed=float(inputs["cir_speed"]),
                vol=float(inputs["cir_sigma"]),
                x0=float(inputs["cir_x0"]),
                level=float(inputs["cir_mean"]),
            ),
            _process(inputs),
            float(inputs["rho"]),
        )
        .with_x_grid(int(inputs["x_grid"]))
        .with_r_grid(int(inputs["r_grid"]))
        .engine()
    )
    _assert_greeks(
        expected,
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name="cir_make_defaults",
    )
    # The builder's defaults reproduce the explicit engine exactly.
    tolerance.exact(
        float(expected["npv"]),
        float(cpp["cir_european_put_rho_small"]["expected"]["npv"]),
        reason="builder defaults == explicit tGrid=10 / ModifiedHundsdorfer",
    )


# --- 5. FdSimpleBSSwingEngine -------------------------------------------------

_SWING_CASES = ["swing_call_2of5", "swing_put_3of5", "swing_rights_binding"]


@pytest.mark.parametrize("name", _SWING_CASES)
def test_fd_simple_bs_swing_engine(cpp: dict[str, Any], name: str) -> None:
    """Swing option on a (log-spot, rights-used) grid."""
    case = cpp[name]
    inputs = case["inputs"]
    exercise = _exercise(inputs)
    assert isinstance(exercise, SwingExercise)
    option = VanillaSwingOption(
        _payoff(inputs),
        exercise,
        int(inputs["min_exercise_rights"]),
        int(inputs["max_exercise_rights"]),
    )
    option.set_pricing_engine(
        FdSimpleBSSwingEngine(
            _process(inputs),
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            _scheme(inputs["scheme"]),
        )
    )
    tolerance.tight(option.npv(), float(case["expected"]["npv"]), reason=name)


def test_swing_more_rights_is_worth_more(cpp: dict[str, Any]) -> None:
    """Rights are consumed, so 3 of 5 beats 2 of 5 on the same market.

    The two probe cases differ in payoff type as well, so the strict statement
    asserted here is the one that isolates the rights axis: forcing all 5
    rights (``swing_rights_binding``) is worth strictly more than choosing the
    best 2 of the same 5 dates.
    """
    two = float(cpp["swing_call_2of5"]["expected"]["npv"])
    five = float(cpp["swing_rights_binding"]["expected"]["npv"])
    assert five > two


# --- 6. FdBlackScholesAsianEngine ---------------------------------------------

_ASIAN_CASES = ["asian_call_no_past", "asian_put_no_past", "asian_call_running"]


def _asian_option(inputs: dict[str, Any]) -> DiscreteAveragingAsianOption:
    return DiscreteAveragingAsianOption(
        AverageType.Arithmetic,
        float(inputs["running_accumulator"]),
        int(inputs["past_fixings"]),
        _dates(inputs["fixing_dates"]),
        _payoff(inputs),
        _exercise(inputs),
    )


@pytest.mark.parametrize("name", _ASIAN_CASES)
def test_fd_black_scholes_asian_engine(cpp: dict[str, Any], name: str) -> None:
    """Arithmetic-average Asian on a (log-spot, average) grid.

    C++ fills value, delta and gamma but deliberately NOT theta, and the probe
    pins that absence as ``theta_throws``.
    """
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    option = _asian_option(inputs)
    option.set_pricing_engine(
        FdBlackScholesAsianEngine(
            _process(inputs),
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            int(inputs["a_grid"]),
            _scheme(inputs["scheme"]),
        )
    )
    tolerance.tight(option.npv(), float(expected["npv"]), reason=f"{name}: npv")
    tolerance.tight(option.delta(), float(expected["delta"]), reason=f"{name}: delta")
    tolerance.tight(option.gamma(), float(expected["gamma"]), reason=f"{name}: gamma")

    assert expected["theta_throws"] is True
    threw, what = _throws(option.theta)
    assert threw
    assert what == expected["theta_what"]


def test_asian_running_average_moves_the_price(cpp: dict[str, Any]) -> None:
    """A running accumulator with past fixings changes the average axis.

    ``asian_call_running`` sets runningAccumulator = 190 over 2 past fixings,
    so avg = 95 != spot = 100 and the ``min``/``max`` that place the average
    mesh pick different branches. The price drops by ~3.5.
    """
    plain = float(cpp["asian_call_no_past"]["expected"]["npv"])
    running = float(cpp["asian_call_running"]["expected"]["npv"])
    assert running < plain - 3.0


@pytest.mark.parametrize(
    ("name", "average_type", "exercise_kind"),
    [
        ("asian_american_rejected", AverageType.Arithmetic, "American"),
        ("asian_geometric_rejected", AverageType.Geometric, "European"),
    ],
)
def test_asian_rejections(
    cpp: dict[str, Any], name: str, average_type: AverageType, exercise_kind: str
) -> None:
    """The two QL_REQUIREs at the top of ``calculate()``."""
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["throws"] is True

    today = _today(inputs)
    maturity = _date("2026-05-15")
    exercise: Exercise = (
        AmericanExercise(today, maturity)
        if exercise_kind == "American"
        else EuropeanExercise(maturity)
    )
    option = DiscreteAveragingAsianOption(
        average_type,
        1.0 if average_type == AverageType.Geometric else 0.0,
        0,
        [today + 182, maturity],
        _payoff(inputs),
        exercise,
    )
    option.set_pricing_engine(FdBlackScholesAsianEngine(_process(inputs), 20, 20, 10))
    threw, what = _throws(option.npv)
    assert threw
    assert what == expected["what"]


# --- 7. FdBlackScholesBarrierEngine + 8. FdBlackScholesRebateEngine -----------

_BARRIER_TYPES = {
    "DownIn": BarrierType.DownIn,
    "UpIn": BarrierType.UpIn,
    "DownOut": BarrierType.DownOut,
    "UpOut": BarrierType.UpOut,
}

_BARRIER_CASES = [
    "barrier_downout_call",
    "barrier_downout_call_rebate",
    "barrier_upout_put_rebate",
    "barrier_downin_call_rebate",
    "barrier_downin_damped",
    "barrier_upin_put_rebate",
    "barrier_downin_small_grid",
    "barrier_downout_dividends",
]

_REBATE_CASES = ["rebate_downout", "rebate_upout", "rebate_downin"]


def _barrier_option(inputs: dict[str, Any]) -> BarrierOption:
    return BarrierOption(
        _BARRIER_TYPES[inputs["barrier_type"]],
        float(inputs["barrier"]),
        float(inputs["rebate"]),
        _payoff(inputs),
        _exercise(inputs),
    )


@pytest.mark.parametrize("name", _BARRIER_CASES)
def test_fd_black_scholes_barrier_engine(cpp: dict[str, Any], name: str) -> None:
    """Knock-outs directly; knock-ins through the in-out parity assembly."""
    case = cpp[name]
    inputs = case["inputs"]
    option = _barrier_option(inputs)
    option.set_pricing_engine(
        FdBlackScholesBarrierEngine(
            _process(inputs),
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            int(inputs["damping_steps"]),
            _scheme(inputs["scheme"]),
            dividends=_dividends(inputs),
        )
    )
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name=name,
    )


@pytest.mark.parametrize("name", _REBATE_CASES)
def test_fd_black_scholes_rebate_engine(cpp: dict[str, Any], name: str) -> None:
    """The rebate leg standalone: a cash-or-nothing terminal payoff."""
    case = cpp[name]
    inputs = case["inputs"]
    option = _barrier_option(inputs)
    option.set_pricing_engine(
        FdBlackScholesRebateEngine(
            _process(inputs),
            int(inputs["t_grid"]),
            int(inputs["x_grid"]),
            int(inputs["damping_steps"]),
            _scheme(inputs["scheme"]),
        )
    )
    _assert_greeks(
        case["expected"],
        npv=option.npv(),
        delta=option.delta(),
        gamma=option.gamma(),
        theta=option.theta(),
        name=name,
    )


def test_barrier_rebate_actually_paid(cpp: dict[str, Any]) -> None:
    """The rebate is not decoration: adding 3.0 moves the down-and-out call.

    Same market, same grids, rebate 0.0 vs 3.0 — C++ prices the second ~2.0
    higher, which is the rebate discounted by the probability of knocking out.
    """
    no_rebate = float(cpp["barrier_downout_call"]["expected"]["npv"])
    with_rebate = float(cpp["barrier_downout_call_rebate"]["expected"]["npv"])
    assert with_rebate > no_rebate + 1.0
    # And the rebate cannot be worth more than the rebate itself.
    assert with_rebate - no_rebate < 3.0


def test_barrier_damping_steps_reach_the_rebate_leg(cpp: dict[str, Any]) -> None:
    """``min(1, dampingSteps/2)`` on the rebate leg is a real, observable knob."""
    plain = float(cpp["barrier_downin_call_rebate"]["expected"]["npv"])
    damped = float(cpp["barrier_downin_damped"]["expected"]["npv"])
    assert plain != damped


@pytest.mark.parametrize(
    "name",
    [
        "barrier_triggered_rejected",
        "barrier_american_rejected",
        "barrier_zero_strike_rejected",
        "rebate_american_rejected",
    ],
)
def test_barrier_rejections(cpp: dict[str, Any], name: str) -> None:
    """Every QL_REQUIRE the two barrier engines carry, with its exact message."""
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["throws"] is True

    today = _today(inputs)
    maturity = _date("2026-05-15")
    process = _process(inputs)
    strike = float(inputs.get("strike", 100.0))
    payoff = PlainVanillaPayoff(OptionType.Call, strike)
    is_american = inputs.get("exercise") == "American"
    exercise: Exercise = (
        AmericanExercise(today, maturity) if is_american else EuropeanExercise(maturity)
    )
    rebate = 5.0 if name == "rebate_american_rejected" else 0.0
    option = BarrierOption(
        _BARRIER_TYPES[inputs["barrier_type"]],
        float(inputs["barrier"]),
        rebate,
        payoff,
        exercise,
    )
    option.set_pricing_engine(
        FdBlackScholesRebateEngine(process, 20, 20)
        if name == "rebate_american_rejected"
        else FdBlackScholesBarrierEngine(process, 20, 20)
    )
    threw, what = _throws(option.npv)
    assert threw
    assert what == expected["what"]


# --- 9. Fd2dBlackScholesVanillaEngine -----------------------------------------

_FD2D_CASES = [
    "fd2d_max_call_rho0",
    "fd2d_max_call_rho_pos",
    "fd2d_min_put_rho_neg",
    "fd2d_spread_call",
    "fd2d_american_max_put",
    "fd2d_bermudan_avg_call",
]


@pytest.mark.parametrize("name", _FD2D_CASES)
def test_fd_2d_black_scholes_vanilla_engine(cpp: dict[str, Any], name: str) -> None:
    """Two correlated underlyings, default scheme Hundsdorfer.

    ``gamma`` here is ``gammaX + gammaY + 2 gammaXY``, a sum of three
    second-derivative terms of comparable magnitude and opposite sign for the
    spread payoff — so it is asserted at a bound derived from that
    cancellation rather than at TIGHT.
    """
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    p1 = _process(inputs, spot_key="spot1", q_key="dividend_yield1", vol_key="volatility1")
    p2 = _process(inputs, spot_key="spot2", q_key="dividend_yield2", vol_key="volatility2")
    option = BasketOption(
        _basket_payoff(
            inputs["payoff_kind"],
            OptionType.Call if inputs["option_type"] == "Call" else OptionType.Put,
            float(inputs["strike"]),
            2,
        ),
        _exercise(inputs),
    )
    option.set_pricing_engine(
        Fd2dBlackScholesVanillaEngine(
            p1,
            p2,
            float(inputs["correlation"]),
            int(inputs["x_grid"]),
            int(inputs["y_grid"]),
            int(inputs["t_grid"]),
            int(inputs["damping_steps"]),
            _scheme(inputs["scheme"]),
        )
    )
    tolerance.tight(option.npv(), float(expected["npv"]), reason=f"{name}: npv")
    tolerance.tight(option.delta(), float(expected["delta"]), reason=f"{name}: delta")
    # gamma = gammaX + gammaY + 2 gammaXY. Each term is O(1e-2) here, and for
    # the spread payoff they cancel to O(1e-5) — a loss of ~3 decimal digits.
    # A TIGHT 1e-12 relative bound on the terms therefore only guarantees
    # ~1e-12 * 1e-2 = 1e-14 ABSOLUTE on the sum, which is the bound used.
    tolerance.custom(
        option.gamma(),
        float(expected["gamma"]),
        abs_tol=1e-13,
        rel_tol=1e-9,
        reason=f"{name}: gamma is gammaX + gammaY + 2 gammaXY, a cancelling sum "
        "of O(1e-2) terms; 1e-12 relative on the terms is ~1e-14 absolute on "
        "the sum",
    )
    tolerance.custom(
        option.theta(),
        float(expected["theta"]),
        abs_tol=_THETA_ABS,
        rel_tol=_THETA_REL,
        reason=f"{name}: theta — see _THETA_REL",
    )


def test_fd2d_correlation_matters(cpp: dict[str, Any]) -> None:
    """rho = 0 vs rho = 0.6 on the same market: the max-call drops by ~2.9.

    Guards against a port that builds the 2-D operator without the mixed
    second-derivative term.
    """
    rho0 = float(cpp["fd2d_max_call_rho0"]["expected"]["npv"])
    rho_pos = float(cpp["fd2d_max_call_rho_pos"]["expected"]["npv"])
    assert rho0 - rho_pos > 2.0


def test_fd2d_default_scheme_is_hundsdorfer(cpp: dict[str, Any]) -> None:
    """The 2-D engine's default is Hundsdorfer, not Douglas."""
    case = cpp["fd2d_max_call_rho_pos"]
    inputs = case["inputs"]
    assert inputs["scheme"] == "Hundsdorfer"
    p1 = _process(inputs, spot_key="spot1", q_key="dividend_yield1", vol_key="volatility1")
    p2 = _process(inputs, spot_key="spot2", q_key="dividend_yield2", vol_key="volatility2")
    option = BasketOption(
        _basket_payoff(inputs["payoff_kind"], OptionType.Call, float(inputs["strike"]), 2),
        _exercise(inputs),
    )
    option.set_pricing_engine(
        Fd2dBlackScholesVanillaEngine(
            p1,
            p2,
            float(inputs["correlation"]),
            int(inputs["x_grid"]),
            int(inputs["y_grid"]),
            int(inputs["t_grid"]),
            int(inputs["damping_steps"]),
        )
    )
    tolerance.tight(
        option.npv(),
        float(case["expected"]["npv"]),
        reason="default scheme must be Hundsdorfer",
    )


# --- 10. FdndimBlackScholesVanillaEngine --------------------------------------

_FDNDIM_CASES = [
    "fdndim_2d_max_call",
    "fdndim_2d_min_put",
    "fdndim_3d_avg_call",
    "fdndim_2d_american_max_put",
    "fdndim_2d_autoscaled",
]


@pytest.mark.parametrize("name", _FDNDIM_CASES)
def test_fdndim_black_scholes_vanilla_engine(cpp: dict[str, Any], name: str) -> None:
    """n-dimensional PCA-space solver, n = 2 and n = 3.

    C++ fills the value only; the probe pins that ``delta()`` throws.
    """
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    today = _today(inputs)
    dc = Actual365Fixed()
    spots = [float(s) for s in inputs["spots"]]
    processes = [
        BlackScholesMertonProcess(
            x0=SimpleQuote(spots[i]),
            dividend_ts=FlatForward.from_rate(
                reference_date=today,
                forward_rate=float(inputs["dividend_yields"][i]),
                day_counter=dc,
            ),
            risk_free_ts=FlatForward.from_rate(
                reference_date=today,
                forward_rate=float(inputs["risk_free_rate"]),
                day_counter=dc,
            ),
            black_vol_ts=BlackConstantVol(
                reference_date=today,
                calendar=NullCalendar(),
                day_counter=dc,
                volatility=float(inputs["volatilities"][i]),
            ),
        )
        for i in range(len(spots))
    ]
    rho: Matrix = np.array(inputs["rho"], dtype=np.float64)
    x_grids: int | list[int] = (
        int(inputs["auto_x_grid"])
        if inputs["auto_scale_grids"]
        else [int(g) for g in inputs["x_grids"]]
    )
    option = BasketOption(
        _basket_payoff(
            inputs["payoff_kind"],
            OptionType.Call if inputs["option_type"] == "Call" else OptionType.Put,
            float(inputs["strike"]),
            len(spots),
        ),
        _exercise(inputs),
    )
    option.set_pricing_engine(
        FdndimBlackScholesVanillaEngine(
            processes,
            rho,
            x_grids,
            int(inputs["t_grid"]),
            int(inputs["damping_steps"]),
            _scheme(inputs["scheme"]),
        )
    )
    tolerance.tight(option.npv(), float(expected["npv"]), reason=f"{name}: npv")

    assert expected["delta_throws"] is True
    threw, what = _throws(option.delta)
    assert threw
    assert what == expected["delta_what"]


def test_fdndim_three_dimensions_actually_runs(cpp: dict[str, Any]) -> None:
    """The n = 3 case really is three-dimensional."""
    inputs = cpp["fdndim_3d_avg_call"]["inputs"]
    assert len(inputs["spots"]) == 3
    assert len(inputs["x_grids"]) == 3
    assert np.array(inputs["rho"]).shape == (3, 3)


@pytest.mark.parametrize(
    "name",
    ["fdndim_too_many_underlyings", "fdndim_wrong_correlation_size", "fdndim_no_processes"],
)
def test_fdndim_rejections(cpp: dict[str, Any], name: str) -> None:
    """Constructor and ``calculate()`` guards, with the exact C++ messages."""
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["throws"] is True

    today = _today(inputs)
    dc = Actual365Fixed()

    def make(n: int, vol: float = 0.20) -> GeneralizedBlackScholesProcess:
        return BlackScholesMertonProcess(
            x0=SimpleQuote(100.0),
            dividend_ts=FlatForward.from_rate(
                reference_date=today, forward_rate=0.0, day_counter=dc
            ),
            risk_free_ts=FlatForward.from_rate(
                reference_date=today, forward_rate=0.05, day_counter=dc
            ),
            black_vol_ts=BlackConstantVol(
                reference_date=today,
                calendar=NullCalendar(),
                day_counter=dc,
                volatility=vol + 0.01 * n,
            ),
        )

    def run() -> None:
        if name == "fdndim_too_many_underlyings":
            n = int(inputs["n_processes"])
            assert n > PDE_MAX_SUPPORTED_DIM
            rho = np.full((n, n), 0.2)
            np.fill_diagonal(rho, 1.0)
            option = BasketOption(
                _basket_payoff("Max", OptionType.Call, 100.0, n),
                EuropeanExercise(_date("2026-05-15")),
            )
            option.set_pricing_engine(
                FdndimBlackScholesVanillaEngine(
                    [make(i) for i in range(n)], rho, 8, 5
                )
            )
            option.npv()
        elif name == "fdndim_wrong_correlation_size":
            FdndimBlackScholesVanillaEngine(
                [make(0), make(1)], np.zeros((3, 3)), 8, 5
            )
        else:
            FdndimBlackScholesVanillaEngine([], np.zeros((0, 0)), 8, 5)

    threw, what = _throws(run)
    assert threw
    assert what == expected["what"]


def test_fdndim_european_discounts_at_the_end(cpp: dict[str, Any]) -> None:
    """European and American take genuinely different code paths.

    For a ``EuropeanExercise`` the Wiener operator gets a *null* discount curve
    and the solution is multiplied by the discount factor afterwards; for an
    ``AmericanExercise`` the curve goes into the operator. The two prices on
    the same market differ by a factor of ~4, so a port that discounts twice or
    not at all cannot pass both.
    """
    european = float(cpp["fdndim_2d_max_call"]["expected"]["npv"])
    american = float(cpp["fdndim_2d_american_max_put"]["expected"]["npv"])
    assert european > 0.0
    assert american > 0.0
    assert not math.isclose(european, american, rel_tol=0.5)
