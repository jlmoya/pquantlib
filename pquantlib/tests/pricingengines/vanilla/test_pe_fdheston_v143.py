"""Cross-validate the v1.43 finite-difference Heston / Bates / SABR engines.

Reference: ``migration-harness/references/v143/pe/fdheston`` (probe at
``migration-harness/cpp/probes/v143_pe_fdheston/probe.cpp``).

Covers :class:`FdHestonVanillaEngine` / :class:`MakeFdHestonVanillaEngine`,
:class:`FdBatesVanillaEngine`, :class:`FdHestonHullWhiteVanillaEngine`,
:class:`FdSabrVanillaEngine`, :class:`FdHestonBarrierEngine`,
:class:`FdHestonDoubleBarrierEngine` and :class:`FdHestonRebateEngine`. The two
FD swaption engines from the same probe live in
``tests/pricingengines/swaption/test_pe_fdswaption_v143.py``.

Why the assertions are exact
----------------------------
A backward finite-difference rollback is deterministic: given the same mesher
locations, the same operator coefficients and the same ADI splitting it is a
fixed sequence of tridiagonal solves, with no RNG, no clock and no
iterate-to-tolerance step anywhere. A correct port therefore reproduces the C++
value bit-for-bit up to accumulated rounding, so the tier is **TIGHT**
(1e-14 abs / 1e-12 rel) rather than a band. Measured worst case across the
sweep is 3.9e-13 relative, on the variance-mesher-driven grid locations.

``theta`` is the one field that cannot be asserted at TIGHT, and the reason is
cancellation, not sloppiness: it is a *difference quotient*, so its absolute
bound is derived from the case's own magnitudes by :func:`_theta_abs_tol`.
Value, delta and gamma stay at TIGHT everywhere.

Each case carries its whole market description in ``inputs``, so the tests
rebuild the setup rather than restating constants.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.cashflows.dividend import dividend_vector
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, BermudanExercise, EuropeanExercise, Exercise
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_heston_variance_mesher import (
    FdmHestonLocalVolatilityVarianceMesher,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.models.equity.bates_model import BatesModel
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.fd_heston_barrier_engine import FdHestonBarrierEngine
from pquantlib.pricingengines.barrier.fd_heston_double_barrier_engine import (
    FdHestonDoubleBarrierEngine,
)
from pquantlib.pricingengines.barrier.fd_heston_rebate_engine import FdHestonRebateEngine
from pquantlib.pricingengines.vanilla.fd_bates_vanilla_engine import FdBatesVanillaEngine
from pquantlib.pricingengines.vanilla.fd_heston_hull_white_vanilla_engine import (
    FdHestonHullWhiteVanillaEngine,
)
from pquantlib.pricingengines.vanilla.fd_heston_vanilla_engine import (
    FdHestonVanillaEngine,
    MakeFdHestonVanillaEngine,
    process_helper,
)
from pquantlib.pricingengines.vanilla.fd_sabr_vanilla_engine import FdSabrVanillaEngine
from pquantlib.processes.bates_process import BatesProcess
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.processes.hull_white_process import HullWhiteProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# probe.cpp — ``const Date kToday(15, May, 2025);`` and
# ``Settings::instance().evaluationDate() = kToday;`` in ``main()``.
TODAY = Date.from_ymd(15, Month.May, 2025)

#: probe.cpp ``makeExercise("bermudan")`` — kToday + 120 / +240 / +365.
_BERMUDAN_OFFSETS = (120, 240, 365)

#: The time at which every ``Fdm*DimSolver`` takes its theta snapshot —
#: ``snapshot_time`` in
#: ``methods/finitedifferences/solvers/fdm_1dim_solver.py``, i.e. ``0.99/365``.
_THETA_SNAPSHOT_TIME = 0.99 / 365.0

#: Relative rounding carried by one rolled-back grid value, in units of the
#: value itself. A handful of ulps; the values in this file are asserted at
#: TIGHT and are observed to agree to <= 4e-13 relative, so 4e-15 is the
#: conservative *per-end-point* figure that feeds the theta bound below.
_VALUE_ROUNDING = 4e-15


def _theta_abs_tol(value_scale: float) -> float:
    """Absolute bound on an FD ``theta``, derived from the arithmetic.

    ``theta = (V(t0) - V(0)) / t0`` with ``t0 = 0.99/365 = 2.712e-3``. The
    subtraction cancels ``|V|`` down to ``|theta| * t0``, so a relative
    rounding ``u`` on each end value becomes ``2 * u * |V| / t0`` **absolute**
    on theta — independent of how small theta itself is. With
    ``u = _VALUE_ROUNDING`` that is the number returned here.

    ``value_scale`` is the magnitude the difference is formed from: the case's
    own NPV for a single solve, or the sum of the leg magnitudes where the
    engine combines several (see :data:`_IN_BARRIER_THETA_SCALE`).

    This is not a looser *tier* — ``rel_tol`` stays at the TIGHT 1e-12 in every
    call. It replaces TIGHT's fixed 1e-14 absolute floor, which is simply the
    wrong floor for a difference quotient.
    """
    return 2.0 * _VALUE_ROUNDING * abs(value_scale) / _THETA_SNAPSHOT_TIME


#: ``FdHestonBarrierEngine``'s in-barrier branch reports
#: ``vanilla + rebate - knockout``. On this cluster's grid those three legs
#: are worth ~9.5 (``heston_vanilla_call_atm``), ~2.9 (``rebate_downout``, the
#: coarse 25/20/10 grid the engine derives) and ~10.0
#: (``barrier_downout_call_rebate``), so the theta difference is formed from a
#: magnitude of ~22.4, not from the ~2.4 answer. 25 rounds that up.
_IN_BARRIER_THETA_SCALE = 25.0

_SCHEMES = {
    "hundsdorfer": FdmSchemeDesc.hundsdorfer,
    "douglas": FdmSchemeDesc.douglas,
    "craigsneyd": FdmSchemeDesc.craig_sneyd,
}

_BARRIER_TYPES = {
    "DownIn": BarrierType.DownIn,
    "DownOut": BarrierType.DownOut,
    "UpIn": BarrierType.UpIn,
    "UpOut": BarrierType.UpOut,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/fdheston")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


# --- market reconstruction (mirrors probe.cpp ``describeHeston``) ----------


def _day_counter() -> DayCounter:
    return Actual365Fixed()


def _heston_process(inp: dict[str, Any]) -> HestonProcess:
    dc = _day_counter()
    return HestonProcess(
        risk_free_rate=FlatForward.from_rate(TODAY, float(inp["r"]), dc),
        dividend_yield=FlatForward.from_rate(TODAY, float(inp["q"]), dc),
        s0=SimpleQuote(float(inp["s0"])),
        v0=float(inp["v0"]),
        kappa=float(inp["kappa"]),
        theta=float(inp["theta"]),
        sigma=float(inp["sigma"]),
        rho=float(inp["rho"]),
    )


def _heston_model(inp: dict[str, Any]) -> HestonModel:
    return HestonModel(_heston_process(inp))


def _bates_model(inp: dict[str, Any]) -> BatesModel:
    dc = _day_counter()
    return BatesModel(
        BatesProcess(
            risk_free_rate=FlatForward.from_rate(TODAY, float(inp["r"]), dc),
            dividend_yield=FlatForward.from_rate(TODAY, float(inp["q"]), dc),
            s0=SimpleQuote(float(inp["s0"])),
            v0=float(inp["v0"]),
            kappa=float(inp["kappa"]),
            theta=float(inp["theta"]),
            sigma=float(inp["sigma"]),
            rho=float(inp["rho"]),
            lambda_=float(inp["jump_lambda"]),
            nu=float(inp["jump_nu"]),
            delta=float(inp["jump_delta"]),
        )
    )


def _exercise(inp: dict[str, Any]) -> Exercise:
    days = int(inp["expiry_days"])
    kind = str(inp["exercise"])
    if kind == "american":
        return AmericanExercise(TODAY, TODAY + days)
    if kind == "bermudan":
        return BermudanExercise([TODAY + off for off in _BERMUDAN_OFFSETS])
    return EuropeanExercise(TODAY + days)


def _payoff(inp: dict[str, Any]) -> PlainVanillaPayoff:
    option_type = OptionType.Call if inp["option_type"] == "Call" else OptionType.Put
    return PlainVanillaPayoff(option_type, float(inp["strike"]))


def _check_theta(actual: float, expected: float, value_scale: float) -> None:
    custom(
        actual,
        expected,
        abs_tol=_theta_abs_tol(value_scale),
        rel_tol=1e-12,
        reason=(
            f"theta is (V({_THETA_SNAPSHOT_TIME:.6f}) - V(0)) / {_THETA_SNAPSHOT_TIME:.6f}; "
            f"the subtraction cancels a magnitude of {abs(value_scale):.4g} down to "
            f"{abs(expected):.4g}, so a {_VALUE_ROUNDING:.0e} relative rounding on each end "
            "value is an absolute, not a relative, error on theta"
        ),
    )


def _check_greeks(
    opt: OneAssetOption, expected: dict[str, Any], theta_scale: float | None = None
) -> None:
    tight(opt.npv(), float(expected["npv"]))
    tight(opt.delta(), float(expected["delta"]))
    tight(opt.gamma(), float(expected["gamma"]))
    _check_theta(
        opt.theta(),
        float(expected["theta"]),
        float(expected["npv"]) if theta_scale is None else theta_scale,
    )


# --- diagnostics ----------------------------------------------------------


def test_diagnostics_variance_and_equity_mesh(cpp: dict[str, Any]) -> None:
    """The two meshers the Heston engines build, before any rollback.

    A failure here localises the problem to the grid rather than the solver.
    ``helper_*_discount_1y`` pins the curve **swap** inside
    ``FdmBlackScholesMesher::processHelper``: the helper process's dividend
    slot holds the real risk-free curve and vice versa.
    """
    case = cpp["diagnostics"]
    inp, exp = case["inputs"], case["expected"]
    process = _heston_process(inp)
    v_grid = int(inp["v_grid"])

    # probe.cpp diagnosticsCases(): tAvgSteps = max(5, tGrid/50) = 5, eps 1e-4.
    v_mesher = FdmHestonLocalVolatilityVarianceMesher(v_grid, process, None, 1.0, 5, 0.0001, 1.0)
    tight(v_mesher.vola_estimate(), float(exp["vola_estimate"]))
    for i in range(v_grid):
        tight(float(v_mesher.location(i)), float(exp[f"v_loc{i}"]))

    v_mesher_mix = FdmHestonLocalVolatilityVarianceMesher(
        v_grid, process, None, 1.0, 5, 0.0001, 0.7
    )
    tight(v_mesher_mix.vola_estimate(), float(exp["vola_estimate_mix07"]))

    helper = process_helper(
        process.s0(), process.dividend_yield(), process.risk_free_rate(), v_mesher.vola_estimate()
    )
    tight(helper.risk_free_rate().discount(1.0), float(exp["helper_riskfree_discount_1y"]))
    tight(helper.dividend_yield().discount(1.0), float(exp["helper_dividend_discount_1y"]))

    x_grid = int(inp["x_grid"])
    equity = FdmBlackScholesMesher(
        x_grid, helper, 1.0, 100.0, None, None, 0.0001, 2.0, (100.0, 0.1)
    )
    for i in range(x_grid):
        tight(float(equity.location(i)), float(exp[f"x_loc{i}"]))


# --- FdHestonVanillaEngine ------------------------------------------------

_VANILLA_CASES = [
    "heston_vanilla_call_atm",
    "heston_vanilla_put_atm",
    "heston_vanilla_call_otm",
    "heston_vanilla_put_itm",
    "heston_vanilla_call_damped_douglas",
    "heston_vanilla_call_craigsneyd",
    "heston_vanilla_put_american",
    "heston_vanilla_call_american",
    "heston_vanilla_put_bermudan",
    "heston_vanilla_call_mixing07",
]


@pytest.mark.parametrize("case", _VANILLA_CASES)
def test_fd_heston_vanilla_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    engine = FdHestonVanillaEngine(
        _heston_model(inp),
        t_grid=int(inp["t_grid"]),
        x_grid=int(inp["x_grid"]),
        v_grid=int(inp["v_grid"]),
        damping_steps=int(inp["damping_steps"]),
        scheme_desc=_SCHEMES[str(inp["scheme"])](),
        mixing_factor=float(inp["mixing_factor"]),
    )
    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(engine)
    _check_greeks(opt, exp)


def test_american_premium_is_visible(cpp: dict[str, Any]) -> None:
    """The American branch of ``vanillaComposite`` really fires.

    The put's early-exercise premium is 0.26 on a 6.57 European, i.e. 4%: an
    engine that silently dropped the step condition would land on the European
    number, which is pinned in the same file.
    """
    european = float(cpp["heston_vanilla_put_atm"]["expected"]["npv"])
    american = float(cpp["heston_vanilla_put_american"]["expected"]["npv"])
    assert american > european
    assert american - european > 0.2


def test_multiple_strikes_cache(cpp: dict[str, Any]) -> None:
    """One solve on the multi-strike mesh, then two pure cache hits.

    The cached entries are read off the same surface at a moneyness-scaled
    spot (``value/d``, ``delta``, ``gamma*d``, ``theta/d``), so an engine that
    re-solved per strike would land on different numbers.
    """
    case = "heston_ms_cache"
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    engine = FdHestonVanillaEngine(
        _heston_model(inp),
        t_grid=int(inp["t_grid"]),
        x_grid=int(inp["x_grid"]),
        v_grid=int(inp["v_grid"]),
        damping_steps=int(inp["damping_steps"]),
        scheme_desc=_SCHEMES[str(inp["scheme"])](),
    )
    strikes = [float(inp[f"strike_{i}"]) for i in range(3)]
    engine.enable_multiple_strikes_caching(strikes)
    assert engine.strikes() == strikes

    # probe.cpp shares one Exercise object across the three options; that is
    # what makes the cache key (type + dates) match.
    exercise = EuropeanExercise(TODAY + int(inp["expiry_days"]))

    for tag, strike in (("direct_100", 100.0), ("cached_80", 80.0), ("cached_120", 120.0)):
        opt = VanillaOption(PlainVanillaPayoff(OptionType.Call, strike), exercise)
        opt.set_pricing_engine(engine)
        tight(opt.npv(), float(exp[f"npv_{tag}"]))
        tight(opt.delta(), float(exp[f"delta_{tag}"]))
        tight(opt.gamma(), float(exp[f"gamma_{tag}"]))
        _check_theta(opt.theta(), float(exp[f"theta_{tag}"]), float(exp[f"npv_{tag}"]))

    # A strike that is not in the cached vector falls through to a full solve.
    miss_exp = cpp["heston_ms_cache_miss"]["expected"]
    opt90 = VanillaOption(PlainVanillaPayoff(OptionType.Call, 90.0), exercise)
    opt90.set_pricing_engine(engine)
    tight(opt90.npv(), float(miss_exp["npv_miss_90"]))
    tight(opt90.delta(), float(miss_exp["delta_miss_90"]))
    tight(opt90.gamma(), float(miss_exp["gamma_miss_90"]))
    _check_theta(
        opt90.theta(), float(miss_exp["theta_miss_90"]), float(miss_exp["npv_miss_90"])
    )


def test_multiple_strikes_mesher_switch(cpp: dict[str, Any]) -> None:
    """Enabling caching changes the equity mesher, hence the strike-100 value.

    ``heston_vanilla_call_atm`` and ``heston_ms_cache.npv_direct_100`` are the
    same option on the same model with the same grid sizes; they differ only
    because the second uses ``FdmBlackScholesMultiStrikeMesher``.
    """
    single = float(cpp["heston_vanilla_call_atm"]["expected"]["npv"])
    multi = float(cpp["heston_ms_cache"]["expected"]["npv_direct_100"])
    assert single != multi
    assert abs(single - multi) > 1e-3


def test_update_clears_the_cache(cpp: dict[str, Any]) -> None:
    """# C++ parity: ``FdHestonVanillaEngine::update`` clears
    ``cachedArgs2results_`` before delegating.

    Observable without touching private state: with the cache warm, strike 80
    is answered by moneyness-scaling the strike-100 surface (the pinned
    ``npv_cached_80``); after ``update()`` the same strike triggers a real
    solve of the strike-80 payoff, which is a different number.
    """
    case = cpp["heston_ms_cache"]
    inp, exp = case["inputs"], case["expected"]
    engine = FdHestonVanillaEngine(
        _heston_model(inp),
        t_grid=int(inp["t_grid"]),
        x_grid=int(inp["x_grid"]),
        v_grid=int(inp["v_grid"]),
        damping_steps=int(inp["damping_steps"]),
    )
    engine.enable_multiple_strikes_caching([float(inp[f"strike_{i}"]) for i in range(3)])
    exercise = EuropeanExercise(TODAY + int(inp["expiry_days"]))

    warm = VanillaOption(PlainVanillaPayoff(OptionType.Call, 100.0), exercise)
    warm.set_pricing_engine(engine)
    warm.npv()

    cached = VanillaOption(PlainVanillaPayoff(OptionType.Call, 80.0), exercise)
    cached.set_pricing_engine(engine)
    tight(cached.npv(), float(exp["npv_cached_80"]))

    engine.update()

    resolved = VanillaOption(PlainVanillaPayoff(OptionType.Call, 80.0), exercise)
    resolved.set_pricing_engine(engine)
    assert abs(resolved.npv() - float(exp["npv_cached_80"])) > 1e-4


# --- MakeFdHestonVanillaEngine --------------------------------------------


def test_make_defaults(cpp: dict[str, Any]) -> None:
    """The builder's untouched defaults: 100 x 100 x 50, no damping, Hundsdorfer."""
    case = "make_defaults"
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    assert (int(inp["t_grid"]), int(inp["x_grid"]), int(inp["v_grid"])) == (100, 100, 50)
    assert int(inp["damping_steps"]) == 0
    assert inp["scheme"] == "hundsdorfer"

    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(MakeFdHestonVanillaEngine(_heston_model(inp)).engine())
    _check_greeks(opt, exp)


def test_make_all_withs(cpp: dict[str, Any]) -> None:
    """Every ``with*`` knob set; must equal the directly constructed engine."""
    case = "make_all_withs"
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    built = (
        MakeFdHestonVanillaEngine(_heston_model(inp))
        .with_t_grid(int(inp["t_grid"]))
        .with_x_grid(int(inp["x_grid"]))
        .with_v_grid(int(inp["v_grid"]))
        .with_damping_steps(int(inp["damping_steps"]))
        .with_fdm_scheme_desc(_SCHEMES[str(inp["scheme"])]())
        .engine()
    )
    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(built)
    _check_greeks(opt, exp)

    # The same numbers out of the plain constructor.
    direct = FdHestonVanillaEngine(
        _heston_model(inp),
        t_grid=int(inp["t_grid"]),
        x_grid=int(inp["x_grid"]),
        v_grid=int(inp["v_grid"]),
        damping_steps=int(inp["damping_steps"]),
        scheme_desc=_SCHEMES[str(inp["scheme"])](),
    )
    opt2 = VanillaOption(_payoff(inp), _exercise(inp))
    opt2.set_pricing_engine(direct)
    tight(opt2.npv(), float(exp["npv"]))


def _dividend_dates_and_amounts(inp: dict[str, Any]) -> tuple[list[Date], list[float]]:
    return (
        [TODAY + int(inp["dividend_1_days"]), TODAY + int(inp["dividend_2_days"])],
        [float(inp["dividend_1_amount"]), float(inp["dividend_2_amount"])],
    )


def test_fd_heston_vanilla_with_cash_dividends(cpp: dict[str, Any]) -> None:
    """A discrete-dividend schedule reaches both the mesh and the step conditions.

    ``FdmBlackScholesMesher`` gets one extra intermediate step per dividend for
    its forward walk, and ``vanillaComposite`` installs an
    ``FdmDividendHandler`` plus *two* copies of each dividend time in the
    stopping times (the second shifted by +1e-5), which also moves the solver's
    time grid. Dropping either half changes the number.
    """
    case = "heston_vanilla_call_dividends"
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    dates, amounts = _dividend_dates_and_amounts(inp)
    engine = FdHestonVanillaEngine(
        _heston_model(inp),
        dividend_vector(dates, amounts),
        None,
        int(inp["t_grid"]),
        int(inp["x_grid"]),
        int(inp["v_grid"]),
        int(inp["damping_steps"]),
        _SCHEMES[str(inp["scheme"])](),
    )
    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(engine)
    _check_greeks(opt, exp)

    # 3.5 of dividends off a 9.52 no-dividend call is a 19% move, so this is
    # not a case a port could match by ignoring the schedule.
    assert float(cpp["heston_vanilla_call_atm"]["expected"]["npv"]) - float(exp["npv"]) > 1.5


def test_make_with_cash_dividends(cpp: dict[str, Any]) -> None:
    """``withCashDividends`` builds a ``DividendVector`` and prices with it.

    # C++ parity: ``dividends_ = DividendVector(dividendDates, dividendAmounts)``.
    """
    case = "make_with_cash_dividends"
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    dates, amounts = _dividend_dates_and_amounts(inp)
    engine = (
        MakeFdHestonVanillaEngine(_heston_model(inp))
        .with_t_grid(int(inp["t_grid"]))
        .with_x_grid(int(inp["x_grid"]))
        .with_v_grid(int(inp["v_grid"]))
        .with_cash_dividends(dates, amounts)
        .engine()
    )
    assert [d.date() for d in engine.dividends()] == dates
    assert [d.amount() for d in engine.dividends()] == amounts

    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(engine)
    _check_greeks(opt, exp)
    # Same numbers as the directly constructed engine.
    tight(
        float(exp["npv"]), float(cpp["heston_vanilla_call_dividends"]["expected"]["npv"])
    )


def test_multiple_strikes_rejects_dividends(cpp: dict[str, Any]) -> None:
    """# C++ parity: the ``QL_REQUIRE`` in ``getSolverDesc``'s multi-strike branch."""
    case = "heston_ms_throw_dividends"
    exp = cpp[case]["expected"]
    assert exp["throws"] is True
    inp = cpp["heston_vanilla_call_dividends"]["inputs"]
    dates, amounts = _dividend_dates_and_amounts(inp)
    engine = FdHestonVanillaEngine(
        _heston_model(inp),
        dividend_vector(dates, amounts),
        None,
        int(inp["t_grid"]),
        int(inp["x_grid"]),
        int(inp["v_grid"]),
    )
    engine.enable_multiple_strikes_caching([80.0, 100.0, 120.0])
    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match=exp["what"]):
        opt.npv()


def test_fd_heston_vanilla_with_quanto_helper(cpp: dict[str, Any]) -> None:
    """The quanto helper reaches both the mesher and the operator.

    ``FdmBlackScholesMesher`` swaps the dividend curve for a
    ``QuantoTermStructure``, and ``FdmHestonSolver`` forwards the same helper
    to ``FdmHestonOp``, which adds the quanto drift adjustment. A port that
    wires only one of the two lands somewhere else.
    """
    case = "heston_vanilla_call_quanto"
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    dc = _day_counter()
    quanto_helper = FdmQuantoHelper(
        FlatForward.from_rate(TODAY, float(inp["r"]), dc),
        FlatForward.from_rate(TODAY, float(inp["quanto_foreign_rate"]), dc),
        BlackConstantVol(
            reference_date=TODAY,
            calendar=NullCalendar(),
            volatility=float(inp["quanto_fx_vol"]),
            day_counter=dc,
        ),
        float(inp["quanto_equity_fx_correlation"]),
        float(inp["quanto_exch_rate_atm_level"]),
    )
    engine = FdHestonVanillaEngine(
        _heston_model(inp),
        None,
        quanto_helper,
        int(inp["t_grid"]),
        int(inp["x_grid"]),
        int(inp["v_grid"]),
        int(inp["damping_steps"]),
        _SCHEMES[str(inp["scheme"])](),
    )
    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(engine)
    _check_greeks(opt, exp)
    # The quanto adjustment is worth 1.93 off the 9.52 unadjusted call.
    assert float(cpp["heston_vanilla_call_atm"]["expected"]["npv"]) - float(exp["npv"]) > 1.5


def test_make_builder_has_no_validation(cpp: dict[str, Any]) -> None:
    """# C++ parity: ``MakeFdHestonVanillaEngine`` contains no ``QL_REQUIRE``.

    Every ``with*`` assigns and returns ``*this``, and the conversion operator
    always succeeds — so a zero grid is accepted by the builder and only fails
    later, inside the mesher.
    """
    inp = cpp["make_all_withs"]["inputs"]
    builder = MakeFdHestonVanillaEngine(_heston_model(inp))
    assert builder.with_t_grid(0) is builder
    assert builder.with_x_grid(0) is builder
    assert builder.with_v_grid(0) is builder
    assert builder.with_damping_steps(0) is builder
    assert builder.with_fdm_scheme_desc(FdmSchemeDesc.douglas()) is builder
    assert builder.engine() is not None


# --- FdBatesVanillaEngine -------------------------------------------------

_BATES_CASES = [
    "bates_call_atm",
    "bates_put_atm",
    "bates_call_atm_lambda05",
    "bates_call_otm_damped",
]


@pytest.mark.parametrize("case", _BATES_CASES)
def test_fd_bates_vanilla_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    engine = FdBatesVanillaEngine(
        _bates_model(inp),
        None,
        int(inp["t_grid"]),
        int(inp["x_grid"]),
        int(inp["v_grid"]),
        int(inp["damping_steps"]),
        _SCHEMES[str(inp["scheme"])](),
    )
    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(engine)
    _check_greeks(opt, exp)


def test_bates_jump_intensity_moves_the_price(cpp: dict[str, Any]) -> None:
    """The jump integral is actually integrated.

    ``bates_call_atm`` (lambda 0.2) and ``bates_call_atm_lambda05``
    (lambda 0.5) differ in nothing else; an engine that dropped
    ``FdmBatesOp``'s integro term would return the same number for both.
    """
    low = float(cpp["bates_call_atm"]["expected"]["npv"])
    high = float(cpp["bates_call_atm_lambda05"]["expected"]["npv"])
    assert high - low > 0.7


# --- FdHestonHullWhiteVanillaEngine ---------------------------------------

_HHW_CASES = [
    "hhw_call_atm_cv",
    "hhw_call_atm_nocv",
    "hhw_put_atm_nocv",
    "hhw_call_otm_cv_damped",
]


@pytest.mark.parametrize("case", _HHW_CASES)
def test_fd_heston_hull_white_vanilla_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    hw_process = HullWhiteProcess(
        FlatForward.from_rate(TODAY, float(inp["r"]), _day_counter()),
        float(inp["hw_a"]),
        float(inp["hw_sigma"]),
    )
    engine = FdHestonHullWhiteVanillaEngine(
        _heston_model(inp),
        hw_process,
        float(inp["corr_equity_short_rate"]),
        None,
        int(inp["t_grid"]),
        int(inp["x_grid"]),
        int(inp["v_grid"]),
        int(inp["r_grid"]),
        int(inp["damping_steps"]),
        bool(inp["control_variate"]),
        _SCHEMES[str(inp["scheme"])](),
    )
    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(engine)
    _check_greeks(opt, exp)


def test_hhw_control_variate_changes_only_the_value(cpp: dict[str, Any]) -> None:
    """The control variate shifts the NPV and leaves the greeks alone.

    ``calculate()`` adds ``analyticNPV - fdNPV`` to ``results_.value`` only,
    after delta/gamma/theta have been written — so an engine that ignored the
    flag would match the greeks but miss the value by 0.23 here.
    """
    cv = cpp["hhw_call_atm_cv"]["expected"]
    nocv = cpp["hhw_call_atm_nocv"]["expected"]
    assert abs(float(cv["npv"]) - float(nocv["npv"])) > 0.2
    for greek in ("delta", "gamma", "theta"):
        assert float(cv[greek]) == float(nocv[greek])


# --- FdSabrVanillaEngine --------------------------------------------------

_SABR_CASES = [
    "sabr_call_atm",
    "sabr_put_atm",
    "sabr_call_otm_damped",
    "sabr_call_atm_scaled",
]


@pytest.mark.parametrize("case", _SABR_CASES)
def test_fd_sabr_vanilla_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    engine = FdSabrVanillaEngine(
        float(inp["f0"]),
        float(inp["alpha"]),
        float(inp["beta"]),
        float(inp["nu"]),
        float(inp["rho"]),
        FlatForward.from_rate(TODAY, float(inp["r"]), _day_counter()),
        int(inp["t_grid"]),
        int(inp["f_grid"]),
        int(inp["x_grid"]),
        int(inp["damping_steps"]),
        float(inp["scaling_factor"]),
        float(inp["eps"]),
        _SCHEMES[str(inp["scheme"])](),
    )
    opt = VanillaOption(_payoff(inp), _exercise(inp))
    opt.set_pricing_engine(engine)
    tight(opt.npv(), float(exp["npv"]))
    # C++ fills only ``results_.value``; the greeks stay Null.
    with pytest.raises(LibraryException, match="delta not provided"):
        opt.delta()


def test_sabr_throws_on_beta_one(cpp: dict[str, Any]) -> None:
    exp = cpp["sabr_throw_beta_one"]["expected"]
    assert exp["throws"] is True
    r_ts = FlatForward.from_rate(TODAY, 0.05, _day_counter())
    with pytest.raises(LibraryException) as excinfo:
        FdSabrVanillaEngine(100.0, 0.35, 1.0, 0.5, -0.4, r_ts)
    assert str(excinfo.value) == exp["what"]


def test_sabr_throws_on_negative_alpha(cpp: dict[str, Any]) -> None:
    exp = cpp["sabr_throw_negative_alpha"]["expected"]
    assert exp["throws"] is True
    r_ts = FlatForward.from_rate(TODAY, 0.05, _day_counter())
    with pytest.raises(LibraryException) as excinfo:
        FdSabrVanillaEngine(100.0, -0.1, 0.8, 0.5, -0.4, r_ts)
    assert str(excinfo.value) == exp["what"]


def test_sabr_throws_on_unit_rho(cpp: dict[str, Any]) -> None:
    """``validateSabrParameters`` rejects ``rho*rho >= 1``.

    ALIGN NEEDED (reported to the controller): C++ streams ``Real(1.0)`` with
    the default ``std::ostream`` precision, producing ``"1"``; pquantlib's
    ``validate_sabr_parameters``
    (``math/interpolations/sabr_formula.py``) f-strings it, producing
    ``"1.0"``. Both spellings are asserted below so the divergence is pinned,
    not hidden — the day the formatter is fixed this test fails and can be
    collapsed to the C++ string.
    """
    exp = cpp["sabr_throw_rho_one"]["expected"]
    assert exp["throws"] is True
    assert exp["what"] == "rho square must be less than one: 1 not allowed"
    r_ts = FlatForward.from_rate(TODAY, 0.05, _day_counter())
    with pytest.raises(LibraryException) as excinfo:
        FdSabrVanillaEngine(100.0, 0.35, 0.8, 0.5, 1.0, r_ts)
    assert str(excinfo.value) == "rho square must be less than one: 1.0 not allowed"


def test_sabr_beta_is_not_range_checked_by_validate(cpp: dict[str, Any]) -> None:
    """# C++ parity: ``validateSabrParameters(alpha, 0.5, nu, rho)``.

    The second argument is the literal ``0.5``, not ``beta``, so a negative
    beta passes validation and only the separate ``beta < 1.0`` requirement
    applies. A port that passed the real beta through would raise here.
    """
    inp = cpp["sabr_call_atm"]["inputs"]
    r_ts = FlatForward.from_rate(TODAY, float(inp["r"]), _day_counter())
    engine = FdSabrVanillaEngine(
        float(inp["f0"]), float(inp["alpha"]), -0.25, float(inp["nu"]), float(inp["rho"]), r_ts
    )
    assert engine is not None


# --- FdHestonBarrierEngine ------------------------------------------------

_BARRIER_CASES = [
    "barrier_downout_call",
    "barrier_downout_call_rebate",
    "barrier_upout_call_rebate",
    "barrier_upout_put",
    "barrier_downin_call_rebate",
    "barrier_upin_call_rebate_damped",
    "barrier_downin_put",
    "barrier_downout_call_dividends",
]


@pytest.mark.parametrize("case", _BARRIER_CASES)
def test_fd_heston_barrier_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    # ``barrier_downout_call_dividends`` exercises the engine's *own*
    # hand-rolled composite: an FdmDividendHandler plus one (not two, as in
    # vanillaComposite) copy of each dividend time in the stopping times.
    dividends = None
    if "dividend_1_days" in inp:
        dates, amounts = _dividend_dates_and_amounts(inp)
        dividends = dividend_vector(dates, amounts)
    engine = FdHestonBarrierEngine(
        _heston_model(inp),
        dividends,
        int(inp["t_grid"]),
        int(inp["x_grid"]),
        int(inp["v_grid"]),
        int(inp["damping_steps"]),
        _SCHEMES[str(inp["scheme"])](),
    )
    opt = BarrierOption(
        _BARRIER_TYPES[str(inp["barrier_type"])],
        float(inp["barrier"]),
        float(inp["rebate"]),
        _payoff(inp),
        _exercise(inp),
    )
    opt.set_pricing_engine(engine)
    # In-barriers combine three separate solves, so theta's cancellation is
    # measured against their summed magnitude rather than the answer.
    in_barrier = str(inp["barrier_type"]) in ("DownIn", "UpIn")
    _check_greeks(opt, exp, _IN_BARRIER_THETA_SCALE if in_barrier else None)


def test_barrier_rebate_reaches_the_dirichlet_face(cpp: dict[str, Any]) -> None:
    """A non-zero rebate is worth 0.84 on the down-and-out call.

    ``barrier_downout_call`` and ``barrier_downout_call_rebate`` differ only in
    ``arguments_.rebate``, which enters solely through the
    ``FdmDirichletBoundary`` value. An engine that installed the boundary at
    zero would return the same number for both.
    """
    no_rebate = float(cpp["barrier_downout_call"]["expected"]["npv"])
    with_rebate = float(cpp["barrier_downout_call_rebate"]["expected"]["npv"])
    assert with_rebate - no_rebate > 0.8


def test_barrier_throws_on_american_exercise(cpp: dict[str, Any]) -> None:
    exp = cpp["barrier_throw_american"]["expected"]
    assert exp["throws"] is True
    inp = cpp["barrier_downout_call"]["inputs"]
    engine = FdHestonBarrierEngine(_heston_model(inp), None, 25, 30, 12, 0)
    opt = BarrierOption(
        BarrierType.DownOut,
        80.0,
        0.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        AmericanExercise(TODAY, TODAY + int(inp["expiry_days"])),
    )
    opt.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match=exp["what"]):
        opt.npv()


# --- FdHestonRebateEngine -------------------------------------------------

_REBATE_CASES = ["rebate_downout", "rebate_upout", "rebate_downin_damped"]


@pytest.mark.parametrize("case", _REBATE_CASES)
def test_fd_heston_rebate_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    engine = FdHestonRebateEngine(
        _heston_model(inp),
        None,
        int(inp["t_grid"]),
        int(inp["x_grid"]),
        int(inp["v_grid"]),
        int(inp["damping_steps"]),
        _SCHEMES[str(inp["scheme"])](),
    )
    opt = BarrierOption(
        _BARRIER_TYPES[str(inp["barrier_type"])],
        float(inp["barrier"]),
        float(inp["rebate"]),
        _payoff(inp),
        _exercise(inp),
    )
    opt.set_pricing_engine(engine)
    _check_greeks(opt, exp)


def test_rebate_value_is_below_the_undiscounted_rebate(cpp: dict[str, Any]) -> None:
    """The rebate leg prices a cash-or-nothing call struck at zero.

    It must be worth less than the rebate itself (discounting plus the chance
    of touching the barrier), which is what distinguishes it from an engine
    that simply returned ``arguments_.rebate``.
    """
    for case in _REBATE_CASES:
        rebate = float(cpp[case]["inputs"]["rebate"])
        npv = float(cpp[case]["expected"]["npv"])
        assert 0.0 < npv < rebate


def test_rebate_throws_on_american_exercise(cpp: dict[str, Any]) -> None:
    exp = cpp["rebate_throw_american"]["expected"]
    assert exp["throws"] is True
    inp = cpp["rebate_downout"]["inputs"]
    engine = FdHestonRebateEngine(_heston_model(inp), None, 25, 20, 10, 0)
    opt = BarrierOption(
        BarrierType.DownOut,
        80.0,
        3.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        AmericanExercise(TODAY, TODAY + int(inp["expiry_days"])),
    )
    opt.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match=exp["what"]):
        opt.npv()


# --- FdHestonDoubleBarrierEngine ------------------------------------------

_DOUBLE_BARRIER_CASES = [
    "double_barrier_ko_call",
    "double_barrier_ko_call_rebate",
    "double_barrier_ko_put_damped",
]


@pytest.mark.parametrize("case", _DOUBLE_BARRIER_CASES)
def test_fd_heston_double_barrier_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    engine = FdHestonDoubleBarrierEngine(
        _heston_model(inp),
        int(inp["t_grid"]),
        int(inp["x_grid"]),
        int(inp["v_grid"]),
        int(inp["damping_steps"]),
        _SCHEMES[str(inp["scheme"])](),
    )
    opt = DoubleBarrierOption(
        DoubleBarrierType.KnockOut,
        float(inp["barrier_lo"]),
        float(inp["barrier_hi"]),
        float(inp["rebate"]),
        _payoff(inp),
        _exercise(inp),
    )
    opt.set_pricing_engine(engine)
    _check_greeks(opt, exp)


def test_double_barrier_both_faces_carry_the_rebate(cpp: dict[str, Any]) -> None:
    """A rebate on a *double* knock-out is worth more than on a single one.

    Both Dirichlet faces are pinned to ``arguments_.rebate``, so the 2.0 rebate
    adds 0.86 here — an engine that installed only one face would add roughly
    half of that.
    """
    no_rebate = float(cpp["double_barrier_ko_call"]["expected"]["npv"])
    with_rebate = float(cpp["double_barrier_ko_call_rebate"]["expected"]["npv"])
    assert with_rebate - no_rebate > 0.8


def test_double_barrier_throws_on_knock_in(cpp: dict[str, Any]) -> None:
    exp = cpp["double_barrier_throw_knockin"]["expected"]
    assert exp["throws"] is True
    inp = cpp["double_barrier_ko_call"]["inputs"]
    engine = FdHestonDoubleBarrierEngine(_heston_model(inp), 25, 30, 12, 0)
    opt = DoubleBarrierOption(
        DoubleBarrierType.KnockIn,
        80.0,
        130.0,
        0.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        _exercise(inp),
    )
    opt.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match=exp["what"]):
        opt.npv()


def test_double_barrier_throws_on_american_exercise(cpp: dict[str, Any]) -> None:
    exp = cpp["double_barrier_throw_american"]["expected"]
    assert exp["throws"] is True
    inp = cpp["double_barrier_ko_call"]["inputs"]
    engine = FdHestonDoubleBarrierEngine(_heston_model(inp), 25, 30, 12, 0)
    opt = DoubleBarrierOption(
        DoubleBarrierType.KnockOut,
        80.0,
        130.0,
        0.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        AmericanExercise(TODAY, TODAY + int(inp["expiry_days"])),
    )
    opt.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match=exp["what"]):
        opt.npv()


# --- market-setup guard ---------------------------------------------------


def test_market_setup_matches_probe(cpp: dict[str, Any]) -> None:
    """Every constant this file rebuilds is the one the probe recorded."""
    inp = cpp["heston_vanilla_call_atm"]["inputs"]
    assert (int(inp["eval_year"]), int(inp["eval_month"]), int(inp["eval_day"])) == (
        TODAY.year(),
        int(TODAY.month()),
        TODAY.day_of_month(),
    )
    assert inp["day_counter"] == "Actual365Fixed"
    assert inp["calendar"] == "NullCalendar"
    assert int(inp["expiry_days"]) == 365
    # 15 May 2025 -> 15 May 2026 is exactly one Act/365F year.
    assert math.isclose(
        _day_counter().year_fraction(TODAY, TODAY + int(inp["expiry_days"])), 1.0
    )
