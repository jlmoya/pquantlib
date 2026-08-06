"""Cross-validate the quanto instrument family against C++ QuantLib v1.43.

Probe: ``v143/inst/quanto``.

Covers:

* ``QuantoVanillaOption``        (ql/instruments/quantovanillaoption.hpp)
* ``QuantoForwardVanillaOption`` (ql/instruments/quantoforwardvanillaoption.hpp)
* ``QuantoBarrierOption``        (ql/instruments/quantobarrieroption.hpp)
* ``QuantoOptionResults``        (the results carrier they share)
* ``QuantoEngine``               (ql/pricingengines/quanto/quantoengine.hpp),
  in all three of its concrete instantiations, plus the
  ``ForwardVanillaEngine`` one of them wraps.

The three inputs that exist *only* to produce the quanto effect — the
foreign risk-free curve, the exchange-rate volatility and the correlation —
are each swept independently, so an engine that accepted any of them and
dropped it fails here. Correlation = 0 and a negative correlation are both
in the grid; note that at correlation = 0 the FX vol drops out of the NPV
but still moves ``vega`` and ``qlambda``, which is why the greeks are pinned
too and not just the price.

Tolerance: TIGHT throughout — every quantity is a short closed-form chain
over identical inputs (Black-Scholes / Reiner-Rubinstein plus a handful of
curve lookups), and the observed disagreement with C++ is ~1e-15 relative.
The one exception is documented at :func:`_check_value`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.barrier_option import BarrierOptionArguments, BarrierType
from pquantlib.instruments.forward_vanilla_option import ForwardOptionArguments
from pquantlib.instruments.quanto_barrier_option import QuantoBarrierOption
from pquantlib.instruments.quanto_forward_vanilla_option import (
    QuantoForwardVanillaOption,
)
from pquantlib.instruments.quanto_vanilla_option import (
    QuantoOptionResults,
    QuantoVanillaOption,
)
from pquantlib.option import OptionArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.analytic_barrier_engine import (
    AnalyticBarrierEngine,
)
from pquantlib.pricingengines.forward.forward_vanilla_engine import ForwardVanillaEngine
from pquantlib.pricingengines.quanto.quanto_engine import QuantoEngine
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
from pquantlib.termstructures.volatility.equity_fx.implied_vol_term_structure import (
    ImpliedVolTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date

GREEKS = ("npv", "delta", "gamma", "theta", "vega", "rho", "dividend_rho")
QUANTO_GREEKS = ("qvega", "qrho", "qlambda")
ALL_RESULTS = GREEKS + QUANTO_GREEKS

# Any option that exposes both the standard and the quanto greeks.
type QuantoOption = QuantoVanillaOption | QuantoForwardVanillaOption | QuantoBarrierOption


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return load_reference("v143/inst/quanto")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date(cpp: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the global evaluation date to the probe's ``today``, then restore.

    The probe runs with ``Settings::instance().evaluationDate() = kToday``.
    Every curve here is fixed-reference so the prices do not depend on it,
    but ``ForwardOptionArguments::validate`` compares the reset date against
    it — without the pin that branch is silently skipped.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Date(int(cpp["market"]["today_serial"]))
    try:
        yield
    finally:
        settings.evaluation_date = previous


# --------------------------------------------------------------------------
# Market, rebuilt from the probe's own literals
# --------------------------------------------------------------------------


def _market(cpp: dict[str, Any]) -> dict[str, Any]:
    m = cpp["market"]
    return {
        "today": Date(int(m["today_serial"])),
        "maturity": Date(int(m["maturity_serial"])),
        "spot": float(m["spot"]),
        "q": float(m["dividend_rate"]),
        "r": float(m["risk_free_rate"]),
        "vol": float(m["volatility"]),
    }


def _process(mkt: dict[str, Any]) -> GeneralizedBlackScholesProcess:
    dc = Actual360()
    today: Date = mkt["today"]
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(mkt["spot"]),
        dividend_ts=FlatForward.from_rate(today, mkt["q"], dc),
        risk_free_ts=FlatForward.from_rate(today, mkt["r"], dc),
        black_vol_ts=BlackConstantVol(
            reference_date=today, calendar=NullCalendar(), day_counter=dc, volatility=mkt["vol"]
        ),
    )


def _quanto_inputs(
    mkt: dict[str, Any], case: dict[str, Any]
) -> tuple[FlatForward, BlackConstantVol, SimpleQuote]:
    """The three quanto-only engine inputs, straight from the case row."""
    dc = Actual360()
    today: Date = mkt["today"]
    return (
        FlatForward.from_rate(today, float(case["fx_rate"]), dc),
        BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=dc,
            volatility=float(case["fx_vol"]),
        ),
        SimpleQuote(float(case["correlation"])),
    )


# --------------------------------------------------------------------------
# Result comparison
# --------------------------------------------------------------------------


def _check_value(name: str, actual: float, expected: float) -> None:
    if abs(expected) < 1.0e-10:
        # Knocked-out barriers price as a cancelling sum of six terms whose
        # individual magnitudes are of order S=100, so the absolute rounding
        # floor is ~100 * 6 * 2^-53 ~= 7e-14 — larger than the TIGHT abs_tol
        # and unrelated to it, since the exact answer is 0.
        custom(
            actual,
            expected,
            abs_tol=1.0e-12,
            rel_tol=0.0,
            reason=(
                f"{name} is an exact zero reached by cancelling ~1e2-magnitude "
                "terms; bound is the 100*6*2^-53 rounding floor, not a relaxation"
            ),
        )
    else:
        tight(actual, expected, reason=name)


def _check_results(option: QuantoOption, case: dict[str, Any]) -> None:
    """Compare every pinned result, including the ``null`` (must-raise) ones."""
    accessors: dict[str, Callable[[], float]] = {
        "npv": option.npv,
        "delta": option.delta,
        "gamma": option.gamma,
        "theta": option.theta,
        "vega": option.vega,
        "rho": option.rho,
        "dividend_rho": option.dividend_rho,
        "qvega": option.qvega,
        "qrho": option.qrho,
        "qlambda": option.qlambda,
    }
    for name in ALL_RESULTS:
        expected = case[name]
        if expected is None:
            # C++ left the field at Null<Real>(); the accessor must refuse.
            with pytest.raises(LibraryException):
                accessors[name]()
        else:
            _check_value(name, accessors[name](), float(expected))


# --------------------------------------------------------------------------
# Instrument builders
# --------------------------------------------------------------------------


def _vanilla(mkt: dict[str, Any], case: dict[str, Any]) -> QuantoVanillaOption:
    fx_rf, fx_vol, corr = _quanto_inputs(mkt, case)
    option = QuantoVanillaOption(
        PlainVanillaPayoff(OptionType(int(case["option_type"])), float(case["strike"])),
        EuropeanExercise(mkt["maturity"]),
    )
    option.set_pricing_engine(
        QuantoEngine(
            _process(mkt),
            fx_rf,
            fx_vol,
            corr,
            arguments=OptionArguments(),
            engine_factory=AnalyticEuropeanEngine,
        )
    )
    return option


def _forward(mkt: dict[str, Any], case: dict[str, Any]) -> QuantoForwardVanillaOption:
    fx_rf, fx_vol, corr = _quanto_inputs(mkt, case)
    option = QuantoForwardVanillaOption(
        float(case["moneyness"]),
        Date(int(case["reset_serial"])),
        # The probe deliberately carries a payoff strike that is NOT the
        # effective strike: the forward engine rebuilds it from moneyness.
        PlainVanillaPayoff(OptionType(int(case["option_type"])), 60.0),
        EuropeanExercise(mkt["maturity"]),
    )
    option.set_pricing_engine(
        QuantoEngine(
            _process(mkt),
            fx_rf,
            fx_vol,
            corr,
            arguments=ForwardOptionArguments(),
            engine_factory=lambda p: ForwardVanillaEngine(p, AnalyticEuropeanEngine),
        )
    )
    return option


def _barrier(mkt: dict[str, Any], case: dict[str, Any]) -> QuantoBarrierOption:
    fx_rf, fx_vol, corr = _quanto_inputs(mkt, case)
    option = QuantoBarrierOption(
        BarrierType(int(case["barrier_type"])),
        float(case["barrier"]),
        float(case["rebate"]),
        PlainVanillaPayoff(OptionType(int(case["option_type"])), float(case["strike"])),
        EuropeanExercise(mkt["maturity"]),
    )
    option.set_pricing_engine(
        QuantoEngine(
            _process(mkt),
            fx_rf,
            fx_vol,
            corr,
            arguments=BarrierOptionArguments(),
            engine_factory=AnalyticBarrierEngine,
        )
    )
    return option


# --------------------------------------------------------------------------
# The grids
# --------------------------------------------------------------------------


def test_quanto_vanilla_grid(cpp: dict[str, Any]) -> None:
    """72 cases: 2 payoff types x 2 strikes x foreign rate x fx vol x correlation."""
    mkt = _market(cpp)
    cases = cpp["vanilla"]
    assert len(cases) == 72
    for case in cases:
        _check_results(_vanilla(mkt, case), case)


def test_quanto_forward_vanilla_grid(cpp: dict[str, Any]) -> None:
    """60 cases: 2 payoff types x 3 moneyness x 2 reset dates x 5 markets."""
    mkt = _market(cpp)
    cases = cpp["forward"]
    assert len(cases) == 60
    for case in cases:
        _check_results(_forward(mkt, case), case)


def test_quanto_barrier_structural_grid(cpp: dict[str, Any]) -> None:
    """32 cases: all 4 barrier types x 2 payoff types x 2 levels x 2 rebates."""
    mkt = _market(cpp)
    cases = cpp["barrier_structural"]
    assert len(cases) == 32
    assert {c["barrier_type"] for c in cases} == {0, 1, 2, 3}
    for case in cases:
        _check_results(_barrier(mkt, case), case)


def test_quanto_barrier_market_grid(cpp: dict[str, Any]) -> None:
    """36 cases: 2 structures x foreign rate x fx vol x correlation."""
    mkt = _market(cpp)
    cases = cpp["barrier_market"]
    assert len(cases) == 36
    for case in cases:
        _check_results(_barrier(mkt, case), case)


# --------------------------------------------------------------------------
# Each quanto-only input must move an output — a dropped handle must fail
# --------------------------------------------------------------------------


def _vanilla_axis_cases(cpp: dict[str, Any], **fixed: float) -> list[dict[str, Any]]:
    return [
        c
        for c in cpp["vanilla"]
        if c["option_type"] == 1 and c["strike"] == 105.0 and all(c[k] == v for k, v in fixed.items())
    ]


def test_foreign_risk_free_curve_moves_the_price(cpp: dict[str, Any]) -> None:
    """Drop the foreign curve and this test fails: NPV must depend on it."""
    mkt = _market(cpp)
    cases = _vanilla_axis_cases(cpp, fx_vol=0.25, correlation=0.0)
    assert len({c["fx_rate"] for c in cases}) == 3
    npvs = [_vanilla(mkt, c).npv() for c in cases]
    assert len({round(v, 10) for v in npvs}) == 3
    for case, npv in zip(cases, npvs, strict=True):
        _check_value("npv", npv, float(case["npv"]))


def test_exchange_rate_volatility_moves_qlambda(cpp: dict[str, Any]) -> None:
    """At correlation 0 the FX vol drops out of the NPV but not out of qlambda.

    That asymmetry is the whole point of pinning greeks and not just prices:
    an engine that dropped the FX-vol handle would still match every NPV in
    the zero-correlation slice.
    """
    mkt = _market(cpp)
    cases = _vanilla_axis_cases(cpp, fx_rate=0.05, correlation=0.0)
    assert len({c["fx_vol"] for c in cases}) == 2
    npvs = [_vanilla(mkt, c).npv() for c in cases]
    qlambdas = [_vanilla(mkt, c).qlambda() for c in cases]
    assert len({round(v, 12) for v in npvs}) == 1
    assert len({round(v, 12) for v in qlambdas}) == 2
    for case, ql in zip(cases, qlambdas, strict=True):
        _check_value("qlambda", ql, float(case["qlambda"]))


def test_correlation_moves_the_price(cpp: dict[str, Any]) -> None:
    """Drop the correlation quote and this test fails."""
    mkt = _market(cpp)
    cases = _vanilla_axis_cases(cpp, fx_rate=0.05, fx_vol=0.25)
    assert {c["correlation"] for c in cases} == {-0.4, 0.0, 0.6}
    npvs = [_vanilla(mkt, c).npv() for c in cases]
    assert len({round(v, 10) for v in npvs}) == 3
    for case, npv in zip(cases, npvs, strict=True):
        _check_value("npv", npv, float(case["npv"]))


def test_correlation_quote_is_observed(cpp: dict[str, Any]) -> None:
    """Moving the correlation quote must reprice — the engine registers with it."""
    mkt = _market(cpp)
    base = next(
        c
        for c in cpp["vanilla"]
        if c["option_type"] == 1
        and c["strike"] == 105.0
        and c["fx_rate"] == 0.05
        and c["fx_vol"] == 0.25
        and c["correlation"] == 0.0
    )
    moved = next(
        c
        for c in cpp["vanilla"]
        if c["option_type"] == 1
        and c["strike"] == 105.0
        and c["fx_rate"] == 0.05
        and c["fx_vol"] == 0.25
        and c["correlation"] == 0.6
    )
    fx_rf, fx_vol, corr = _quanto_inputs(mkt, base)
    option = QuantoVanillaOption(
        PlainVanillaPayoff(OptionType.Call, 105.0), EuropeanExercise(mkt["maturity"])
    )
    option.set_pricing_engine(
        QuantoEngine(
            _process(mkt),
            fx_rf,
            fx_vol,
            corr,
            arguments=OptionArguments(),
            engine_factory=AnalyticEuropeanEngine,
        )
    )
    _check_value("npv", option.npv(), float(base["npv"]))
    corr.set_value(0.6)
    _check_value("npv", option.npv(), float(moved["npv"]))


# --------------------------------------------------------------------------
# ImpliedVolTermStructure — the piece ForwardVanillaEngine leans on
# --------------------------------------------------------------------------


def test_implied_vol_term_structure_delegates_and_shifts(cpp: dict[str, Any]) -> None:
    """Accessors delegate to the original surface; variance is forward variance.

    Over a flat surface the forward variance between ``s`` and ``s + t`` is
    ``sigma^2 * t`` regardless of ``s``, so the implied structure's variance
    at ``t`` must equal the original's variance at ``t`` — an identity the
    port either satisfies or fails outright.
    """
    mkt = _market(cpp)
    original = BlackConstantVol(
        reference_date=mkt["today"],
        calendar=NullCalendar(),
        day_counter=Actual360(),
        volatility=mkt["vol"],
    )
    reset = mkt["today"] + 90
    implied = ImpliedVolTermStructure(original, reset)

    assert implied.reference_date() == reset
    assert implied.day_counter().name() == original.day_counter().name()
    assert implied.max_date() == original.max_date()
    assert implied.min_strike() == original.min_strike()
    assert implied.max_strike() == original.max_strike()

    for t in (0.25, 0.5, 1.0):
        tight(
            implied.black_variance_at_time(t, 100.0, True),
            mkt["vol"] * mkt["vol"] * t,
            reason=f"flat forward variance at t={t}",
        )


# --------------------------------------------------------------------------
# Forward-specific optional arguments: moneyness and reset date
# --------------------------------------------------------------------------


def test_forward_moneyness_moves_the_price(cpp: dict[str, Any]) -> None:
    """Drop the moneyness and this test fails."""
    mkt = _market(cpp)
    cases = [
        c
        for c in cpp["forward"]
        if c["option_type"] == 1
        and c["reset_days"] == 90
        and c["fx_rate"] == 0.05
        and c["fx_vol"] == 0.20
        and c["correlation"] == 0.30
    ]
    assert {c["moneyness"] for c in cases} == {0.9, 1.0, 1.1}
    npvs = [_forward(mkt, c).npv() for c in cases]
    assert len({round(v, 10) for v in npvs}) == 3
    for case, npv in zip(cases, npvs, strict=True):
        _check_value("npv", npv, float(case["npv"]))


def test_forward_reset_date_moves_the_price(cpp: dict[str, Any]) -> None:
    """Drop the reset date and this test fails."""
    mkt = _market(cpp)
    cases = [
        c
        for c in cpp["forward"]
        if c["option_type"] == 1
        and c["moneyness"] == 1.0
        and c["fx_rate"] == 0.05
        and c["fx_vol"] == 0.20
        and c["correlation"] == 0.30
    ]
    assert {c["reset_days"] for c in cases} == {30, 90}
    npvs = [_forward(mkt, c).npv() for c in cases]
    assert len({round(v, 10) for v in npvs}) == 2
    for case, npv in zip(cases, npvs, strict=True):
        _check_value("npv", npv, float(case["npv"]))


# --------------------------------------------------------------------------
# Barrier-specific optional arguments: type, level, rebate
# --------------------------------------------------------------------------


def test_barrier_rebate_moves_the_price(cpp: dict[str, Any]) -> None:
    """Drop the rebate and this test fails on 12 of the 16 structures."""
    mkt = _market(cpp)
    by_structure: dict[tuple[int, float, int], dict[float, dict[str, Any]]] = {}
    for case in cpp["barrier_structural"]:
        key = (case["barrier_type"], case["barrier"], case["option_type"])
        by_structure.setdefault(key, {})[case["rebate"]] = case
    moved = 0
    for pair in by_structure.values():
        npv_no_rebate = _barrier(mkt, pair[0.0]).npv()
        npv_rebate = _barrier(mkt, pair[3.0]).npv()
        _check_value("npv", npv_no_rebate, float(pair[0.0]["npv"]))
        _check_value("npv", npv_rebate, float(pair[3.0]["npv"]))
        if abs(npv_no_rebate - npv_rebate) > 1.0e-12:
            moved += 1
    assert moved == 12


def test_barrier_level_moves_the_price(cpp: dict[str, Any]) -> None:
    """Drop the barrier level and this test fails."""
    mkt = _market(cpp)
    cases = [
        c
        for c in cpp["barrier_structural"]
        if c["barrier_type"] == BarrierType.DownOut and c["option_type"] == 1 and c["rebate"] == 0.0
    ]
    assert {c["barrier"] for c in cases} == {90.0, 100.0}
    npvs = [_barrier(mkt, c).npv() for c in cases]
    assert len({round(v, 10) for v in npvs}) == 2


def test_barrier_at_spot_is_not_triggered(cpp: dict[str, Any]) -> None:
    """Spot sitting exactly on the barrier prices; it does not knock out.

    C++ ``BarrierOption::engine::triggered`` uses strict ``<`` / ``>``
    (barrieroption.cpp:126-136). The 16 ``barrier == spot == 100`` rows in
    the structural grid all carry a price in the reference, so a port using
    ``<=`` / ``>=`` raises "barrier touched" here.
    """
    mkt = _market(cpp)
    cases = [c for c in cpp["barrier_structural"] if c["barrier"] == mkt["spot"]]
    assert len(cases) == 16
    for case in cases:
        _check_value("npv", _barrier(mkt, case).npv(), float(case["npv"]))


# --------------------------------------------------------------------------
# reset() / setupExpired() — two deliberately different sentinels
# --------------------------------------------------------------------------


def test_quanto_option_results_reset(cpp: dict[str, Any]) -> None:
    """``QuantoOptionResults.reset()`` nulls the quanto greeks (Null<Real>)."""
    ref = cpp["results_reset"]
    assert ref["qvega"] is None
    assert ref["qrho"] is None
    assert ref["qlambda"] is None

    results = QuantoOptionResults()
    results.value = 1.0
    results.delta = 2.0
    results.qvega = 3.0
    results.qrho = 4.0
    results.qlambda = 5.0
    results.reset()

    assert results.value is None
    assert results.delta is None
    assert results.qvega is None
    assert results.qrho is None
    assert results.qlambda is None


class _ExpiredVanilla(QuantoVanillaOption):
    def is_expired(self) -> bool:
        return True


class _ExpiredForward(QuantoForwardVanillaOption):
    def is_expired(self) -> bool:
        return True


class _ExpiredBarrier(QuantoBarrierOption):
    def is_expired(self) -> bool:
        return True


def test_expired_quanto_greeks_are_zero_not_null(cpp: dict[str, Any]) -> None:
    """After ``setup_expired`` the quanto greeks are 0.0 — NOT the reset() null.

    C++ ``Quanto*Option::setupExpired`` writes ``0.0``
    (quantovanillaoption.cpp:52-55, quantoforwardvanillaoption.cpp:52-55,
    quantobarrieroption.cpp:52-55), which is a different sentinel from the
    ``Null<Real>()`` that ``QuantoOptionResults::reset()`` writes. The probe
    pins both, and this test pins the pair against each other.

    Only NPV and the quanto greeks are asserted: the standard greeks go
    through ``OneAssetOption.setup_expired``, which in this port nulls them
    instead of zeroing them (see the report — pre-existing divergence from
    C++ ``OneAssetOption::setupExpired``).
    """
    mkt = _market(cpp)
    payoff = PlainVanillaPayoff(OptionType.Call, 105.0)
    exercise = EuropeanExercise(mkt["maturity"])
    base = cpp["barrier_market"][0]

    options: list[QuantoOption] = [
        _ExpiredVanilla(payoff, exercise),
        _ExpiredForward(1.0, mkt["today"] + 90, payoff, exercise),
        _ExpiredBarrier(BarrierType.DownOut, 90.0, 3.0, payoff, exercise),
    ]
    refs = [cpp["expired"]["vanilla"], cpp["expired"]["forward"], cpp["expired"]["barrier"]]
    engines: list[Callable[[], Any]] = [
        lambda: QuantoEngine(
            _process(mkt),
            *_quanto_inputs(mkt, base),
            arguments=OptionArguments(),
            engine_factory=AnalyticEuropeanEngine,
        ),
        lambda: QuantoEngine(
            _process(mkt),
            *_quanto_inputs(mkt, base),
            arguments=ForwardOptionArguments(),
            engine_factory=lambda p: ForwardVanillaEngine(p, AnalyticEuropeanEngine),
        ),
        lambda: QuantoEngine(
            _process(mkt),
            *_quanto_inputs(mkt, base),
            arguments=BarrierOptionArguments(),
            engine_factory=AnalyticBarrierEngine,
        ),
    ]

    for option, ref, make_engine in zip(options, refs, engines, strict=True):
        option.set_pricing_engine(make_engine())
        assert ref["npv"] == 0.0
        assert option.npv() == float(ref["npv"])
        for name in QUANTO_GREEKS:
            assert ref[name] == 0.0
            assert getattr(option, name)() == float(ref[name])


# --------------------------------------------------------------------------
# QL_REQUIRE branches
# --------------------------------------------------------------------------


def test_quanto_engine_rejects_null_underlying(cpp: dict[str, Any]) -> None:
    assert cpp["raises"]["quanto_engine_null_underlying"]["raises"] is True
    mkt = _market(cpp)
    dead = dict(mkt)
    dead["spot"] = 0.0
    base = cpp["vanilla"][0]
    fx_rf, fx_vol, corr = _quanto_inputs(mkt, base)
    option = QuantoVanillaOption(
        PlainVanillaPayoff(OptionType.Call, 105.0), EuropeanExercise(mkt["maturity"])
    )
    option.set_pricing_engine(
        QuantoEngine(
            _process(dead),
            fx_rf,
            fx_vol,
            corr,
            arguments=OptionArguments(),
            engine_factory=AnalyticEuropeanEngine,
        )
    )
    with pytest.raises(LibraryException, match="negative or null underlying"):
        option.npv()


@pytest.mark.parametrize(
    ("key", "moneyness", "reset_offset", "message"),
    [
        ("forward_zero_moneyness", 0.0, 90, "negative or zero moneyness"),
        ("forward_reset_after_maturity", 1.0, 180, "reset date later or equal to maturity"),
        ("forward_reset_in_the_past", 1.0, -1, "reset date in the past"),
    ],
)
def test_forward_argument_validation(
    cpp: dict[str, Any], key: str, moneyness: float, reset_offset: int, message: str
) -> None:
    assert cpp["raises"][key]["raises"] is True
    mkt = _market(cpp)
    base = cpp["vanilla"][0]
    fx_rf, fx_vol, corr = _quanto_inputs(mkt, base)
    option = QuantoForwardVanillaOption(
        moneyness,
        mkt["today"] + reset_offset,
        PlainVanillaPayoff(OptionType.Call, 105.0),
        EuropeanExercise(mkt["maturity"]),
    )
    option.set_pricing_engine(
        QuantoEngine(
            _process(mkt),
            fx_rf,
            fx_vol,
            corr,
            arguments=ForwardOptionArguments(),
            engine_factory=lambda p: ForwardVanillaEngine(p, AnalyticEuropeanEngine),
        )
    )
    with pytest.raises(LibraryException, match=message):
        option.npv()


def test_barrier_touched(cpp: dict[str, Any]) -> None:
    assert cpp["raises"]["barrier_touched"]["raises"] is True
    mkt = _market(cpp)
    base = cpp["barrier_market"][0]
    option = _barrier(mkt, {**base, "barrier_type": int(BarrierType.DownOut), "barrier": 110.0})
    with pytest.raises(LibraryException, match="barrier touched"):
        option.npv()


@pytest.mark.parametrize(
    ("key", "accessor"),
    [
        ("barrier_delta_not_provided", "delta"),
        ("barrier_qvega_not_provided", "qvega"),
        ("barrier_qrho_not_provided", "qrho"),
        ("barrier_qlambda_not_provided", "qlambda"),
    ],
)
def test_barrier_greeks_are_not_provided(cpp: dict[str, Any], key: str, accessor: str) -> None:
    """AnalyticBarrierEngine fills only the NPV, so every greek must refuse."""
    assert cpp["raises"][key]["raises"] is True
    mkt = _market(cpp)
    base = cpp["barrier_market"][0]
    option = _barrier(mkt, {**base, "barrier_type": int(BarrierType.DownOut), "barrier": 90.0})
    assert option.npv() > 0.0
    with pytest.raises(LibraryException):
        getattr(option, accessor)()
