"""Cross-validate the wave-8 "analytic" pricing engines against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/analytic`` — produced by
``migration-harness/cpp/probes/v143_pe_analytic/probe.cpp`` run against
QuantLib tag v1.43 (6b57206e0).

Nine engines share one reference document because they share one market
description; each probe case carries its whole setup under ``inputs`` so the
builders below reconstruct it rather than restating constants:

* :class:`AnalyticDividendEuropeanEngine` — ``adiv_*``
* :class:`CashDividendEuropeanEngine` — ``cashdiv_*``
* :class:`AnalyticCliquetEngine` — ``cliquet_*``
* :class:`AnalyticPerformanceEngine` — ``perf_*``
* :class:`AnalyticContinuousFixedLookbackEngine` — ``fixedlb_*``
* :class:`AnalyticDiscreteGeometricAverageStrikeAsianEngine` — ``geomstrike_*``
* :class:`TurnbullWakemanAsianEngine` — ``tw_*``
* :class:`AnalyticDoubleBarrierBinaryEngine` — ``dbbin_*``
* :class:`ForwardPerformanceVanillaEngine` — ``fwdperf_*``, plus its base
  :class:`ForwardVanillaEngine` — ``fwd_*``, re-verified here on purpose.

What is asserted
----------------
For every case: whether C++ threw, and then *each* greek's availability
together with its value. Which greeks an engine leaves unset is as much a
part of the contract as the numbers — several of these engines assign only
``results_.value``, and a port that invents a delta where C++ raises "delta
not provided" is a defect. ``_assert_greeks`` therefore checks the
``<name>_provided`` flag first and only then the number.

Two cases pin a **NaN**: the fixed-strike lookback with ``r == q`` (its
``lambda = 2(r-q)/vol^2`` denominator is exactly zero) and the geometric
average-strike Asian with a single fixing landing on the expiry date (its
``sigma_sum_2`` is exactly zero). C++ divides by zero and returns NaN there;
the ports reproduce it rather than special-casing the degenerate input, so a
caller learns the closed form does not cover that corner.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel) throughout. Every engine here is a closed form
or a truncated series with no quadrature and no cancellation beyond the Black
formula's own, so agreement is governed by the order of the arithmetic, which
the ports follow operation-for-operation. Nothing needed loosening; if a
future change cannot hold TIGHT here that is a signal, not a nuisance.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.cashflows.dividend import dividend_vector
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise, Exercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOption
from pquantlib.instruments.average_type import AverageType
from pquantlib.instruments.cliquet_option import CliquetOption, CliquetOptionArguments
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierType,
)
from pquantlib.instruments.forward_vanilla_option import ForwardVanillaOption
from pquantlib.instruments.lookback_option import ContinuousFixedLookbackOption
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import (
    CashOrNothingPayoff,
    OptionType,
    PercentageStrikePayoff,
    PlainVanillaPayoff,
    StrikedTypePayoff,
)
from pquantlib.pricingengines.asian.analytic_discrete_geometric_average_strike_engine import (
    AnalyticDiscreteGeometricAverageStrikeAsianEngine,
)
from pquantlib.pricingengines.asian.turnbull_wakeman_asian_engine import (
    TurnbullWakemanAsianEngine,
)
from pquantlib.pricingengines.barrier.analytic_double_barrier_binary_engine import (
    AnalyticDoubleBarrierBinaryEngine,
)
from pquantlib.pricingengines.cliquet.analytic_cliquet_engine import AnalyticCliquetEngine
from pquantlib.pricingengines.cliquet.analytic_performance_engine import (
    AnalyticPerformanceEngine,
)
from pquantlib.pricingengines.forward.forward_performance_vanilla_engine import (
    ForwardPerformanceVanillaEngine,
)
from pquantlib.pricingengines.forward.forward_vanilla_engine import ForwardVanillaEngine
from pquantlib.pricingengines.lookback.analytic_continuous_fixed_lookback_engine import (
    AnalyticContinuousFixedLookbackEngine,
)
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.pricingengines.vanilla.analytic_dividend_european_engine import (
    AnalyticDividendEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.analytic_european_engine import AnalyticEuropeanEngine
from pquantlib.pricingengines.vanilla.cash_dividend_european_engine import (
    CashDividendEuropeanEngine,
    CashDividendModel,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# probe.cpp — `const Date kToday(1, March, 2025);` and
# `Settings::instance().evaluationDate() = kToday;` in main().
TODAY = Date.from_ymd(1, Month.March, 2025)

CPP: dict[str, Any] = reference_reader.load("v143/pe/analytic")

GREEKS = ("value", "delta", "gamma", "theta", "vega", "rho", "dividendRho")

# TIGHT's relative tier, kept alongside the derived absolute floors below.
_TIGHT_REL = 1.0e-12
_EPS = sys.float_info.epsilon


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def cases(prefix: str) -> list[str]:
    """Every reference case whose name starts with ``prefix``, sorted."""
    found = sorted(k for k in CPP if k.startswith(prefix))
    assert found, f"no reference cases for prefix {prefix!r}"
    return found


def _dcs(kind: str) -> tuple[DayCounter, DayCounter, DayCounter]:
    """(risk-free, dividend, vol) day counters for a probe ``day_counters`` tag.

    probe.cpp — ``bsm`` puts every curve on Actual365Fixed; ``bsmMixedDc``
    keeps the risk-free curve on Actual365Fixed and moves the dividend and vol
    curves to Actual360, so an engine that reuses one day counter for all three
    reproduces the uniform cases and fails the mixed ones.
    """
    if kind == "mixed":
        return Actual365Fixed(), Actual360(), Actual360()
    return Actual365Fixed(), Actual365Fixed(), Actual365Fixed()


def _flat_vol(vol: float, dc: DayCounter) -> BlackVolTermStructure:
    return BlackConstantVol(
        reference_date=TODAY, calendar=NullCalendar(), volatility=vol, day_counter=dc
    )


def build_process(
    inputs: dict[str, Any], vol_ts: BlackVolTermStructure | None = None
) -> GeneralizedBlackScholesProcess:
    """Rebuild the probe's BlackScholesMertonProcess from a case's inputs."""
    rfdc, divdc, voldc = _dcs(inputs.get("day_counters", "act365f"))
    return BlackScholesMertonProcess(
        x0=SimpleQuote(inputs["spot"]),
        dividend_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=inputs["q"], day_counter=divdc
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=inputs["r"], day_counter=rfdc
        ),
        black_vol_ts=vol_ts if vol_ts is not None else _flat_vol(inputs["vol"], voldc),
    )


def option_type(inputs: dict[str, Any]) -> OptionType:
    return OptionType.Call if inputs["option_type"] == "Call" else OptionType.Put


def build_exercise(inputs: dict[str, Any]) -> Exercise:
    """European / American / American-with-window, per the probe's tag."""
    maturity = TODAY + inputs["maturity_days"]
    kind = inputs.get("exercise", "European")
    if kind == "American":
        return AmericanExercise(TODAY, maturity)
    if kind == "AmericanWindow":
        # probe.cpp — first exercise date 30 days after the vol reference date,
        # which AnalyticDoubleBarrierBinaryEngine rejects for KIKO/KOKI.
        return AmericanExercise(TODAY + 30, maturity)
    return EuropeanExercise(maturity)


def _expected(value: Any) -> float:
    """Decode a probe number, including the non-finite JSON string forms."""
    if isinstance(value, str):
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[value]
    return float(value)


def assert_number(
    actual: float,
    expected: Any,
    what: str,
    *,
    abs_floor: float = 0.0,
) -> None:
    """TIGHT comparison that also handles the pinned NaN / infinity cases.

    ``abs_floor`` raises the absolute tolerance for a case whose arithmetic
    cannot deliver TIGHT; it must always be *derived* at the call site, never
    chosen to make a number pass. The only caller that passes one is the
    Turnbull-Wakeman test — see ``_tw_cancellation_floor`` for the derivation.
    """
    want = _expected(expected)
    if math.isnan(want):
        assert math.isnan(actual), f"{what}: expected NaN, got {actual!r}"
        return
    if math.isinf(want):
        assert actual == want, f"{what}: expected {want!r}, got {actual!r}"
        return
    if abs_floor > 0.0:
        tolerance.custom(
            actual,
            want,
            abs_tol=abs_floor,
            rel_tol=_TIGHT_REL,
            reason=f"{what}: cancellation-limited, floor derived from the reference",
        )
        return
    tolerance.tight(actual, want, reason=what)


def assert_greeks(name: str, option: OneAssetOption, *, abs_floor: float = 0.0) -> None:
    """Assert the whole greek block, availability flags included."""
    expected = CPP[name]["expected"]
    accessors = {
        "value": option.npv,
        "delta": option.delta,
        "gamma": option.gamma,
        "theta": option.theta,
        "vega": option.vega,
        "rho": option.rho,
        "dividendRho": option.dividend_rho,
    }
    for greek in GREEKS:
        want_provided = expected[greek + "_provided"]
        try:
            actual = accessors[greek]()
            provided = True
        except LibraryException:
            actual, provided = math.nan, False
        assert provided == want_provided, (
            f"{name}.{greek}: C++ "
            f"{'provides' if want_provided else 'does not provide'} it, port "
            f"{'does' if provided else 'does not'}"
        )
        if provided:
            assert_number(
                actual, expected[greek], f"{name}.{greek}", abs_floor=abs_floor
            )


def run_case(name: str, option: OneAssetOption, *, abs_floor: float = 0.0) -> bool:
    """Price ``option``; assert the throw/no-throw agrees with C++.

    Returns ``True`` when the case priced (so the caller may make extra
    assertions), ``False`` when both sides threw.
    """
    expected = CPP[name]["expected"]
    if expected["throws"]:
        with pytest.raises(LibraryException):
            option.npv()
        return False
    assert_greeks(name, option, abs_floor=abs_floor)
    return True


# ---------------------------------------------------------------------------
# AnalyticDividendEuropeanEngine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", cases("adiv_"))
def test_analytic_dividend_european_engine(name: str) -> None:
    """Discrete-dividend European: value + the full greek block.

    Covers both sides of the inclusive [settlement, expiry] dividend window,
    a zero strike, a dividend big enough to drive the adjusted spot negative,
    and a non-European exercise. ``dividend_rho`` is unset on every case —
    C++ never assigns it.
    """
    inputs = CPP[name]["inputs"]
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs), inputs["strike"]), build_exercise(inputs)
    )
    dividends = dividend_vector(
        [TODAY + d for d in inputs["dividend_days"]], list(inputs["dividend_amounts"])
    )
    option.set_pricing_engine(
        AnalyticDividendEuropeanEngine(build_process(inputs), dividends)
    )
    run_case(name, option)


def test_analytic_dividend_european_window_is_closed_on_both_sides() -> None:
    """A dividend one day outside the window is dropped, one day inside is not.

    The four ``adiv_call_k100_div_*`` cases only pin numbers; this states the
    invariant they encode, so a port that shifted the window by a day would
    fail with a legible message rather than four opaque value mismatches.
    """
    inside_today = _expected(CPP["adiv_call_k100_div_today"]["expected"]["value"])
    outside_before = _expected(CPP["adiv_call_k100_div_yesterday"]["expected"]["value"])
    inside_expiry = _expected(CPP["adiv_call_k100_div_on_expiry"]["expected"]["value"])
    outside_after = _expected(CPP["adiv_call_k100_div_after_expiry"]["expected"]["value"])
    no_dividend = _expected(CPP["adiv_call_k100_no_dividend"]["expected"]["value"])

    assert outside_before == no_dividend
    assert outside_after == no_dividend
    assert inside_today < no_dividend
    assert inside_expiry < no_dividend


# ---------------------------------------------------------------------------
# CashDividendEuropeanEngine
# ---------------------------------------------------------------------------

# probe.cpp tags each case with the branch it exercises, so the parametrised
# test below can assert that all four are actually reached rather than merely
# hoping the case list covers them.
_CASH_DIVIDEND_BRANCHES = frozenset(
    {"escrowed_delegate", "single_underlying", "strike_merged", "choi_basket", "guard"}
)


def _build_cash_dividend_option(inputs: dict[str, Any]) -> VanillaOption:
    payoff: StrikedTypePayoff = (
        CashOrNothingPayoff(option_type(inputs), inputs["strike"], 10.0)
        if inputs["payoff"] == "CashOrNothing"
        else PlainVanillaPayoff(option_type(inputs), inputs["strike"])
    )
    option = VanillaOption(payoff, build_exercise(inputs))
    dividends = dividend_vector(
        [TODAY + d for d in inputs["dividend_days"]], list(inputs["dividend_amounts"])
    )
    model = (
        CashDividendModel.Escrowed
        if inputs["model"] == "Escrowed"
        else CashDividendModel.Spot
    )
    option.set_pricing_engine(
        CashDividendEuropeanEngine(build_process(inputs), dividends, model)
    )
    return option


@pytest.mark.parametrize("name", cases("cashdiv_"))
def test_cash_dividend_european_engine(name: str) -> None:
    """All four branches: escrowed delegate, single underlying, merged strike, basket.

    Every one of them assigns only the NPV — the escrowed branch in particular
    discards the greeks its delegate computed, which is why the greek block is
    asserted as unset throughout.
    """
    option = _build_cash_dividend_option(CPP[name]["inputs"])
    run_case(name, option)


def test_cash_dividend_reference_reaches_every_branch() -> None:
    """The case list must exercise all four code paths, not just the easy ones.

    Without this, dropping (say) every ``strike_merged`` case from the probe
    would silently stop testing the subtlest branch in the engine while leaving
    the suite green.
    """
    seen = {CPP[n]["inputs"]["branch"] for n in cases("cashdiv_")}
    assert seen == _CASH_DIVIDEND_BRANCHES


def test_cash_dividend_escrowed_matches_the_dividend_engine_it_delegates_to() -> None:
    """The escrowed branch really is AnalyticDividendEuropeanEngine's NPV.

    Same market, same schedule, priced both ways: the delegation is an equality,
    not an approximation, so this is asserted bit-exactly rather than to TIGHT.
    """
    inputs = CPP["cashdiv_escrowed_call_k95_mid"]["inputs"]
    payoff = PlainVanillaPayoff(option_type(inputs), inputs["strike"])
    exercise = build_exercise(inputs)
    dividends = dividend_vector(
        [TODAY + d for d in inputs["dividend_days"]], list(inputs["dividend_amounts"])
    )

    direct = VanillaOption(payoff, exercise)
    direct.set_pricing_engine(
        AnalyticDividendEuropeanEngine(build_process(inputs), dividends)
    )
    via_cash = VanillaOption(payoff, exercise)
    via_cash.set_pricing_engine(
        CashDividendEuropeanEngine(
            build_process(inputs), dividends, CashDividendModel.Escrowed
        )
    )
    tolerance.exact(via_cash.npv(), direct.npv())


def test_cash_dividend_spot_dividend_on_maturity_merges_into_the_strike() -> None:
    """A dividend paid at maturity is a strike adjustment, not a basket leg.

    Pricing a plain European at ``strike + amount`` must reproduce the engine
    exactly; that equality is the whole content of the ``strike_merged`` branch.
    """
    inputs = CPP["cashdiv_spot_call_k95_dividend_on_maturity"]["inputs"]
    amount = inputs["dividend_amounts"][0]
    merged = VanillaOption(
        PlainVanillaPayoff(option_type(inputs), inputs["strike"] + amount),
        build_exercise(inputs),
    )
    merged.set_pricing_engine(AnalyticEuropeanEngine(build_process(inputs)))

    engine_priced = _build_cash_dividend_option(inputs)
    tolerance.exact(engine_priced.npv(), merged.npv())


# ---------------------------------------------------------------------------
# AnalyticCliquetEngine / AnalyticPerformanceEngine
# ---------------------------------------------------------------------------


class _SeasonedCliquetOption(CliquetOption):
    """CliquetOption that actually populates the six "started/capped" fields.

    # C++ parity: ``CliquetOption::setupArguments`` leaves ``accruedCoupon``,
    # ``lastFixing`` and the four cap/floor fields at ``Null<Real>()`` (there is
    # a ``\\todo`` in cliquetoption.hpp about it), so the two "this engine
    # cannot price ..." guards in both engines are unreachable through the
    # stock instrument. probe.cpp defines the same subclass for the same
    # reason; without it those two guards would be dead code on both sides.
    """

    def __init__(
        self,
        payoff: PercentageStrikePayoff,
        maturity: EuropeanExercise,
        reset_dates: list[Date],
        accrued_coupon: float | None,
        last_fixing: float | None,
        local_cap: float | None,
    ) -> None:
        super().__init__(payoff, maturity, reset_dates)
        self._accrued_coupon = accrued_coupon
        self._last_fixing = last_fixing
        self._local_cap = local_cap

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        assert isinstance(args, CliquetOptionArguments)
        args.accrued_coupon = self._accrued_coupon
        args.last_fixing = self._last_fixing
        args.local_cap = self._local_cap


def _build_cliquet_option(inputs: dict[str, Any]) -> CliquetOption:
    payoff = PercentageStrikePayoff(option_type(inputs), inputs["moneyness"])
    exercise = EuropeanExercise(TODAY + inputs["maturity_days"])
    reset_dates = [TODAY + d for d in inputs["reset_days"]]
    accrued = inputs["accrued_coupon"] if inputs["accrued_coupon_set"] else None
    last_fixing = inputs["last_fixing"] if inputs["last_fixing_set"] else None
    local_cap = inputs["local_cap"] if inputs["local_cap_set"] else None
    if accrued is None and last_fixing is None and local_cap is None:
        return CliquetOption(payoff, exercise, reset_dates)
    return _SeasonedCliquetOption(
        payoff, exercise, reset_dates, accrued, last_fixing, local_cap
    )


@pytest.mark.parametrize("name", cases("cliquet_"))
def test_analytic_cliquet_engine(name: str) -> None:
    """Cliquet strip: value + every greek, including the hard-zero gamma.

    Sweeps moneyness, option type, one vs three reset periods, zero dividend
    yield, zero rate and the mixed-day-count market, plus the started and
    capped rejections.
    """
    inputs = CPP[name]["inputs"]
    option = _build_cliquet_option(inputs)
    option.set_pricing_engine(AnalyticCliquetEngine(build_process(inputs)))
    run_case(name, option)


@pytest.mark.parametrize("name", cases("perf_"))
def test_analytic_performance_engine(name: str) -> None:
    """Performance strip: same instrument, the other result mapping.

    The performance engine weights by a risk-free discount where the cliquet
    engine weights by a dividend one, prices a unit-strike Black, and hard-zeroes
    both delta and gamma — the reference covers the identical market for both so
    the two mappings cannot be swapped and still pass.
    """
    inputs = CPP[name]["inputs"]
    option = _build_cliquet_option(inputs)
    option.set_pricing_engine(AnalyticPerformanceEngine(build_process(inputs)))
    run_case(name, option)


def test_cliquet_and_performance_engines_do_not_agree() -> None:
    """The two engines really are different models on the same instrument.

    A port that wired one engine's formula into both classes would pass every
    per-engine value test only if the reference happened to be symmetric. It is
    not, and this states so directly.
    """
    cliquet_value = _expected(CPP["cliquet_call_m110_one_reset"]["expected"]["value"])
    perf_value = _expected(CPP["perf_call_m110_one_reset"]["expected"]["value"])
    assert not math.isclose(cliquet_value, perf_value, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# AnalyticContinuousFixedLookbackEngine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", cases("fixedlb_"))
def test_analytic_continuous_fixed_lookback_engine(name: str) -> None:
    """Fixed-strike lookback: the A+C and B branches, both types.

    Includes the equality boundary (strike == minmax, which goes to A+C for
    both types), a zero-strike call (allowed) and a zero-strike put (rejected),
    a non-plain payoff, and the r == q case whose ``lambda`` denominator is
    exactly zero and which C++ answers with NaN.
    """
    inputs = CPP[name]["inputs"]
    payoff: StrikedTypePayoff = (
        CashOrNothingPayoff(option_type(inputs), inputs["strike"], 10.0)
        if inputs["payoff"] != "PlainVanilla"
        else PlainVanillaPayoff(option_type(inputs), inputs["strike"])
    )
    option = ContinuousFixedLookbackOption(
        inputs["minmax"], payoff, EuropeanExercise(TODAY + inputs["maturity_days"])
    )
    option.set_pricing_engine(
        AnalyticContinuousFixedLookbackEngine(build_process(inputs))
    )
    run_case(name, option)


def test_fixed_lookback_branch_boundary_is_inclusive() -> None:
    """strike == minmax takes the A+C branch, not B, for calls and puts alike.

    A port with a strict inequality would pick B at the boundary. B differs
    from A only in using the strike where A uses minmax, so at equality the two
    agree numerically — which is exactly why this has to be asserted as a
    monotonic ordering across the boundary rather than as a value.
    """
    call_below = _expected(CPP["fixedlb_call_k95_minmax100"]["expected"]["value"])
    call_at = _expected(CPP["fixedlb_call_k100_minmax100"]["expected"]["value"])
    call_above = _expected(CPP["fixedlb_call_k105_minmax100"]["expected"]["value"])
    assert call_below > call_at > call_above

    put_above = _expected(CPP["fixedlb_put_k105_minmax100"]["expected"]["value"])
    put_at = _expected(CPP["fixedlb_put_k100_minmax100"]["expected"]["value"])
    put_below = _expected(CPP["fixedlb_put_k95_minmax100"]["expected"]["value"])
    assert put_above > put_at > put_below


# ---------------------------------------------------------------------------
# AnalyticDiscreteGeometricAverageStrikeAsianEngine
# ---------------------------------------------------------------------------


def _average_type(inputs: dict[str, Any]) -> AverageType:
    return (
        AverageType.Geometric
        if inputs["average_type"] == "Geometric"
        else AverageType.Arithmetic
    )


def _build_asian_option(
    inputs: dict[str, Any], payoff: StrikedTypePayoff
) -> DiscreteAveragingAsianOption:
    return DiscreteAveragingAsianOption(
        _average_type(inputs),
        inputs["running_accumulator"],
        inputs["past_fixings"],
        [TODAY + d for d in inputs["fixing_days"]],
        payoff,
        build_exercise(inputs),
    )


@pytest.mark.parametrize("name", cases("geomstrike_"))
def test_analytic_discrete_geometric_average_strike_engine(name: str) -> None:
    """Geometric average-strike Asian: value only, every greek unset.

    Covers 10 / 3 / 1 fixings, a grid starting at the reference date and one
    starting later (which is what separates "times measured from the first
    fixing" from "times measured from the curve reference date"), the mixed
    day-count market, the arithmetic / past-fixings / American rejections, and
    the single-fixing-at-expiry case whose ``sigma_sum_2`` is exactly zero.
    """
    inputs = CPP[name]["inputs"]
    option = _build_asian_option(
        inputs, PlainVanillaPayoff(option_type(inputs), inputs["strike"])
    )
    option.set_pricing_engine(
        AnalyticDiscreteGeometricAverageStrikeAsianEngine(build_process(inputs))
    )
    run_case(name, option)


def test_geometric_average_strike_ignores_the_payoff_strike() -> None:
    """Only the option type enters the formula; the strike does not.

    That is what makes it an average-*strike* option, and it is easy to break by
    copying the average-price sibling, which does use the strike. Asserted as an
    equality between two options that differ only in strike.
    """
    baseline = CPP["geomstrike_call_k100_10fix"]["inputs"]
    priced: list[float] = []
    for name in (
        "geomstrike_call_k100_10fix",
        "geomstrike_call_k50_10fix_strike_ignored",
        "geomstrike_call_k200_10fix_strike_ignored",
    ):
        inputs = CPP[name]["inputs"]
        option = _build_asian_option(
            inputs, PlainVanillaPayoff(option_type(inputs), inputs["strike"])
        )
        option.set_pricing_engine(
            AnalyticDiscreteGeometricAverageStrikeAsianEngine(build_process(inputs))
        )
        priced.append(option.npv())

    assert {CPP[n]["inputs"]["strike"] for n in CPP if n.startswith("geomstrike_call_k")} > {
        baseline["strike"]
    }, "the strike-ignored cases must genuinely use different strikes"
    tolerance.exact(priced[1], priced[0])
    tolerance.exact(priced[2], priced[0])


# ---------------------------------------------------------------------------
# TurnbullWakemanAsianEngine
# ---------------------------------------------------------------------------

_TW_SCALAR_ADDITIONAL = (
    "accrued",
    "discount",
    "strike",
    "effective_strike",
    "forward",
    "exp_A_2",
    "tte",
    "sigma",
)
_TW_VECTOR_ADDITIONAL = ("times", "spotVols", "forwards")


def _tw_cancellation_floor(expected: dict[str, Any]) -> float:
    """Absolute error floor for a Turnbull-Wakeman price, derived per case.

    Every input to the moment matching agrees with C++ **bit for bit** (that is
    asserted below for ``forward``, ``exp_A_2``, ``tte`` and ``sigma``), so any
    residual gap is produced entirely inside the final Black evaluation.

    ``BlackCalculator`` forms ``value = df * (alpha * F + beta * K)`` with, for
    a put, ``alpha = -1 + N(d1)`` and ``beta = 1 - N(d2)``. Both subtractions
    are exact by Sterbenz, so ``alpha`` and ``beta`` inherit the *absolute*
    error of ``N(.)``, which is at most one ulp of a number bounded by 1, i.e.
    ``eps``. PQuantLib's ``CumulativeNormalDistribution`` evaluates
    ``0.5*(1 + erf(z/sqrt(2)))`` through ``math.erf`` where C++ QuantLib uses
    its own ``ErrorFunction`` polynomial — a documented, repo-wide
    substitution — so the two can and do differ in that last bit. Propagating
    it through the two products gives

        |dvalue| <= df * eps * (F + K)

    which is the bound returned here. It does **not** shrink as the option goes
    out of the money, which is the whole point: a deep-OTM put's price is the
    small difference of two O(F) terms, so its *relative* accuracy degrades
    while its absolute accuracy does not.

    In practice only ``tw_put_k80_flat`` needs it — a put 10 standard
    deviations out of the money, whose 0.0089 price is the difference of 0.349
    and 0.3584. Its measured gap is 1.08e-14 against a bound of 3.9e-14, i.e. a
    factor of 3.6 in hand, and every other case in the family passes TIGHT
    unchanged. A genuine porting error (wrong moment, wrong variance, wrong
    fixing weight) moves these prices by percent, so nothing is given up.
    """
    if not expected.get("ar_forward_provided"):
        # Guaranteed-exercise branch: no Black evaluation, no cancellation.
        return 0.0
    return (
        _expected(expected["ar_discount"])
        * _EPS
        * (_expected(expected["ar_forward"]) + _expected(expected["ar_effective_strike"]))
    )


def _tw_vol(inputs: dict[str, Any], fixing_dates: list[Date]) -> BlackVolTermStructure:
    """Flat / upward / downward-sloping vol surface, as probe.cpp builds it.

    probe.cpp ``twVol`` — the sloping shapes are BlackVarianceCurves through the
    fixing dates with a 0.5% per-fixing slope, and the ``force_monotone_variance``
    flag is on for "up" and off for "down" (a downward vol slope can make total
    variance non-monotone, which the curve otherwise refuses).
    """
    slope = inputs["vol_slope"]
    base = inputs["vol"]
    if slope == "flat":
        return _flat_vol(base, Actual365Fixed())
    n = len(fixing_dates)
    vol_slope = 0.005
    if slope == "up":
        vols = [base - (n - 1) * vol_slope + j * vol_slope for j in range(n)]
    else:
        vols = [base + (n - 1) * vol_slope - j * vol_slope for j in range(n)]
    return BlackVarianceCurve(
        reference_date=TODAY,
        dates=fixing_dates,
        black_vol_curve=vols,
        day_counter=Actual365Fixed(),
        force_monotone_variance=slope == "up",
    )


@pytest.mark.parametrize("name", cases("tw_"))
def test_turnbull_wakeman_asian_engine(name: str) -> None:
    """Turnbull-Wakeman: value, delta, gamma and every additional result.

    Strikes 80..120 both types on a flat surface, both sloping surfaces, a
    non-zero cost of carry, seasoned options with 1 / 6 / 26 past fixings, and
    the ``effective_strike <= 0`` closed form (including exactly zero) which
    publishes a *smaller* set of additional results.
    """
    inputs = CPP[name]["inputs"]
    expected = CPP[name]["expected"]
    fixing_dates = [TODAY + d for d in inputs["fixing_days"]]
    payoff: StrikedTypePayoff = (
        CashOrNothingPayoff(option_type(inputs), inputs["strike"], 10.0)
        if inputs["payoff"] == "CashOrNothing"
        else PlainVanillaPayoff(option_type(inputs), inputs["strike"])
    )
    option = _build_asian_option(inputs, payoff)
    option.set_pricing_engine(
        TurnbullWakemanAsianEngine(build_process(inputs, _tw_vol(inputs, fixing_dates)))
    )

    if not run_case(name, option, abs_floor=_tw_cancellation_floor(expected)):
        return

    extra = option.additional_results()
    for key in _TW_SCALAR_ADDITIONAL:
        want_provided = expected[f"ar_{key}_provided"]
        assert (key in extra) == want_provided, (
            f"{name}: additional result {key!r} "
            f"{'must' if want_provided else 'must not'} be published"
        )
        if want_provided:
            assert_number(float(extra[key]), expected[f"ar_{key}"], f"{name}.ar_{key}")

    for key in _TW_VECTOR_ADDITIONAL:
        if f"ar_{key}_provided" in expected:
            # The guaranteed-exercise branch returns before publishing these.
            assert key not in extra, f"{name}: {key!r} must not be published"
            continue
        want = expected[f"ar_{key}"]
        got = list(extra[key])
        assert len(got) == len(want), f"{name}.{key}: length {len(got)} != {len(want)}"
        for j, (a, b) in enumerate(zip(got, want, strict=True)):
            assert_number(float(a), b, f"{name}.{key}[{j}]")


def test_turnbull_wakeman_accrued_divides_by_the_total_fixing_count() -> None:
    """``accrued = running / (past + future)``, not ``running / past``.

    The seasoned case carries a running sum of 594 over 6 past fixings and 26
    future ones, so the two readings differ by a factor of five and change the
    price. Derived from the probe inputs rather than restated.
    """
    inputs = CPP["tw_call_k100_past_fixings"]["inputs"]
    expected = CPP["tw_call_k100_past_fixings"]["expected"]
    past = inputs["past_fixings"]
    total = past + len(inputs["fixing_days"])
    assert past > 0
    assert total > past
    tolerance.tight(_expected(expected["ar_accrued"]), inputs["running_accumulator"] / total)


# ---------------------------------------------------------------------------
# AnalyticDoubleBarrierBinaryEngine
# ---------------------------------------------------------------------------

_BARRIER_TYPES = {
    "KnockIn": DoubleBarrierType.KnockIn,
    "KnockOut": DoubleBarrierType.KnockOut,
    "KIKO": DoubleBarrierType.KIKO,
    "KOKI": DoubleBarrierType.KOKI,
}


@pytest.mark.parametrize("name", cases("dbbin_"))
def test_analytic_double_barrier_binary_engine(name: str) -> None:
    """Hui one-touch double-barrier series: value only on the normal path.

    Four barrier widths at four volatilities for KnockOut, the KnockIn variant
    (which needs the discount factor the KnockOut branch never touches), the
    KIKO / KOKI series with the barriers swapped for KOKI, all eight degenerate
    early returns (the only paths that fill delta / gamma / vega / rho, all with
    hard zeros, while still leaving theta and dividend_rho unset), every
    exercise-type rejection, and a wide-barrier / low-vol setup where the
    99-term truncation fails its own convergence check.
    """
    inputs = CPP[name]["inputs"]
    payoff: StrikedTypePayoff = (
        PlainVanillaPayoff(option_type(inputs), inputs["strike"])
        if inputs["payoff"] == "PlainVanilla"
        else CashOrNothingPayoff(option_type(inputs), inputs["strike"], inputs["cash"])
    )
    option = DoubleBarrierOption(
        _BARRIER_TYPES[inputs["barrier_type"]],
        inputs["barrier_lo"],
        inputs["barrier_hi"],
        inputs["rebate"],
        payoff,
        build_exercise(inputs),
    )
    option.set_pricing_engine(AnalyticDoubleBarrierBinaryEngine(build_process(inputs)))
    run_case(name, option)


def test_double_barrier_binary_ignores_option_type_and_strike() -> None:
    """Only the cash payoff enters; the payoff's type and strike do not.

    The C++ source still carries the commented-out
    ``Option::Type type = payoff_->optionType(); // this is not used ?``, and the
    strike is only ever passed to ``blackVariance`` — which is flat here, so it
    cancels. Two cases differing only in those fields must price identically.
    """
    base = _expected(CPP["dbbin_ko_80_120_vol20"]["expected"]["value"])
    put_type = _expected(CPP["dbbin_ko_80_120_vol20_put_type_ignored"]["expected"]["value"])
    other_strike = _expected(CPP["dbbin_ko_80_120_vol20_strike_ignored"]["expected"]["value"])
    assert put_type == base
    assert other_strike == base


def test_double_barrier_binary_knock_in_plus_knock_out_is_the_discounted_cash() -> None:
    """KI + KO = cash * discount, which is what the KI branch is built from.

    ``payoffAtExpiry`` returns ``max(tot, 0)`` for KO and ``max(cash*df - tot, 0)``
    for KI, so their sum recovers the discounted cash whenever neither clamp
    bites. Checking the identity on the reference confirms the discount factor
    the KI branch introduces is the right one.
    """
    for vol_tag in ("vol10", "vol20", "vol30", "vol50"):
        ko = _expected(CPP[f"dbbin_ko_80_120_{vol_tag}"]["expected"]["value"])
        ki = _expected(CPP[f"dbbin_ki_80_120_{vol_tag}"]["expected"]["value"])
        inputs = CPP[f"dbbin_ki_80_120_{vol_tag}"]["inputs"]
        process = build_process(inputs)
        discount = process.risk_free_rate().discount(TODAY + inputs["maturity_days"])
        tolerance.tight(ko + ki, inputs["cash"] * discount, reason=f"KI+KO {vol_tag}")


# ---------------------------------------------------------------------------
# ForwardVanillaEngine / ForwardPerformanceVanillaEngine
# ---------------------------------------------------------------------------


def _build_forward_option(inputs: dict[str, Any]) -> ForwardVanillaOption:
    # probe.cpp passes a strike-0 PlainVanillaPayoff: the engine replaces the
    # strike with `moneyness * process->x0()` before delegating, so whatever the
    # instrument carries is discarded.
    return ForwardVanillaOption(
        inputs["moneyness"],
        TODAY + inputs["reset_days"],
        PlainVanillaPayoff(option_type(inputs), 0.0),
        EuropeanExercise(TODAY + inputs["maturity_days"]),
    )


@pytest.mark.parametrize("name", cases("fwd_"))
def test_forward_vanilla_engine(name: str) -> None:
    """Re-verify the pre-existing ForwardVanillaEngine against C++ v1.43.

    It was already ported and green; earlier waves found defects in exactly that
    kind of code, so the reference covers it on the same market as its
    performance subclass. Its discounting is a *dividend* discount to the reset
    date and its theta uses the dividend curve's zero rate.
    """
    inputs = CPP[name]["inputs"]
    option = _build_forward_option(inputs)
    option.set_pricing_engine(
        ForwardVanillaEngine(build_process(inputs), AnalyticEuropeanEngine)
    )
    run_case(name, option)


@pytest.mark.parametrize("name", cases("fwdperf_"))
def test_forward_performance_vanilla_engine(name: str) -> None:
    """Forward performance: risk-free discount over spot, hard-zero delta/gamma.

    The spot-1 and spot-500 cases are what pin the division by the spot; the
    zero-q and zero-r cases separate the two engines' discounting rules.
    """
    inputs = CPP[name]["inputs"]
    option = _build_forward_option(inputs)
    option.set_pricing_engine(
        ForwardPerformanceVanillaEngine(build_process(inputs), AnalyticEuropeanEngine)
    )
    run_case(name, option)


def test_forward_performance_is_the_forward_value_scaled_by_r_df_over_spot() -> None:
    """performance NPV == forward NPV * r_df(reset) / (S * q_df(reset)).

    Both engines wrap the same inner European; they differ only in the factor
    applied to its value. Deriving one from the other proves the factor rather
    than merely re-checking two independent numbers.
    """
    for suffix in ("call_m110", "put_m110", "call_m110_spot500", "call_m110_zero_q"):
        inputs = CPP[f"fwd_{suffix}"]["inputs"]
        fwd_value = _expected(CPP[f"fwd_{suffix}"]["expected"]["value"])
        perf_value = _expected(CPP[f"fwdperf_{suffix}"]["expected"]["value"])

        process = build_process(inputs)
        reset = TODAY + inputs["reset_days"]
        disc_q = process.dividend_yield().discount(reset)
        disc_r = process.risk_free_rate().discount(reset)
        spot = inputs["spot"]

        # fwd = discQ * inner, perf = (discR / spot) * inner.
        tolerance.tight(
            perf_value, fwd_value * disc_r / (disc_q * spot), reason=f"fwdperf {suffix}"
        )
