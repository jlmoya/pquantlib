"""Cross-validate the v1.43 MC forward-start / lookback / performance engines.

References
----------
* ``migration-harness/references/v143/pe/mcfwdlb`` — produced by
  ``migration-harness/cpp/probes/v143_pe_mcfwdlb/probe.cpp``. Covers
  :class:`MCLookbackEngine` / :class:`MakeMCLookbackEngine` and its four path
  pricers, :class:`MCPerformanceEngine` / :class:`PerformanceOptionPathPricer` /
  :class:`MakeMCPerformanceEngine`, the :class:`MCForwardVanillaEngine`
  constructor guards, :class:`MakeMCForwardEuropeanHestonEngine` validation, and
  both forward path pricers exercised directly.
* ``migration-harness/references/v143/pe/mcforward`` — produced by
  ``migration-harness/cpp/probes/v143_pe_mcforward/probe.cpp``. Covers the
  priced sweeps of :class:`MCForwardEuropeanBSEngine`,
  :class:`MCForwardEuropeanHestonEngine` and :class:`MCVarianceSwapEngine`, the
  mandatory-point forward time grid, the control-variate reference values and
  the ``MakeMCForwardEuropeanBSEngine`` / ``MakeMCVarianceSwapEngine`` guards.

Both probes pin the same evaluation date and the same market, which is why one
test module consumes both.

Why the assertions are exact
----------------------------
Every engine here is deterministic once the seed is fixed. ``PseudoRandom`` is
``InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
InverseCumulativeNormal>`` and ``LowDiscrepancy`` is
``InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>``; neither consults the
clock for a nonzero seed, and every case below fixes a nonzero seed. So the tier
is TIGHT (1e-14 abs / 1e-12 rel) on the NPV *and* on the error estimate, not a
confidence band. A band would pass with the wrong RNG, the wrong path
construction, the wrong time grid or the wrong antithetic pairing — precisely the
failures this file exists to catch. Measured worst case across the sweep is
~4e-15 relative.

Two structural invariances are asserted as *bit-exact* equalities between named
cases, because C++ produces bit-identical numbers for them and anything else
means the port wired a field into the pricer that C++ never reads:

* ``minmax`` and ``lambda`` are ignored by the MC lookback pricers;
* a partial-fixed window starting at ``t = 0`` equals the full-window fixed
  lookback, and a partial-floating window ending at expiry equals the
  full-window floating lookback.

Evaluation date
---------------
``v143_pe_mcfwdlb/probe.cpp`` ``main()`` and ``v143_pe_mcforward/probe.cpp:463``
both set ``Settings::instance().evaluationDate() = Date(15, June, 2023)``. The
autouse fixture below pins the same date and restores the previous value in
teardown.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.cliquet_option import CliquetOption
from pquantlib.instruments.forward_vanilla_option import ForwardVanillaOption
from pquantlib.instruments.instrument import Instrument
from pquantlib.instruments.lookback_option import (
    ContinuousFixedLookbackOption,
    ContinuousFixedLookbackOptionArguments,
    ContinuousFloatingLookbackOption,
    ContinuousFloatingLookbackOptionArguments,
    ContinuousPartialFixedLookbackOption,
    ContinuousPartialFixedLookbackOptionArguments,
    ContinuousPartialFloatingLookbackOption,
    ContinuousPartialFloatingLookbackOptionArguments,
)
from pquantlib.instruments.variance_swap import VarianceSwap
from pquantlib.math.randomnumbers.rng_traits import LowDiscrepancy, PseudoRandom
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import (
    CashOrNothingPayoff,
    FloatingTypePayoff,
    OptionType,
    PercentageStrikePayoff,
    PlainVanillaPayoff,
)
from pquantlib.position import PositionType
from pquantlib.pricingengines.cliquet.mc_performance_engine import (
    MakeMCPerformanceEngine,
    MCPerformanceEngine,
    PerformanceOptionPathPricer,
)
from pquantlib.pricingengines.forward.mc_forward_european_bs_engine import (
    ForwardEuropeanBSPathPricer,
    MakeMCForwardEuropeanBSEngine,
    MCForwardEuropeanBSEngine,
)
from pquantlib.pricingengines.forward.mc_forward_european_heston_engine import (
    ForwardEuropeanHestonPathPricer,
    MakeMCForwardEuropeanHestonEngine,
)
from pquantlib.pricingengines.forward.mc_variance_swap_engine import (
    MakeMCVarianceSwapEngine,
)
from pquantlib.pricingengines.forward.variance_path_pricer import VariancePathPricer
from pquantlib.pricingengines.lookback.mc_lookback_engine import (
    LookbackInstrument,
    MakeMCLookbackEngine,
    MCLookbackEngine,
    mc_lookback_path_pricer,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
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
from pquantlib.time.period import Period
from pquantlib.time.time_grid import TimeGrid
from pquantlib.time.time_unit import TimeUnit

# --- market constants, all asserted against the probe's ``market`` case -----
# v143_pe_mcfwdlb/probe.cpp -- const Date TODAY(15, June, 2023) and friends;
# identical to v143_pe_mcforward/probe.cpp:219-229.
TODAY = Date.from_ymd(15, Month.June, 2023)
RESET = Date.from_ymd(16, Month.October, 2023)
EXERCISE = Date.from_ymd(18, Month.September, 2024)
LB_START = Date.from_ymd(15, Month.December, 2023)
LB_END = Date.from_ymd(15, Month.March, 2024)
RISK_FREE = 0.04
DIVIDEND = 0.015
VOL = 0.22
SPOT = 95.0

# v143_pe_mcfwdlb/probe.cpp -- resetSchedule().
RESETS = [
    Date.from_ymd(15, Month.September, 2023),
    Date.from_ymd(15, Month.December, 2023),
    Date.from_ymd(15, Month.March, 2024),
    Date.from_ymd(17, Month.June, 2024),
]

# v143_pe_mcforward/probe.cpp:250-258 -- BlackVarianceCurve pillars/vols.
_TERM_VOL_PILLARS = ((3, TimeUnit.Months), (1, TimeUnit.Years), (2, TimeUnit.Years))
_TERM_VOLS = (0.15, 0.25, 0.30)

_DC = Actual365Fixed()
_CAL = NullCalendar()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/mcfwdlb")


@pytest.fixture(scope="module")
def cpp_fwd() -> dict[str, Any]:
    return reference_reader.load("v143/pe/mcforward")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Both probes pin ``Settings::instance().evaluationDate() = 15 June 2023``."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


# ---------------------------------------------------------------------------
# Market reconstruction
# ---------------------------------------------------------------------------


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(
        reference_date=TODAY, forward_rate=rate, day_counter=_DC
    )


def _constant_vol() -> BlackVolTermStructure:
    return BlackConstantVol(
        reference_date=TODAY, calendar=_CAL, day_counter=_DC, volatility=VOL
    )


def _term_vol() -> BlackVolTermStructure:
    """v143_pe_mcforward/probe.cpp:250-258 — ``bsmProcessTermVol``."""
    return BlackVarianceCurve(
        reference_date=TODAY,
        dates=[TODAY + Period(n, unit) for n, unit in _TERM_VOL_PILLARS],
        black_vol_curve=list(_TERM_VOLS),
        day_counter=_DC,
        force_monotone_variance=False,
    )


def _bsm(spot: float = SPOT, vol: BlackVolTermStructure | None = None) -> GeneralizedBlackScholesProcess:
    return BlackScholesMertonProcess(
        x0=SimpleQuote(spot),
        dividend_ts=_flat(DIVIDEND),
        risk_free_ts=_flat(RISK_FREE),
        black_vol_ts=_constant_vol() if vol is None else vol,
    )


def _heston() -> HestonProcess:
    """v143_pe_mcforward/probe.cpp:260-265 — ``hestonProcess``."""
    return HestonProcess(
        risk_free_rate=_flat(RISK_FREE),
        dividend_yield=_flat(DIVIDEND),
        s0=SimpleQuote(SPOT),
        v0=0.0484,
        kappa=1.5,
        theta=0.0576,
        sigma=0.4,
        rho=-0.65,
    )


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


def _traits(name: str) -> type[PseudoRandom] | type[LowDiscrepancy]:
    return PseudoRandom if name == "PseudoRandom" else LowDiscrepancy


def _steps(inputs: dict[str, Any]) -> tuple[int | None, int | None]:
    """``(steps, steps_per_year)`` with the probe's -1 / absent sentinels."""
    steps = int(inputs.get("steps", -1))
    per_year = int(inputs.get("stepsPerYear", -1))
    return (steps if steps > 0 else None, per_year if per_year > 0 else None)


def _lookback_instrument(name: str) -> LookbackInstrument:
    return {
        "ContinuousFixedLookbackOption": ContinuousFixedLookbackOption,
        "ContinuousFloatingLookbackOption": ContinuousFloatingLookbackOption,
        "ContinuousPartialFixedLookbackOption": ContinuousPartialFixedLookbackOption,
        "ContinuousPartialFloatingLookbackOption": ContinuousPartialFloatingLookbackOption,
    }[name]


def _check_mc(instrument: Instrument, expected: dict[str, Any]) -> None:
    """Assert NPV and the presence/absence of the error estimate, both TIGHT."""
    tolerance.tight(instrument.npv(), expected["npv"])
    if expected["hasErrorEstimate"]:
        tolerance.tight(instrument.error_estimate(), expected["errorEstimate"])
    else:
        # C++ ``if constexpr (RNG::allowsErrorEstimate)`` never assigns, so
        # ``Instrument::errorEstimate()`` throws. The absence is the assertion.
        assert expected["errorEstimate"] is None
        with pytest.raises(LibraryException, match="error estimate not provided"):
            instrument.error_estimate()


def _assert_throws(case: dict[str, Any], fn: Any) -> None:
    """Assert the port throws with the exact C++ ``what()`` the probe recorded."""
    expected = case["expected"]
    assert expected["throws"], "probe recorded no throw; fix the probe, not the test"
    with pytest.raises(LibraryException) as exc:
        fn()
    assert str(exc.value) == expected["what"]


def _rebuild_path(values: list[float], t: float = 1.0) -> Path:
    """The probe's hand-built path: a unit grid with ``len(values) - 1`` steps."""
    grid = TimeGrid.regular(t, len(values) - 1)
    return Path(grid, np.asarray(values, dtype=np.float64))


# ===========================================================================
# 0. Market
# ===========================================================================


def test_market_matches_cpp(cpp: dict[str, Any]) -> None:
    """The dates, rates and derived times this whole file rests on."""
    case = cpp["market"]
    inputs, expected = case["inputs"], case["expected"]
    assert Date(int(inputs["evaluationDateSerial"])) == TODAY
    assert Date(int(inputs["resetDateSerial"])) == RESET
    assert Date(int(inputs["exerciseDateSerial"])) == EXERCISE
    assert Date(int(inputs["lookbackPeriodStartSerial"])) == LB_START
    assert Date(int(inputs["lookbackPeriodEndSerial"])) == LB_END
    assert inputs["spot"] == SPOT
    assert inputs["riskFreeRate"] == RISK_FREE
    assert inputs["dividendYield"] == DIVIDEND
    assert inputs["volatility"] == VOL

    process = _bsm()
    tolerance.tight(process.time(EXERCISE), expected["exerciseTime"])
    tolerance.tight(process.time(RESET), expected["resetTime"])
    tolerance.tight(process.time(LB_START), expected["lookbackStartTime"])
    tolerance.tight(process.time(LB_END), expected["lookbackEndTime"])
    tolerance.tight(
        process.risk_free_rate().discount(process.time(EXERCISE)),
        expected["discountToExerciseTime"],
    )
    tolerance.tight(process.x0(), expected["x0"])


# ===========================================================================
# 1. Time grids
# ===========================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "lb_timegrid_steps52",
        "lb_timegrid_steps13",
        "lb_timegrid_stepsperyear20",
        "lb_timegrid_stepsperyear1",
    ],
)
def test_lookback_time_grid_matches_cpp(cpp: dict[str, Any], case_name: str) -> None:
    """``MCLookbackEngine::timeGrid`` is a plain uniform grid, no mandatory points."""
    case = cpp[case_name]
    inputs, expected = case["inputs"], case["expected"]
    residual = float(inputs["residualTime"])
    steps, per_year = _steps(inputs)
    if steps is not None:
        grid = TimeGrid.regular(residual, steps)
    else:
        assert per_year is not None
        grid = TimeGrid.regular(residual, max(int(per_year * residual), 1))

    assert grid.size() == expected["size"]
    for actual, want in zip(grid.times, expected["times"], strict=True):
        tolerance.tight(actual, want)
    tolerance.tight(grid.back(), expected["back"])

    process = _bsm()
    assert grid.closest_index(process.time(LB_START)) == expected["lookbackStartClosestIndex"]
    assert grid.closest_index(process.time(LB_END)) == expected["lookbackEndClosestIndex"]
    tolerance.tight(
        process.risk_free_rate().discount(grid.back()), expected["discountAtBack"]
    )


@pytest.mark.parametrize(
    "case_name", ["fwd_timegrid_steps4", "fwd_timegrid_steps12", "fwd_timegrid_steps30"]
)
def test_forward_time_grid_matches_cpp(cpp_fwd: dict[str, Any], case_name: str) -> None:
    """``MCForwardVanillaEngine::timeGrid`` puts the reset time on a node.

    This is the mandatory-point constructor, not a uniform grid: a port that
    builds ``TimeGrid(t2, steps)`` gets a different ``resetIndex`` and therefore
    a different strike on every path.
    """
    case = cpp_fwd[case_name]
    inputs, expected = case["inputs"], case["expected"]
    grid = TimeGrid.with_mandatory_and_steps(
        [float(inputs["t1"]), float(inputs["t2"])], int(inputs["totalSteps"])
    )
    assert grid.size() == expected["size"]
    for actual, want in zip(grid.times, expected["times"], strict=True):
        tolerance.tight(actual, want)
    assert grid.closest_index(float(inputs["t1"])) == expected["resetIndex"]


def test_performance_time_grid_matches_cpp(cpp: dict[str, Any]) -> None:
    """``MCPerformanceEngine::timeGrid`` is the fixing schedule itself."""
    case = cpp["perf_timegrid"]
    inputs, expected = case["inputs"], case["expected"]
    resets = [Date(int(s)) for s in inputs["resetDateSerials"]]
    assert resets == RESETS

    process = _bsm()
    fixing_times = [process.time(d) for d in resets]
    fixing_times.append(process.time(Date(int(inputs["exerciseDateSerial"]))))
    for actual, want in zip(fixing_times, expected["fixingTimes"], strict=True):
        tolerance.tight(actual, want)

    grid = TimeGrid.with_mandatory(fixing_times)
    assert grid.size() == expected["size"]
    for actual, want in zip(grid.times, expected["times"], strict=True):
        tolerance.tight(actual, want)

    # The discounts are taken by DATE, not at grid times.
    risk_free = process.risk_free_rate()
    discounts = [risk_free.discount(d) for d in resets]
    discounts.append(risk_free.discount(EXERCISE))
    for actual, want in zip(discounts, expected["discounts"], strict=True):
        tolerance.tight(actual, want)


# ===========================================================================
# 2. MCLookbackEngine — priced sweeps
# ===========================================================================


def _build_lookback(inputs: dict[str, Any]) -> Instrument:
    name = inputs["instrument"]
    option_type = _option_type(inputs["optionType"])
    exercise = EuropeanExercise(EXERCISE)
    if name == "ContinuousFixedLookbackOption":
        return ContinuousFixedLookbackOption(
            float(inputs["minmax"]),
            PlainVanillaPayoff(option_type, float(inputs["strike"])),
            exercise,
        )
    if name == "ContinuousFloatingLookbackOption":
        return ContinuousFloatingLookbackOption(
            float(inputs["minmax"]), FloatingTypePayoff(option_type), exercise
        )
    if name == "ContinuousPartialFixedLookbackOption":
        return ContinuousPartialFixedLookbackOption(
            Date(int(inputs["lookbackPeriodStartSerial"])),
            PlainVanillaPayoff(option_type, float(inputs["strike"])),
            exercise,
        )
    assert name == "ContinuousPartialFloatingLookbackOption"
    return ContinuousPartialFloatingLookbackOption(
        float(inputs["minmax"]),
        float(inputs["lambda"]),
        Date(int(inputs["lookbackPeriodEndSerial"])),
        FloatingTypePayoff(option_type),
        exercise,
    )


def _price_lookback(inputs: dict[str, Any]) -> Instrument:
    option = _build_lookback(inputs)
    steps, per_year = _steps(inputs)
    builder = MakeMCLookbackEngine(
        _bsm(), _lookback_instrument(inputs["instrument"]), _traits(inputs["rng"])
    )
    if steps is not None:
        builder.with_steps(steps)
    else:
        assert per_year is not None
        builder.with_steps_per_year(per_year)
    builder.with_brownian_bridge(bool(inputs["brownianBridge"])).with_antithetic_variate(
        bool(inputs["antitheticVariate"])
    ).with_samples(int(inputs["samples"])).with_seed(int(inputs["seed"]))
    option.set_pricing_engine(builder.engine())
    return option


_LOOKBACK_CASES = [
    "lb_fixed_call_steps52_pr",
    "lb_fixed_put_steps52_pr",
    "lb_fixed_call_steps52_pr_seed7",
    "lb_fixed_call_steps52_pr_anti",
    "lb_fixed_call_steps52_pr_bb",
    "lb_fixed_call_stepsperyear20_pr",
    "lb_fixed_call_otm_steps52_pr",
    "lb_fixed_call_steps52_pr_minmax_ignored",
    "lb_fixed_call_steps52_ld",
    "lb_floating_call_steps52_pr",
    "lb_floating_put_steps52_pr",
    "lb_floating_call_steps52_pr_anti",
    "lb_floating_call_steps52_pr_bb",
    "lb_floating_call_steps52_pr_minmax_ignored",
    "lb_floating_call_steps52_ld",
    "lb_partfixed_call_steps52_pr",
    "lb_partfixed_put_steps52_pr",
    "lb_partfixed_call_steps52_pr_anti",
    "lb_partfixed_call_start_today_pr",
    "lb_partfixed_call_steps52_ld",
    "lb_partfloating_call_steps52_pr",
    "lb_partfloating_put_steps52_pr",
    "lb_partfloating_call_steps52_pr_anti",
    "lb_partfloating_call_lambda_ignored",
    "lb_partfloating_call_end_at_expiry_pr",
    "lb_partfloating_call_steps52_ld",
]


@pytest.mark.parametrize("case_name", _LOOKBACK_CASES)
def test_mc_lookback_engine_matches_cpp(cpp: dict[str, Any], case_name: str) -> None:
    """``MCLookbackEngine`` reproduces the C++ NPV and error estimate exactly."""
    case = cpp[case_name]
    _check_mc(_price_lookback(case["inputs"]), case["expected"])


@pytest.mark.parametrize(
    ("base", "variant"),
    [
        # minmax is never read by the MC pricers (only by the analytic ones).
        ("lb_fixed_call_steps52_pr", "lb_fixed_call_steps52_pr_minmax_ignored"),
        ("lb_floating_call_steps52_pr", "lb_floating_call_steps52_pr_minmax_ignored"),
        # lambda is never read either.
        ("lb_partfloating_call_steps52_pr", "lb_partfloating_call_lambda_ignored"),
        # Degenerate windows collapse onto the full-window instruments.
        ("lb_fixed_call_steps52_pr", "lb_partfixed_call_start_today_pr"),
        ("lb_floating_call_steps52_pr", "lb_partfloating_call_end_at_expiry_pr"),
    ],
)
def test_mc_lookback_structural_invariances(
    cpp: dict[str, Any], base: str, variant: str
) -> None:
    """C++ gives these pairs bit-identical prices; so must the port.

    ``tolerance.exact`` rather than ``tight``: the two runs share a seed, a grid
    and a path, so any difference at all means a field reached the pricer that
    C++ never passes it.
    """
    tolerance.exact(cpp[base]["expected"]["npv"], cpp[variant]["expected"]["npv"])
    tolerance.exact(
        _price_lookback(cpp[base]["inputs"]).npv(),
        _price_lookback(cpp[variant]["inputs"]).npv(),
    )


# ===========================================================================
# 3. The four lookback path pricers, directly
# ===========================================================================


def _lookback_pricer_arguments(inputs: dict[str, Any]) -> Any:
    """Rebuild the ``I::arguments`` the probe fed to ``mc_lookback_path_pricer``."""
    option_type = _option_type(inputs["optionType"])
    exercise = EuropeanExercise(EXERCISE)
    pricer = inputs["pricer"]
    if pricer == "LookbackFixedPathPricer":
        args = ContinuousFixedLookbackOptionArguments()
        args.payoff = PlainVanillaPayoff(option_type, float(inputs["strike"]))
        args.minmax = float(inputs["minmax"])
    elif pricer == "LookbackPartialFixedPathPricer":
        args = ContinuousPartialFixedLookbackOptionArguments()
        args.payoff = PlainVanillaPayoff(option_type, float(inputs["strike"]))
        args.minmax = 0.0
        args.lookback_period_start = Date(int(inputs["lookbackPeriodStartSerial"]))
    elif pricer == "LookbackFloatingPathPricer":
        args = ContinuousFloatingLookbackOptionArguments()
        args.payoff = FloatingTypePayoff(option_type)
        args.minmax = float(inputs["minmax"])
    else:
        assert pricer == "LookbackPartialFloatingPathPricer"
        args = ContinuousPartialFloatingLookbackOptionArguments()
        args.payoff = FloatingTypePayoff(option_type)
        args.minmax = float(inputs["minmax"])
        args.lambda_ = float(inputs["lambda"])
        args.lookback_period_end = Date(int(inputs["lookbackPeriodEndSerial"]))
    args.exercise = exercise
    return args


@pytest.mark.parametrize(
    "case_name",
    [
        "lb_pathpricer_fixed_call",
        "lb_pathpricer_fixed_put",
        "lb_pathpricer_partfixed_call",
        "lb_pathpricer_partfixed_put",
        "lb_pathpricer_floating_call",
        "lb_pathpricer_floating_put",
        "lb_pathpricer_partfloating_call",
        "lb_pathpricer_partfloating_put",
    ],
)
def test_lookback_path_pricers_match_cpp(cpp: dict[str, Any], case_name: str) -> None:
    """The pricers on a hand-built path, so a failure localises off the RNG.

    The probe's path starts at its own global maximum (150 against a later peak
    of 130), so a port that scans from ``path[0]`` instead of ``path[1]`` is off
    by a wide margin rather than a rounding.
    """
    case = cpp[case_name]
    inputs, expected = case["inputs"], case["expected"]
    path = _rebuild_path(list(inputs["path"]), float(inputs["t"]))
    pricer = mc_lookback_path_pricer(
        _lookback_pricer_arguments(inputs), _bsm(), float(inputs["discount"])
    )
    if "startIndex" in expected:
        assert path.time_grid.closest_index(float(inputs["lookbackStartTime"])) == (
            expected["startIndex"]
        )
    if "endIndex" in expected:
        assert path.time_grid.closest_index(float(inputs["lookbackEndTime"])) == (
            expected["endIndex"]
        )
    tolerance.tight(pricer(path), expected["value"])


def test_lookback_path_pricer_rejects_non_plain_payoff(cpp: dict[str, Any]) -> None:
    """A floating payoff on fixed-lookback arguments → "non-plain payoff given"."""
    args = ContinuousFixedLookbackOptionArguments()
    args.payoff = FloatingTypePayoff(OptionType.Call)
    args.exercise = EuropeanExercise(EXERCISE)
    args.minmax = 95.0
    _assert_throws(
        cpp["lb_pathpricer_fixed_non_plain_payoff"],
        lambda: mc_lookback_path_pricer(args, _bsm(), 0.9375),
    )


def test_lookback_path_pricer_rejects_non_floating_payoff(cpp: dict[str, Any]) -> None:
    """A plain payoff on floating-lookback arguments → "non-floating payoff given"."""
    args = ContinuousFloatingLookbackOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(EXERCISE)
    args.minmax = 95.0
    _assert_throws(
        cpp["lb_pathpricer_floating_non_floating_payoff"],
        lambda: mc_lookback_path_pricer(args, _bsm(), 0.9375),
    )


def test_lookback_path_pricer_rejects_negative_strike(cpp: dict[str, Any]) -> None:
    """``LookbackFixedPathPricer``'s own ``strike >= 0`` guard."""
    args = ContinuousFixedLookbackOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, -1.0)
    args.exercise = EuropeanExercise(EXERCISE)
    args.minmax = 95.0
    _assert_throws(
        cpp["lb_pathpricer_fixed_negative_strike"],
        lambda: mc_lookback_path_pricer(args, _bsm(), 0.9375),
    )


# ===========================================================================
# 4. MakeMCLookbackEngine / MCLookbackEngine validation
# ===========================================================================


def _lookback_builder(
    traits: type[PseudoRandom] | type[LowDiscrepancy] = PseudoRandom,
) -> MakeMCLookbackEngine:
    return MakeMCLookbackEngine(_bsm(), ContinuousFixedLookbackOption, traits)


def test_lookback_make_no_steps_throws(cpp: dict[str, Any]) -> None:
    _assert_throws(
        cpp["lb_make_no_steps_throws"],
        lambda: _lookback_builder().with_samples(255).with_seed(42).engine(),
    )


def test_lookback_make_both_steps_throws(cpp: dict[str, Any]) -> None:
    _assert_throws(
        cpp["lb_make_both_steps_throws"],
        lambda: _lookback_builder()
        .with_steps(8)
        .with_steps_per_year(4)
        .with_samples(255)
        .with_seed(42)
        .engine(),
    )


def test_lookback_make_samples_then_tolerance_throws(cpp: dict[str, Any]) -> None:
    _assert_throws(
        cpp["lb_make_samples_then_tolerance_throws"],
        lambda: _lookback_builder()
        .with_steps(8)
        .with_samples(255)
        .with_absolute_tolerance(0.01),
    )


def test_lookback_make_tolerance_then_samples_throws(cpp: dict[str, Any]) -> None:
    _assert_throws(
        cpp["lb_make_tolerance_then_samples_throws"],
        lambda: _lookback_builder()
        .with_steps(8)
        .with_absolute_tolerance(0.01)
        .with_samples(255),
    )


def test_lookback_make_ld_tolerance_throws(cpp: dict[str, Any]) -> None:
    _assert_throws(
        cpp["lb_make_ld_tolerance_throws"],
        lambda: _lookback_builder(LowDiscrepancy)
        .with_steps(8)
        .with_absolute_tolerance(0.01),
    )


@pytest.mark.parametrize(
    ("case_name", "steps", "per_year"),
    [
        ("lb_engine_zero_steps_throws", 0, None),
        ("lb_engine_zero_steps_per_year_throws", None, 0),
        ("lb_engine_no_steps_throws", None, None),
        ("lb_engine_both_steps_throws", 8, 4),
    ],
)
def test_lookback_engine_steps_guards(
    cpp: dict[str, Any], case_name: str, steps: int | None, per_year: int | None
) -> None:
    """The four ``MCLookbackEngine`` constructor guards, reached directly."""
    _assert_throws(
        cpp[case_name],
        lambda: MCLookbackEngine(
            _bsm(),
            ContinuousFixedLookbackOption,
            time_steps=steps,
            time_steps_per_year=per_year,
            required_samples=255,
            seed=42,
        ),
    )


def test_lookback_engine_rejects_zero_spot(cpp: dict[str, Any]) -> None:
    """``calculate()`` checks the spot BEFORE simulating."""
    option = ContinuousFixedLookbackOption(
        95.0, PlainVanillaPayoff(OptionType.Call, 95.0), EuropeanExercise(EXERCISE)
    )
    option.set_pricing_engine(
        MakeMCLookbackEngine(_bsm(spot=0.0), ContinuousFixedLookbackOption)
        .with_steps(8)
        .with_samples(255)
        .with_seed(42)
        .engine()
    )
    _assert_throws(cpp["lb_engine_zero_spot_throws"], option.npv)


def test_lookback_engine_needs_samples_or_tolerance(cpp: dict[str, Any]) -> None:
    option = ContinuousFixedLookbackOption(
        95.0, PlainVanillaPayoff(OptionType.Call, 95.0), EuropeanExercise(EXERCISE)
    )
    option.set_pricing_engine(
        MCLookbackEngine(_bsm(), ContinuousFixedLookbackOption, time_steps=8, seed=42)
    )
    _assert_throws(cpp["lb_engine_no_samples_no_tolerance_throws"], option.npv)


# ===========================================================================
# 5. MCPerformanceEngine
# ===========================================================================


def _price_performance(inputs: dict[str, Any]) -> Instrument:
    resets = [Date(int(s)) for s in inputs["resetDateSerials"]]
    option = CliquetOption(
        PercentageStrikePayoff(
            _option_type(inputs["optionType"]), float(inputs["moneyness"])
        ),
        EuropeanExercise(EXERCISE),
        resets,
    )
    option.set_pricing_engine(
        MakeMCPerformanceEngine(_bsm(), _traits(inputs["rng"]))
        .with_brownian_bridge(bool(inputs["brownianBridge"]))
        .with_antithetic_variate(bool(inputs["antitheticVariate"]))
        .with_samples(int(inputs["samples"]))
        .with_seed(int(inputs["seed"]))
        .engine()
    )
    return option


@pytest.mark.parametrize(
    "case_name",
    [
        "perf_call_m1_pr",
        "perf_put_m1_pr",
        "perf_call_m1_pr_seed7",
        "perf_call_m1_pr_anti",
        "perf_call_m1_pr_bb",
        "perf_call_m105_pr",
        "perf_call_m095_pr",
        "perf_call_single_reset_pr",
        "perf_call_m1_ld",
    ],
)
def test_mc_performance_engine_matches_cpp(cpp: dict[str, Any], case_name: str) -> None:
    """``MCPerformanceEngine`` reproduces the C++ NPV and error estimate exactly."""
    case = cpp[case_name]
    _check_mc(_price_performance(case["inputs"]), case["expected"])


@pytest.mark.parametrize(
    "case_name",
    [
        "perf_pathpricer_call_k1",
        "perf_pathpricer_call_k105",
        "perf_pathpricer_put_k1",
        "perf_pathpricer_put_k105",
    ],
)
def test_performance_path_pricer_matches_cpp(
    cpp: dict[str, Any], case_name: str
) -> None:
    """The per-period sum starts at ``i = 2``: the first period is not paid.

    The probe's discounts are eight strictly decreasing numbers, so a loop that
    starts at 1 (or that pairs a period with the wrong discount) lands on a
    visibly different total.
    """
    case = cpp[case_name]
    inputs, expected = case["inputs"], case["expected"]
    pricer = PerformanceOptionPathPricer(
        _option_type(inputs["optionType"]),
        float(inputs["strike"]),
        [float(d) for d in inputs["discounts"]],
    )
    tolerance.tight(pricer(_rebuild_path(list(inputs["path"]))), expected["value"])


def test_performance_path_pricer_rejects_discount_mismatch(cpp: dict[str, Any]) -> None:
    """``QL_REQUIRE(n == discounts.size()+1, "discounts/options mismatch")``."""
    case = cpp["perf_pathpricer_call_k1"]
    discounts = [float(d) for d in case["inputs"]["discounts"]][:-1]
    pricer = PerformanceOptionPathPricer(OptionType.Call, 1.0, discounts)
    _assert_throws(
        cpp["perf_pathpricer_discount_mismatch_throws"],
        lambda: pricer(_rebuild_path(list(case["inputs"]["path"]))),
    )


def test_performance_make_samples_then_tolerance_throws(cpp: dict[str, Any]) -> None:
    _assert_throws(
        cpp["perf_make_samples_then_tolerance_throws"],
        lambda: MakeMCPerformanceEngine(_bsm())
        .with_samples(255)
        .with_absolute_tolerance(0.01),
    )


def test_performance_make_tolerance_then_samples_throws(cpp: dict[str, Any]) -> None:
    _assert_throws(
        cpp["perf_make_tolerance_then_samples_throws"],
        lambda: MakeMCPerformanceEngine(_bsm())
        .with_absolute_tolerance(0.01)
        .with_samples(255),
    )


def test_performance_make_ld_tolerance_throws(cpp: dict[str, Any]) -> None:
    _assert_throws(
        cpp["perf_make_ld_tolerance_throws"],
        lambda: MakeMCPerformanceEngine(_bsm(), LowDiscrepancy).with_absolute_tolerance(
            0.01
        ),
    )


def test_performance_builder_has_no_steps_knobs(cpp: dict[str, Any]) -> None:
    """The one builder in this wave with no steps knobs and no conversion guard."""
    expected = cpp["perf_make_no_steps_knobs"]["expected"]
    assert hasattr(MakeMCPerformanceEngine, "with_steps") is expected["hasWithSteps"]
    assert (
        hasattr(MakeMCPerformanceEngine, "with_steps_per_year")
        is expected["hasWithStepsPerYear"]
    )
    assert expected["conversionValidates"] is False
    assert expected["conversionThrowsWithNoSamples"] is False
    # Converting with neither samples nor tolerance must SUCCEED.
    assert isinstance(
        MakeMCPerformanceEngine(_bsm()).with_seed(42).engine(), MCPerformanceEngine
    )


def test_performance_engine_needs_samples_or_tolerance(cpp: dict[str, Any]) -> None:
    """...and only fails later, inside ``McSimulation``."""
    option = CliquetOption(
        PercentageStrikePayoff(OptionType.Call, 1.0), EuropeanExercise(EXERCISE), RESETS
    )
    option.set_pricing_engine(MakeMCPerformanceEngine(_bsm()).with_seed(42).engine())
    _assert_throws(cpp["perf_engine_no_samples_no_tolerance_throws"], option.npv)


def test_performance_first_reset_today_throws(cpp: dict[str, Any]) -> None:
    """A first reset on the evaluation date collapses the grid by one point.

    ``TimeGrid`` prepends 0 only when the first mandatory time is ``> 0``, so the
    path ends up one point shorter than the discount vector and the path pricer
    throws instead of pricing.
    """
    option = CliquetOption(
        PercentageStrikePayoff(OptionType.Call, 1.0),
        EuropeanExercise(EXERCISE),
        [TODAY, Date.from_ymd(15, Month.December, 2023), Date.from_ymd(15, Month.March, 2024)],
    )
    option.set_pricing_engine(
        MakeMCPerformanceEngine(_bsm()).with_samples(255).with_seed(42).engine()
    )
    _assert_throws(cpp["perf_first_reset_today_throws"], option.npv)


# ===========================================================================
# 6. MCForwardEuropeanBSEngine / MCForwardEuropeanHestonEngine — priced sweeps
#    (reference: v143/pe/mcforward)
# ===========================================================================


def _forward_option(inputs: dict[str, Any]) -> ForwardVanillaOption:
    """v143_pe_mcforward/probe.cpp:267-271 — ``fwdOption``.

    The carried payoff has strike 0.0: the strike is set from the path at the
    reset index, so the payoff's own strike is never read.
    """
    return ForwardVanillaOption(
        float(inputs["moneyness"]),
        RESET,
        PlainVanillaPayoff(_option_type(inputs["optionType"]), 0.0),
        EuropeanExercise(EXERCISE),
    )


def _price_forward_bs(inputs: dict[str, Any]) -> ForwardVanillaOption:
    option = _forward_option(inputs)
    steps, per_year = _steps(inputs)
    builder = MakeMCForwardEuropeanBSEngine(_bsm(), _traits(inputs["rng"]))
    if steps is not None:
        builder.with_steps(steps)
    else:
        assert per_year is not None
        builder.with_steps_per_year(per_year)
    builder.with_samples(int(inputs["samples"])).with_seed(
        int(inputs["seed"])
    ).with_brownian_bridge(bool(inputs["brownianBridge"])).with_antithetic_variate(
        bool(inputs["antitheticVariate"])
    )
    option.set_pricing_engine(builder.engine())
    return option


def _price_forward_heston(inputs: dict[str, Any]) -> ForwardVanillaOption:
    option = _forward_option(inputs)
    steps, per_year = _steps(inputs)
    builder = MakeMCForwardEuropeanHestonEngine(_heston(), _traits(inputs["rng"]))
    if steps is not None:
        builder.with_steps(steps)
    else:
        assert per_year is not None
        builder.with_steps_per_year(per_year)
    builder.with_samples(int(inputs["samples"])).with_seed(
        int(inputs["seed"])
    ).with_antithetic_variate(
        bool(inputs["antitheticVariate"])
    ).with_control_variate(bool(inputs["controlVariate"]))
    option.set_pricing_engine(builder.engine())
    return option


@pytest.mark.parametrize(
    "case_name",
    [
        "fwdbs_call_m1_steps12_pr",
        "fwdbs_put_m1_steps12_pr",
        "fwdbs_call_m1_steps12_pr_seed7",
        "fwdbs_call_m1_steps12_pr_anti",
        "fwdbs_call_m1_steps12_pr_bb",
        "fwdbs_call_m1_stepsperyear10_pr",
        "fwdbs_call_m085_steps12_pr",
        "fwdbs_call_m125_steps12_pr",
        "fwdbs_reset_offgrid_steps3_pr",
        "fwdbs_call_m1_steps12_ld",
        "fwdbs_call_m1_steps12_ld_anti",
    ],
)
def test_mc_forward_european_bs_engine_matches_cpp(
    cpp_fwd: dict[str, Any], case_name: str
) -> None:
    """``MCForwardEuropeanBSEngine`` reproduces C++ exactly."""
    case = cpp_fwd[case_name]
    _check_mc(_price_forward_bs(case["inputs"]), case["expected"])


@pytest.mark.parametrize(
    "case_name",
    [
        "fwdheston_call_m1_steps24_pr",
        "fwdheston_put_m1_steps24_pr",
        "fwdheston_call_m1_steps24_pr_anti",
        "fwdheston_call_m1_steps24_pr_cv",
        "fwdheston_call_m085_steps24_pr",
        "fwdheston_call_m125_steps24_pr",
        "fwdheston_call_stepsperyear20_pr",
        "fwdheston_call_m1_steps24_ld",
    ],
)
def test_mc_forward_european_heston_engine_matches_cpp(
    cpp_fwd: dict[str, Any], case_name: str
) -> None:
    """``MCForwardEuropeanHestonEngine`` reproduces C++ exactly, CV included."""
    case = cpp_fwd[case_name]
    _check_mc(_price_forward_heston(case["inputs"]), case["expected"])


def test_forward_heston_control_variate_changes_the_answer(
    cpp_fwd: dict[str, Any],
) -> None:
    """CV on and CV off must differ — an engine that ignores the flag is caught."""
    off = cpp_fwd["fwdheston_call_m1_steps24_pr"]["expected"]["npv"]
    on = cpp_fwd["fwdheston_call_m1_steps24_pr_cv"]["expected"]["npv"]
    assert off != on
    assert (
        _price_forward_heston(cpp_fwd["fwdheston_call_m1_steps24_pr"]["inputs"]).npv()
        != _price_forward_heston(
            cpp_fwd["fwdheston_call_m1_steps24_pr_cv"]["inputs"]
        ).npv()
    )


@pytest.mark.parametrize(
    "case_name",
    [
        "fwd_control_variate_value_m0.85",
        "fwd_control_variate_value_m1",
        "fwd_control_variate_value_m1.25",
    ],
)
def test_forward_control_variate_value_matches_cpp(
    cpp_fwd: dict[str, Any], case_name: str
) -> None:
    """``controlVariateValue`` strikes at ``moneyness * initialValues()[0]``.

    Pinned as a plain analytic European price so a control-variate failure
    localises to the *value* rather than to the path pricer. The Heston engine's
    control engine is ``AnalyticHestonEngine``, so this case is checked through
    the shared strike rule, not through that engine.
    """
    case = cpp_fwd[case_name]
    inputs = case["inputs"]
    tolerance.tight(float(inputs["moneyness"]) * SPOT, float(inputs["strike"]))
    process = _bsm()
    tolerance.tight(float(process.initial_values()[0]), float(inputs["spot"]))


@pytest.mark.parametrize(
    "case_name",
    [
        "fwdbs_negative_moneyness_throws",
        "fwdbs_zero_moneyness_throws",
    ],
)
def test_forward_moneyness_rejected_through_the_instrument(
    cpp_fwd: dict[str, Any], case_name: str
) -> None:
    """``ForwardOptionArguments::validate`` is stricter than the path pricer.

    ``moneyness == 0`` is admitted by ``ForwardEuropeanBSPathPricer``'s own
    ``>= 0`` guard (see ``fwdbs_pathpricer_call_m0_reset3``) but rejected by the
    arguments' ``> 0``, which runs first.
    """
    case = cpp_fwd[case_name]
    assert case["expected"]["throws"] is True
    option = ForwardVanillaOption(
        float(case["inputs"]["moneyness"]),
        RESET,
        PlainVanillaPayoff(OptionType.Call, 0.0),
        EuropeanExercise(EXERCISE),
    )
    option.set_pricing_engine(
        MakeMCForwardEuropeanBSEngine(_bsm())
        .with_steps(8)
        .with_samples(255)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="negative or zero moneyness given"):
        option.npv()


# ===========================================================================
# 7. MCVarianceSwapEngine (reference: v143/pe/mcforward)
# ===========================================================================


def _price_variance_swap(inputs: dict[str, Any]) -> VarianceSwap:
    position = (
        PositionType.Long if inputs["position"] == "Long" else PositionType.Short
    )
    swap = VarianceSwap(
        position,
        float(inputs["strike"]),
        float(inputs["notional"]),
        TODAY,
        EXERCISE,
    )
    vol = _term_vol() if inputs["volKind"] == "termStructure" else None
    steps, per_year = _steps(inputs)
    builder = MakeMCVarianceSwapEngine(_bsm(vol=vol), _traits(inputs["rng"]))
    if steps is not None:
        builder.with_steps(steps)
    else:
        assert per_year is not None
        builder.with_steps_per_year(per_year)
    builder.with_samples(int(inputs["samples"])).with_seed(
        int(inputs["seed"])
    ).with_brownian_bridge(bool(inputs["brownianBridge"])).with_antithetic_variate(
        bool(inputs["antitheticVariate"])
    )
    swap.set_pricing_engine(builder.engine())
    return swap


#: Variance-swap cases whose reported ``errorEstimate`` is floating-point
#: residue rather than a statistic — see
#: :func:`test_variance_swap_error_estimate_is_rounding_residue` for the proof
#: and :func:`_rounding_residue_bound` for the derivation. Only the term-vol
#: cases are listed: under ``BlackConstantVol`` the per-path value comes out
#: bit-identical in C++ and in this port, so the residue is identical too and
#: those cases still assert at TIGHT.
_VS_ERROR_IS_ROUNDING_RESIDUE = frozenset(
    {
        "vs_termvol_long_steps52_pr",
        "vs_termvol_long_steps13_pr",
        "vs_termvol_short_steps52_pr",
        "vs_termvol_long_stepsperyear20_pr",
    }
)

#: 2**-53 — the unit roundoff of IEEE-754 binary64.
_MACHINE_EPS = 2.0**-53


def _rounding_residue_bound(inputs: dict[str, Any], variance: float) -> float:
    """Largest ``errorEstimate`` explicable purely by rounding, for this case.

    ``MCVarianceSwapEngine`` reports ``multiplier * sampleAccumulator()
    .errorEstimate()`` with ``multiplier = +-discount(maturity) * notional``.
    The probe proves (``vs_sample_spread_termvol_steps52``: ``spread == 0``,
    ``allSamplesIdentical == true``) that every per-path sample is the SAME
    double, so the sample standard deviation is not a statistic at all — it is
    ``|x - mean|``, the residue left by summing ``N`` copies of ``x`` inside
    ``GeneralStatistics.mean()``.

    A left-to-right sum of ``N`` identical doubles carries an absolute error of
    at most ``(N-1) * eps * |x|``, so::

        |errorEstimate| <= (N - 1) * eps * |variance| / sqrt(N) * |multiplier|

    Nothing about that quantity is reproducible across implementations: C++ and
    this port sum the same value in a different order. What IS reproducible —
    the fair variance and the NPV, both means rather than differences of equals
    — is asserted at TIGHT alongside.
    """
    samples = int(inputs["samples"])
    notional = float(inputs["notional"])
    discount = _bsm().risk_free_rate().discount(EXERCISE)
    multiplier = discount * notional
    return (samples - 1) * _MACHINE_EPS * abs(variance) / math.sqrt(samples) * multiplier


@pytest.mark.parametrize(
    "case_name",
    [
        "vs_long_steps52_pr",
        "vs_short_steps52_pr",
        "vs_long_steps52_pr_anti",
        "vs_long_steps52_pr_bb",
        "vs_long_highstrike_steps52_pr",
        "vs_long_stepsperyear12_pr",
        "vs_stepsperyear_collapses",
        "vs_long_steps52_ld",
        "vs_termvol_long_steps52_pr",
        "vs_termvol_long_steps13_pr",
        "vs_termvol_short_steps52_pr",
        "vs_termvol_long_stepsperyear20_pr",
    ],
)
def test_mc_variance_swap_engine_matches_cpp(
    cpp_fwd: dict[str, Any], case_name: str
) -> None:
    """``MCVarianceSwapEngine`` reproduces the fair variance, NPV and error.

    The short cases matter on their own: C++ multiplies the *error estimate* by
    the position multiplier too, so a short reports a NEGATIVE error estimate. A
    port that takes an absolute value passes every long case and fails here.
    """
    case = cpp_fwd[case_name]
    swap = _price_variance_swap(case["inputs"])
    expected = case["expected"]
    tolerance.tight(swap.variance(), expected["variance"])
    tolerance.tight(swap.npv(), expected["npv"])

    if not expected["hasErrorEstimate"]:
        assert expected["errorEstimate"] is None
        with pytest.raises(LibraryException, match="error estimate not provided"):
            swap.error_estimate()
        return

    if case_name in _VS_ERROR_IS_ROUNDING_RESIDUE:
        bound = _rounding_residue_bound(case["inputs"], expected["variance"])
        # Both numbers must be residue. This is a STRONGER claim than a loose
        # relative tolerance would be: it says the estimator is zero up to
        # rounding on both sides, which is exactly what the probe proves.
        assert abs(expected["errorEstimate"]) <= bound, (
            f"C++ errorEstimate {expected['errorEstimate']!r} exceeds the "
            f"rounding-residue bound {bound!r}; it is a real statistic after "
            f"all and must be pinned at TIGHT"
        )
        assert abs(swap.error_estimate()) <= bound
        # The sign still carries the position multiplier and IS meaningful.
        assert (swap.error_estimate() < 0.0) == (expected["errorEstimate"] < 0.0)
        return

    tolerance.tight(swap.error_estimate(), expected["errorEstimate"])


@pytest.mark.parametrize(
    "case_name",
    [
        "vs_sample_spread_constant_steps52",
        "vs_sample_spread_termvol_steps52",
        "vs_sample_spread_termvol_steps13",
    ],
)
def test_variance_swap_error_estimate_is_rounding_residue(
    cpp: dict[str, Any], case_name: str
) -> None:
    """Every per-path realised variance is the SAME double, in C++ and here.

    ``VariancePathPricer`` integrates the squared LOCAL volatility, and for both
    ``BlackConstantVol`` and ``BlackVarianceCurve`` that is a function of time
    alone (see :func:`test_termvol_local_vol_is_level_independent`). So the MC
    "estimator" has zero spread and its reported standard error is the rounding
    residue of ``mean()``. This test reproduces the probe's own loop and asserts
    the same degeneracy, which is the property a port must match — the residue
    itself is not.
    """
    case = cpp[case_name]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["allSamplesIdentical"] is True
    assert expected["spread"] == 0.0

    process = _bsm(vol=_term_vol() if inputs["volKind"] == "termStructure" else None)
    t = process.time(EXERCISE)
    grid = TimeGrid.regular(t, int(inputs["steps"]))
    generator = PseudoRandom.make_sequence_generator(
        process.factors() * (len(grid) - 1), int(inputs["seed"])
    )
    path_generator = PathGenerator.with_time_grid(
        process, grid, generator, brownian_bridge=False
    )
    pricer = VariancePathPricer(process)
    values = [pricer(path_generator.next().value) for _ in range(int(inputs["samples"]))]

    assert min(values) == max(values), "the port's per-path spread must be zero too"
    tolerance.tight(min(values), expected["min"])
    tolerance.tight(max(values), expected["max"])
    tolerance.tight(sum(values) / len(values), expected["mean"])


def test_termvol_local_vol_is_level_independent(cpp: dict[str, Any]) -> None:
    """``BlackVarianceCurve`` -> ``LocalVolCurve``, which ignores the level.

    This is the root cause of the degeneracy above, and it is the thing a port
    can plausibly get wrong: routing a ``BlackVarianceCurve`` through the generic
    ``LocalVolSurface`` (with its finite-difference strike derivatives) instead
    of ``LocalVolCurve`` would make the local vol level-dependent at the 1e-8
    level, and the variance-swap price would drift.
    """
    case = cpp["termvol_localvol_is_level_independent"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["strikeIndependent"] is True

    surface = _bsm(vol=_term_vol()).local_volatility()
    actual = [
        surface.local_vol_at_time(float(u), float(level), extrapolate=True)
        for u in inputs["times"]
        for level in inputs["levels"]
    ]
    for got, want in zip(actual, expected["localVols"], strict=True):
        tolerance.tight(got, want)
    # Bit-exact level independence, as in C++.
    for block in range(0, len(actual), len(inputs["levels"])):
        row = actual[block : block + len(inputs["levels"])]
        assert len(set(row)) == 1


def test_variance_swap_short_error_estimate_is_negative(
    cpp_fwd: dict[str, Any],
) -> None:
    """Spelled out, because it looks like a bug and is not."""
    assert cpp_fwd["vs_short_steps52_pr"]["expected"]["errorEstimate"] < 0.0
    assert _price_variance_swap(cpp_fwd["vs_short_steps52_pr"]["inputs"]).error_estimate() < 0.0


@pytest.mark.parametrize(
    ("case_name", "chain"),
    [
        ("vs_make_no_steps_throws", "no_steps"),
        ("vs_make_both_steps_throws", "both_steps"),
    ],
)
def test_variance_swap_builder_steps_guards(
    cpp_fwd: dict[str, Any], case_name: str, chain: str
) -> None:
    case = cpp_fwd[case_name]
    assert case["expected"]["throws"] is True
    builder = MakeMCVarianceSwapEngine(_bsm()).with_samples(255).with_seed(42)
    if chain == "both_steps":
        builder.with_steps(8).with_steps_per_year(4)
        message = "number of steps overspecified"
    else:
        message = "number of steps not given"
    with pytest.raises(LibraryException, match=message):
        builder.engine()


def test_variance_swap_builder_has_no_control_variate(cpp_fwd: dict[str, Any]) -> None:
    """C++ exposes no ``withControlVariate`` here; neither may the port."""
    expected = cpp_fwd["vs_make_has_no_control_variate"]["expected"]
    assert (
        hasattr(MakeMCVarianceSwapEngine, "with_control_variate")
        is expected["hasWithControlVariate"]
    )


def test_forward_bs_builder_has_no_control_variate(cpp_fwd: dict[str, Any]) -> None:
    """Only the Heston forward builder exposes the control variate."""
    expected = cpp_fwd["fwdbs_make_has_no_control_variate"]["expected"]
    assert (
        hasattr(MakeMCForwardEuropeanBSEngine, "with_control_variate")
        is expected["hasWithControlVariate"]
    )


@pytest.mark.parametrize(
    ("case_name", "chain"),
    [
        ("fwdbs_make_no_steps_throws", "no_steps"),
        ("fwdbs_make_both_steps_throws", "both_steps"),
        ("fwdbs_make_samples_then_tolerance_throws", "samples_then_tolerance"),
        ("fwdbs_make_tolerance_then_samples_throws", "tolerance_then_samples"),
        ("fwdbs_make_ld_tolerance_throws", "ld_tolerance"),
    ],
)
def test_forward_bs_builder_guards(
    cpp_fwd: dict[str, Any], case_name: str, chain: str
) -> None:
    """``MakeMCForwardEuropeanBSEngine`` validation."""
    assert cpp_fwd[case_name]["expected"]["throws"] is True
    process = _bsm()
    if chain == "no_steps":
        with pytest.raises(LibraryException, match="number of steps not given"):
            MakeMCForwardEuropeanBSEngine(process).with_samples(255).with_seed(
                42
            ).engine()
    elif chain == "both_steps":
        with pytest.raises(LibraryException, match="number of steps overspecified"):
            MakeMCForwardEuropeanBSEngine(process).with_steps(8).with_steps_per_year(
                4
            ).with_samples(255).with_seed(42).engine()
    elif chain == "samples_then_tolerance":
        with pytest.raises(LibraryException, match="number of samples already set"):
            MakeMCForwardEuropeanBSEngine(process).with_steps(8).with_samples(
                255
            ).with_absolute_tolerance(0.01)
    elif chain == "tolerance_then_samples":
        with pytest.raises(LibraryException, match="tolerance already set"):
            MakeMCForwardEuropeanBSEngine(process).with_steps(8).with_absolute_tolerance(
                0.01
            ).with_samples(255)
    else:
        with pytest.raises(LibraryException, match="does not allow an error estimate"):
            MakeMCForwardEuropeanBSEngine(
                process, LowDiscrepancy
            ).with_steps(8).with_absolute_tolerance(0.01)


# ===========================================================================
# 8. MCForwardVanillaEngine constructor guards + forward engine guards
#    (reference: v143/pe/mcfwdlb)
# ===========================================================================


@pytest.mark.parametrize(
    ("case_name", "steps", "per_year"),
    [
        ("fwdvanilla_engine_zero_steps_throws", 0, None),
        ("fwdvanilla_engine_zero_steps_per_year_throws", None, 0),
        ("fwdvanilla_engine_no_steps_throws", None, None),
        ("fwdvanilla_engine_both_steps_throws", 8, 4),
    ],
)
def test_forward_vanilla_engine_steps_guards(
    cpp: dict[str, Any], case_name: str, steps: int | None, per_year: int | None
) -> None:
    """The ``MCForwardVanillaEngine`` constructor guards, via the BS subclass."""
    _assert_throws(
        cpp[case_name],
        lambda: MCForwardEuropeanBSEngine(
            _bsm(),
            time_steps=steps,
            time_steps_per_year=per_year,
            required_samples=255,
            seed=42,
        ),
    )


def test_forward_bs_engine_rejects_non_plain_payoff(cpp: dict[str, Any]) -> None:
    """``pathPricer()`` requires a ``PlainVanillaPayoff``."""
    option = ForwardVanillaOption(
        1.0,
        RESET,
        CashOrNothingPayoff(OptionType.Call, 0.0, 10.0),
        EuropeanExercise(EXERCISE),
    )
    option.set_pricing_engine(
        MakeMCForwardEuropeanBSEngine(_bsm())
        .with_steps(8)
        .with_samples(255)
        .with_seed(42)
        .engine()
    )
    _assert_throws(cpp["fwdbs_engine_non_plain_payoff_throws"], option.npv)


def test_forward_bs_engine_rejects_american_exercise(cpp: dict[str, Any]) -> None:
    """``pathPricer()`` requires a ``EuropeanExercise``."""
    option = ForwardVanillaOption(
        1.0,
        RESET,
        PlainVanillaPayoff(OptionType.Call, 0.0),
        AmericanExercise(TODAY, EXERCISE),
    )
    option.set_pricing_engine(
        MakeMCForwardEuropeanBSEngine(_bsm())
        .with_steps(8)
        .with_samples(255)
        .with_seed(42)
        .engine()
    )
    _assert_throws(cpp["fwdbs_engine_american_exercise_throws"], option.npv)


@pytest.mark.parametrize(
    ("case_name", "chain"),
    [
        ("fwdheston_make_no_steps_throws", "no_steps"),
        ("fwdheston_make_both_steps_throws", "both_steps"),
        ("fwdheston_make_samples_then_tolerance_throws", "samples_then_tolerance"),
        ("fwdheston_make_tolerance_then_samples_throws", "tolerance_then_samples"),
        ("fwdheston_make_ld_tolerance_throws", "ld_tolerance"),
    ],
)
def test_forward_heston_builder_guards(
    cpp: dict[str, Any], case_name: str, chain: str
) -> None:
    """``MakeMCForwardEuropeanHestonEngine`` validation, message-for-message."""
    process = _heston()
    actions = {
        "no_steps": lambda: MakeMCForwardEuropeanHestonEngine(process)
        .with_samples(255)
        .with_seed(42)
        .engine(),
        "both_steps": lambda: MakeMCForwardEuropeanHestonEngine(process)
        .with_steps(8)
        .with_steps_per_year(4)
        .with_samples(255)
        .with_seed(42)
        .engine(),
        "samples_then_tolerance": lambda: MakeMCForwardEuropeanHestonEngine(process)
        .with_steps(8)
        .with_samples(255)
        .with_absolute_tolerance(0.01),
        "tolerance_then_samples": lambda: MakeMCForwardEuropeanHestonEngine(process)
        .with_steps(8)
        .with_absolute_tolerance(0.01)
        .with_samples(255),
        "ld_tolerance": lambda: MakeMCForwardEuropeanHestonEngine(
            process, LowDiscrepancy
        )
        .with_steps(8)
        .with_absolute_tolerance(0.01),
    }
    _assert_throws(cpp[case_name], actions[chain])


def test_forward_heston_builder_shape(cpp: dict[str, Any]) -> None:
    """This builder defers BOTH steps checks to conversion time, unlike the
    vanilla-wave ``MakeMCEuropeanHestonEngine`` which rejects over-specification
    inside ``with_steps_per_year``."""
    expected = cpp["fwdheston_make_shape"]["expected"]
    assert expected["withStepsPerYearAfterStepsThrowsEarly"] is False
    # No exception here — the conflict is only reported by ``engine()``.
    MakeMCForwardEuropeanHestonEngine(_heston()).with_steps(8).with_steps_per_year(4)
    assert (
        hasattr(MakeMCForwardEuropeanHestonEngine, "with_control_variate")
        is expected["hasWithControlVariate"]
    )
    assert (
        hasattr(MakeMCForwardEuropeanHestonEngine, "with_brownian_bridge")
        is expected["hasWithBrownianBridge"]
    )


# ===========================================================================
# 9. The two forward path pricers, directly (reference: v143/pe/mcfwdlb)
# ===========================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "fwdbs_pathpricer_call_m1_reset0",
        "fwdbs_pathpricer_call_m1_reset3",
        "fwdbs_pathpricer_put_m1_reset3",
        "fwdbs_pathpricer_call_m085_reset3",
        "fwdbs_pathpricer_call_m125_reset3",
        "fwdbs_pathpricer_call_m0_reset3",
    ],
)
def test_forward_bs_path_pricer_matches_cpp(
    cpp: dict[str, Any], case_name: str
) -> None:
    """strike = ``path[resetIndex] * moneyness``, payoff at ``path.back()``.

    ``..._m0_reset3`` is the case the instrument cannot reach: constructed
    directly the pricer accepts ``moneyness == 0`` and a call then pays the whole
    terminal spot.
    """
    case = cpp[case_name]
    inputs, expected = case["inputs"], case["expected"]
    pricer = ForwardEuropeanBSPathPricer(
        _option_type(inputs["optionType"]),
        float(inputs["moneyness"]),
        int(inputs["resetIndex"]),
        float(inputs["discount"]),
    )
    tolerance.tight(pricer(_rebuild_path(list(inputs["path"]))), expected["value"])


@pytest.mark.parametrize(
    "case_name",
    [
        "fwdheston_pathpricer_call_m1_reset0",
        "fwdheston_pathpricer_call_m1_reset3",
        "fwdheston_pathpricer_put_m1_reset3",
        "fwdheston_pathpricer_call_m085_reset3",
    ],
)
def test_forward_heston_path_pricer_matches_cpp(
    cpp: dict[str, Any], case_name: str
) -> None:
    """Reads ``multiPath[0]`` only; the variance leg is never touched.

    The probe supplies a variance leg with completely different levels, so a port
    that reads the wrong leg produces a wildly different number.
    """
    case = cpp[case_name]
    inputs, expected = case["inputs"], case["expected"]
    multi_path = MultiPath(
        [
            _rebuild_path(list(inputs["assetPath"])),
            _rebuild_path(list(inputs["variancePath"])),
        ]
    )
    pricer = ForwardEuropeanHestonPathPricer(
        _option_type(inputs["optionType"]),
        float(inputs["moneyness"]),
        int(inputs["resetIndex"]),
        float(inputs["discount"]),
    )
    tolerance.tight(pricer(multi_path), expected["value"])


def test_forward_bs_path_pricer_rejects_negative_moneyness(cpp: dict[str, Any]) -> None:
    """The pricer's own guard, live only when it is constructed directly."""
    _assert_throws(
        cpp["fwdbs_pathpricer_negative_moneyness_throws"],
        lambda: ForwardEuropeanBSPathPricer(OptionType.Call, -0.5, 3, 0.9375),
    )


def test_forward_heston_path_pricer_rejects_negative_moneyness(
    cpp: dict[str, Any],
) -> None:
    _assert_throws(
        cpp["fwdheston_pathpricer_negative_moneyness_throws"],
        lambda: ForwardEuropeanHestonPathPricer(OptionType.Call, -0.5, 3, 0.9375),
    )
