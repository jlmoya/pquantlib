"""Cross-validate the v1.43 Monte Carlo *vanilla* engine cluster against C++.

Reference: ``migration-harness/references/v143/pe/mcvanilla`` (probe at
``migration-harness/cpp/probes/v143_pe_mcvanilla/probe.cpp``).

Covers :class:`MCEuropeanEngine` / :class:`MakeMCEuropeanEngine`,
:class:`MCAmericanEngine` / :class:`MakeMCAmericanEngine`,
:class:`MCDigitalEngine` / :class:`DigitalPathPricer` /
:class:`MakeMCDigitalEngine`, :class:`MCEuropeanHestonEngine` /
:class:`EuropeanHestonPathPricer` / :class:`MakeMCEuropeanHestonEngine`,
:class:`MCEuropeanGJRGARCHEngine` / :class:`EuropeanGJRGARCHPathPricer` /
:class:`MakeMCEuropeanGJRGARCHEngine`, and :class:`MCHestonHullWhiteEngine` /
:class:`HestonHullWhitePathPricer` / :class:`MakeMCHestonHullWhiteEngine`
(plus the :class:`HybridHestonHullWhiteProcess` they need in order to price).

Why the assertions are exact
----------------------------
Every engine here is deterministic once the seed is fixed:
``PseudoRandom`` is ``InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
InverseCumulativeNormal>`` and ``LowDiscrepancy`` is
``InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>``; neither consults
the clock for a nonzero seed. So a correct port reproduces the C++ NPV *and*
the C++ error estimate, and the tier is TIGHT (1e-14 abs / 1e-12 rel) rather
than a confidence band. A band would pass with the wrong RNG, the wrong path
construction, or the wrong antithetic pairing — which is the entire failure
mode this file exists to catch. Measured worst case across the sweep is
3.6e-13 relative, on the Heston/Hull-White cases where the payoff is divided by
a stochastic numeraire.

Each case carries its whole market description in ``inputs``, so the tests
rebuild the setup rather than restating constants.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.randomnumbers.rng_traits import LowDiscrepancy, PseudoRandom
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer import (
    LongstaffSchwartzPathPricer,
)
from pquantlib.methods.montecarlo.lsm_basis_system import PolynomialType
from pquantlib.methods.montecarlo.monte_carlo_model import MonteCarloModel
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import CashOrNothingPayoff, OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.mc_american_engine import (
    AmericanPathPricer,
    MakeMCAmericanEngine,
)
from pquantlib.pricingengines.vanilla.mc_digital_engine import MakeMCDigitalEngine
from pquantlib.pricingengines.vanilla.mc_european_engine import (
    EuropeanPathPricer,
    MakeMCEuropeanEngine,
    MCEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.mc_european_gjr_garch_engine import (
    MakeMCEuropeanGJRGARCHEngine,
)
from pquantlib.pricingengines.vanilla.mc_european_heston_engine import (
    MakeMCEuropeanHestonEngine,
)
from pquantlib.pricingengines.vanilla.mc_heston_hull_white_engine import (
    MakeMCHestonHullWhiteEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.gjr_garch_process import (
    Discretization as GjrDiscretization,
)
from pquantlib.processes.gjr_garch_process import GJRGARCHProcess
from pquantlib.processes.heston_process import (
    Discretization as HestonDiscretization,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.processes.hybrid_heston_hull_white_process import (
    Discretization as HhwDiscretization,
)
from pquantlib.processes.hybrid_heston_hull_white_process import (
    HybridHestonHullWhiteProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.interpolated_zero_curve import (
    InterpolatedZeroCurve,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# probe.cpp — ``const Date kToday(15, May, 2025);`` and
# ``Settings::instance().evaluationDate() = kToday;`` in ``main()``.
TODAY = Date.from_ymd(15, Month.May, 2025)

#: probe.cpp — the zero curve used by the ``*_zerocurve_*`` cases.
_ZERO_OFFSETS = (0, 90, 180, 365, 730)
_ZERO_RATES = (0.02, 0.028, 0.035, 0.045, 0.05)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/mcvanilla")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


# ---------------------------------------------------------------------------
# Market reconstruction
# ---------------------------------------------------------------------------

_DC = Actual365Fixed()
_CAL = NullCalendar()


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(
        reference_date=TODAY, forward_rate=rate, day_counter=_DC
    )


def _zero_curve() -> InterpolatedZeroCurve:
    return InterpolatedZeroCurve(
        [TODAY + d for d in _ZERO_OFFSETS], list(_ZERO_RATES), _DC
    )


def _bsm(inputs: dict[str, Any]) -> GeneralizedBlackScholesProcess:
    """Rebuild the probe's ``BlackScholesMertonProcess`` from ``inputs``."""
    risk_free = _zero_curve() if inputs["curve"] == "zerocurve" else _flat(inputs["r"])
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(inputs["s0"]),
        dividend_ts=_flat(inputs["q"]),
        risk_free_ts=risk_free,
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY,
            calendar=_CAL,
            day_counter=_DC,
            volatility=inputs["vol"],
        ),
    )


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


def _traits(name: str) -> type[PseudoRandom] | type[LowDiscrepancy]:
    return PseudoRandom if name == "pseudo" else LowDiscrepancy


def _check_results(option: VanillaOption, expected: dict[str, Any]) -> None:
    """Assert NPV and the presence/absence of the error estimate."""
    tolerance.tight(option.npv(), expected["npv"])
    if expected["has_error_estimate"]:
        tolerance.tight(option.error_estimate(), expected["error_estimate"])
    else:
        # C++ ``if constexpr (RNG::allowsErrorEstimate)`` leaves
        # ``results_.errorEstimate`` at Null for LowDiscrepancy, and
        # ``Instrument::errorEstimate()`` throws. Pin the absence.
        with pytest.raises(LibraryException):
            option.error_estimate()


def _names(cpp: dict[str, Any], prefix: str, *, priced: bool = True) -> list[str]:
    return sorted(
        k
        for k in cpp
        if k.startswith(prefix) and ("npv" in cpp[k]["expected"]) is priced
    )


# ---------------------------------------------------------------------------
# 1. MCEuropeanEngine / EuropeanPathPricer / MakeMCEuropeanEngine
# ---------------------------------------------------------------------------


def _build_european(inputs: dict[str, Any]) -> VanillaOption:
    process = _bsm(inputs)
    builder = MakeMCEuropeanEngine(process, _traits(inputs["rng"]))
    if inputs["steps"] > 0:
        builder.with_steps(inputs["steps"])
    else:
        builder.with_steps_per_year(inputs["steps_per_year"])
    builder.with_brownian_bridge(inputs["brownian_bridge"]).with_antithetic_variate(
        inputs["antithetic"]
    ).with_seed(inputs["seed"])
    if inputs["samples"] > 0:
        builder.with_samples(inputs["samples"])
    else:
        builder.with_absolute_tolerance(inputs["tolerance"])
    if inputs["max_samples"] > 0:
        builder.with_max_samples(inputs["max_samples"])

    option = VanillaOption(
        PlainVanillaPayoff(_option_type(inputs["option_type"]), inputs["strike"]),
        EuropeanExercise(TODAY + inputs["expiry_days"]),
    )
    option.set_pricing_engine(builder.engine())
    return option


def test_european_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCEuropeanEngine case, exactly."""
    names = _names(cpp, "european_")
    assert names, "no European cases in the reference"
    for name in names:
        case = cpp[name]
        _check_results(_build_european(case["inputs"]), case["expected"])


def test_european_seed_actually_reaches_the_generator(cpp: dict[str, Any]) -> None:
    """Two seeds, one otherwise identical setup, two different answers.

    A port that swallows the seed (or substitutes its own) passes the
    single-seed test and fails this one.
    """
    a = cpp["european_ps_call_atm"]["expected"]["npv"]
    b = cpp["european_ps_call_atm_seed7"]["expected"]["npv"]
    assert a != b
    tolerance.tight(_build_european(cpp["european_ps_call_atm"]["inputs"]).npv(), a)
    tolerance.tight(
        _build_european(cpp["european_ps_call_atm_seed7"]["inputs"]).npv(), b
    )


def test_european_builder_validation(cpp: dict[str, Any]) -> None:
    """``MakeMCEuropeanEngine``'s guards, each pinned by the probe."""
    process = _bsm(cpp["european_ps_call_atm"]["inputs"])

    def expect(name: str) -> bool:
        return bool(cpp[name]["expected"]["throws"])

    assert expect("european_make_no_steps")
    with pytest.raises(LibraryException, match="number of steps not given"):
        MakeMCEuropeanEngine(process).with_samples(1023).engine()

    assert expect("european_make_both_steps")
    with pytest.raises(LibraryException, match="number of steps overspecified"):
        MakeMCEuropeanEngine(process).with_steps(4).with_steps_per_year(
            12
        ).with_samples(1023).engine()

    assert expect("european_make_samples_after_tolerance")
    with pytest.raises(LibraryException, match="tolerance already set"):
        MakeMCEuropeanEngine(process).with_absolute_tolerance(0.02).with_samples(1023)

    assert expect("european_make_tolerance_after_samples")
    with pytest.raises(LibraryException, match="number of samples already set"):
        MakeMCEuropeanEngine(process).with_samples(1023).with_absolute_tolerance(0.02)

    assert expect("european_make_tolerance_lowdiscrepancy")
    with pytest.raises(LibraryException, match="does not allow an error estimate"):
        MakeMCEuropeanEngine(process, LowDiscrepancy).with_absolute_tolerance(0.02)

    # Positive control: the probe pins this one as NOT throwing.
    assert not expect("european_make_steps_only_ok")
    MakeMCEuropeanEngine(process).with_steps(4).with_samples(1023).engine()


def test_european_engine_guards(cpp: dict[str, Any]) -> None:
    """The ``MCVanillaEngine`` / ``McSimulation`` / path-pricer guards."""
    inputs = cpp["european_ps_call_atm"]["inputs"]
    process = _bsm(inputs)

    assert cpp["european_engine_zero_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="timeSteps must be positive"):
        MCEuropeanEngine(process, time_steps=0, required_samples=1023, seed=42)

    assert cpp["european_engine_zero_steps_per_year"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="timeStepsPerYear must be positive"):
        MCEuropeanEngine(
            process, time_steps_per_year=0, required_samples=1023, seed=42
        )

    assert cpp["european_engine_no_samples_no_tolerance"]["expected"]["throws"]
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(TODAY + 365),
    )
    option.set_pricing_engine(MCEuropeanEngine(process, time_steps=1, seed=42))
    with pytest.raises(LibraryException, match="neither tolerance nor number"):
        option.npv()

    assert cpp["european_pathpricer_negative_strike"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="strike less than zero"):
        EuropeanPathPricer(OptionType.Call, -1.0, 0.95)

    assert cpp["european_engine_non_plain_payoff"]["expected"]["throws"]
    binary = VanillaOption(
        CashOrNothingPayoff(OptionType.Call, 100.0, 10.0),
        EuropeanExercise(TODAY + 365),
    )
    binary.set_pricing_engine(
        MakeMCEuropeanEngine(process)
        .with_steps(1)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="non-plain payoff given"):
        binary.npv()


def test_european_tolerance_driven_termination(cpp: dict[str, Any]) -> None:
    """The ``McSimulation::value`` growth rule, pinned by its terminal answer.

    The tolerance loop's batch schedule
    ``nextBatch = max(Size(n*(err/tol)^2*0.8 - n), 1023)`` makes the terminal
    sample count a deterministic function of the seed, so reproducing the NPV
    *and* the error estimate is only possible with the same schedule.
    """
    for name in _names(cpp, "european_tol_"):
        case = cpp[name]
        _check_results(_build_european(case["inputs"]), case["expected"])


def test_european_tolerance_max_samples_exceeded(cpp: dict[str, Any]) -> None:
    """``QL_REQUIRE(sampleNumber < maxSamples)`` inside the tolerance loop."""
    assert cpp["european_tol_maxsamples_exceeded"]["expected"]["throws"]
    inputs = cpp["european_tol_maxsamples_exceeded"]["inputs"]
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        EuropeanExercise(TODAY + inputs["expiry_days"]),
    )
    option.set_pricing_engine(
        MakeMCEuropeanEngine(_bsm(inputs))
        .with_steps(inputs["steps"])
        .with_absolute_tolerance(inputs["tolerance"])
        .with_max_samples(inputs["max_samples"])
        .with_seed(inputs["seed"])
        .engine()
    )
    with pytest.raises(LibraryException, match="max number of samples"):
        option.npv()


# ---------------------------------------------------------------------------
# 2. MCAmericanEngine / AmericanPathPricer / MakeMCAmericanEngine
# ---------------------------------------------------------------------------


def _build_american(inputs: dict[str, Any]) -> tuple[VanillaOption, Any]:
    builder = MakeMCAmericanEngine(_bsm(inputs))
    builder.with_steps(inputs["steps"]).with_samples(inputs["samples"]).with_seed(
        inputs["seed"]
    )
    if "control_variate" in inputs:
        builder.with_control_variate(inputs["control_variate"])
        builder.with_antithetic_variate(inputs["antithetic"])
        builder.with_polynomial_order(inputs["polynomial_order"])
        builder.with_basis_system(PolynomialType[inputs["polynomial_type"]])
        builder.with_calibration_samples(inputs["calibration_samples"])
        if inputs["seed_calibration"] >= 0:
            builder.with_seed_calibration(inputs["seed_calibration"])
    engine = builder.engine()
    option = VanillaOption(
        PlainVanillaPayoff(_option_type(inputs["option_type"]), inputs["strike"]),
        AmericanExercise(TODAY, TODAY + inputs["expiry_days"]),
    )
    option.set_pricing_engine(engine)
    return option, engine


def test_american_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCAmericanEngine case, including exercise probability."""
    names = [n for n in _names(cpp, "american_") if n != "american_lsm_diagnostics"]
    assert names
    for name in names:
        case = cpp[name]
        option, engine = _build_american(case["inputs"])
        _check_results(option, case["expected"])
        tolerance.tight(
            engine.exercise_probability(), case["expected"]["exercise_probability"]
        )


def test_american_make_defaults_match_the_explicit_twin(cpp: dict[str, Any]) -> None:
    """The builder defaults (order 2, Monomial, 2048 calibration samples).

    ``american_make_defaults`` leaves every knob alone; ``american_ps_put_atm``
    sets them all explicitly to the same values. Equal answers prove the
    defaults, and a mismatch localises to whichever default drifted.
    """
    assert (
        cpp["american_make_defaults"]["expected"]["npv"]
        == cpp["american_ps_put_atm"]["expected"]["npv"]
    )
    option, _ = _build_american(cpp["american_make_defaults"]["inputs"])
    tolerance.tight(option.npv(), cpp["american_make_defaults"]["expected"]["npv"])


def test_american_control_variate_clamps_at_zero(cpp: dict[str, Any]) -> None:
    """``MCAmericanEngine::calculate`` clamps only when the CV is on.

    The probe case is a deep-OTM put whose raw CV-adjusted estimator is
    negative, so C++ reports exactly 0.0.
    """
    expected = cpp["american_cv_deep_otm_clamped"]["expected"]
    assert expected["npv"] == 0.0
    option, _ = _build_american(cpp["american_cv_deep_otm_clamped"]["inputs"])
    tolerance.exact(option.npv(), 0.0)


def test_american_builder_validation(cpp: dict[str, Any]) -> None:
    process = _bsm(cpp["american_ps_put_atm"]["inputs"])

    assert cpp["american_make_no_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps not given"):
        MakeMCAmericanEngine(process).with_samples(1023).engine()

    assert cpp["american_make_both_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps overspecified"):
        MakeMCAmericanEngine(process).with_steps(4).with_steps_per_year(
            12
        ).with_samples(1023).engine()

    assert cpp["american_make_samples_after_tolerance"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="tolerance already set"):
        MakeMCAmericanEngine(process).with_absolute_tolerance(0.02).with_samples(1023)

    assert cpp["american_make_tolerance_after_samples"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of samples already set"):
        MakeMCAmericanEngine(process).with_samples(1023).with_absolute_tolerance(0.02)

    assert cpp["american_make_tolerance_lowdiscrepancy"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="does not allow an error estimate"):
        MakeMCAmericanEngine(process, LowDiscrepancy).with_absolute_tolerance(0.02)


def test_american_engine_guards(cpp: dict[str, Any]) -> None:
    process = _bsm(cpp["american_ps_put_atm"]["inputs"])

    assert cpp["american_pathpricer_legendre_rejected"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="insufficient polynomial type"):
        AmericanPathPricer(
            PlainVanillaPayoff(OptionType.Put, 100.0), 2, PolynomialType.Legendre
        )

    assert cpp["american_engine_european_exercise"]["expected"]["throws"]
    european = VanillaOption(
        PlainVanillaPayoff(OptionType.Put, 100.0), EuropeanExercise(TODAY + 365)
    )
    european.set_pricing_engine(
        MakeMCAmericanEngine(process)
        .with_steps(10)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="wrong exercise given"):
        european.npv()

    assert cpp["american_engine_payoff_at_expiry"]["expected"]["throws"]
    at_expiry = VanillaOption(
        PlainVanillaPayoff(OptionType.Put, 100.0),
        AmericanExercise(TODAY, TODAY + 365, True),
    )
    at_expiry.set_pricing_engine(
        MakeMCAmericanEngine(process)
        .with_steps(10)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="payoff at expiry not handled"):
        at_expiry.npv()


def test_american_lsm_internals(cpp: dict[str, Any]) -> None:
    """Grid, discount factors and every regression coefficient of the LSM pass.

    This is the diagnostic layer: it localises an American failure to the time
    grid, the calibration stream, or the least-squares solve rather than only
    showing a wrong NPV. It is also what caught the calibration-seed defect —
    the coefficients matched to 1e-13 while the NPV was 1.3% out, which is only
    possible if the *stream* is right and something downstream is not.
    """
    case = cpp["american_lsm_diagnostics"]
    inputs, expected = case["inputs"], case["expected"]
    process = _bsm(inputs)
    grid_times = [process.time(TODAY + inputs["expiry_days"])]

    from pquantlib.time.time_grid import TimeGrid  # noqa: PLC0415

    grid = TimeGrid.with_mandatory_and_steps(grid_times, inputs["steps"])
    assert len(grid) == expected["grid_size"]
    for i in range(len(grid)):
        tolerance.tight(grid[i], expected[f"grid_t{i}"])

    early = AmericanPathPricer(
        PlainVanillaPayoff(_option_type(inputs["option_type"]), inputs["strike"]),
        inputs["polynomial_order"],
        PolynomialType[inputs["polynomial_type"]],
    )
    assert len(early.basis_system()) == expected["basis_size"]

    lsm = LongstaffSchwartzPathPricer[Path, float](
        grid, early, process.risk_free_rate()
    )
    generator = PseudoRandom.make_sequence_generator(
        process.factors() * (len(grid) - 1), inputs["seed_calibration"]
    )
    calibration = MonteCarloModel[Path](
        path_generator=PathGenerator.with_time_grid(
            process, grid, generator, brownian_bridge=False
        ),
        path_pricer=lsm,
        sample_accumulator=GeneralStatistics(),
        antithetic_variate=False,
    )
    calibration.add_samples(inputs["calibration_samples"])
    lsm.calibrate()

    for i, coefficients in enumerate(lsm.coefficients()):
        for level, value in enumerate(coefficients):
            tolerance.tight(float(value), expected[f"coeff{i}_{level}"])

    # Pricing pass on a fresh generator seeded with the *pricing* seed.
    pricing = PathGenerator.with_time_grid(
        process,
        grid,
        PseudoRandom.make_sequence_generator(
            process.factors() * (len(grid) - 1), inputs["seed"]
        ),
        brownian_bridge=False,
    )
    for k in range(5):
        path = pricing.next().value
        if k == 0:
            for j in range(1, path.length()):
                tolerance.tight(float(path.values[j]), expected[f"path0_{j}"])
        tolerance.tight(lsm(path), expected[f"price{k}"])
    tolerance.tight(
        lsm.exercise_probability(), expected["exercise_probability_first5"]
    )


# ---------------------------------------------------------------------------
# 3. MCDigitalEngine / DigitalPathPricer / MakeMCDigitalEngine
# ---------------------------------------------------------------------------


def _build_digital(inputs: dict[str, Any]) -> VanillaOption:
    builder = MakeMCDigitalEngine(_bsm(inputs), _traits(inputs["rng"]))
    if inputs["steps"] > 0:
        builder.with_steps(inputs["steps"])
    else:
        builder.with_steps_per_year(inputs["steps_per_year"])
    builder.with_brownian_bridge(inputs["brownian_bridge"]).with_antithetic_variate(
        inputs["antithetic"]
    ).with_samples(inputs["samples"]).with_seed(inputs["seed"])
    option = VanillaOption(
        CashOrNothingPayoff(
            _option_type(inputs["option_type"]), inputs["strike"], inputs["cash_payoff"]
        ),
        AmericanExercise(
            TODAY, TODAY + inputs["expiry_days"], inputs["payoff_at_expiry"]
        ),
    )
    option.set_pricing_engine(builder.engine())
    return option


def test_digital_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCDigitalEngine case.

    The coarse-grid cases (2 and 4 steps over a year) are where the
    Beaglehole-Dybvig-Zhou bridge correction dominates rather than merely
    refines: a port that drops it, or that draws the hit uniforms from the
    engine's own generator instead of the hard-coded seed-76 one, misses these
    by a wide margin while still looking plausible on the 90-steps-per-year
    cases.
    """
    names = _names(cpp, "digital_")
    assert names
    for name in names:
        case = cpp[name]
        _check_results(_build_digital(case["inputs"]), case["expected"])


def test_digital_bridge_uniform_seed_is_pinned(cpp: dict[str, Any]) -> None:
    """The bridge generator's seed is a constant of the algorithm, not a knob."""
    from pquantlib.pricingengines.vanilla.mc_digital_engine import (  # noqa: PLC0415
        BRIDGE_UNIFORM_SEED,
    )

    assert cpp["digital_ps_call_coarse4"]["inputs"][
        "bridge_uniform_seed"
    ] == BRIDGE_UNIFORM_SEED


def test_digital_payoff_at_expiry_changes_the_discount(cpp: dict[str, Any]) -> None:
    """Same paths, later payment: strictly smaller value."""
    hit = cpp["digital_ps_call_coarse4"]["expected"]["npv"]
    at_expiry = cpp["digital_ps_call_coarse4_payoff_at_expiry"]["expected"]["npv"]
    assert at_expiry < hit
    tolerance.tight(
        _build_digital(
            cpp["digital_ps_call_coarse4_payoff_at_expiry"]["inputs"]
        ).npv(),
        at_expiry,
    )


def test_digital_builder_and_engine_validation(cpp: dict[str, Any]) -> None:
    process = _bsm(cpp["digital_ps_call_coarse4"]["inputs"])

    assert cpp["digital_make_no_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps not given"):
        MakeMCDigitalEngine(process).with_samples(1023).engine()

    assert cpp["digital_make_both_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps overspecified"):
        MakeMCDigitalEngine(process).with_steps(4).with_steps_per_year(
            12
        ).with_samples(1023).engine()

    assert cpp["digital_make_tolerance_lowdiscrepancy"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="does not allow an error estimate"):
        MakeMCDigitalEngine(process, LowDiscrepancy).with_absolute_tolerance(0.02)

    assert cpp["digital_engine_plain_payoff"]["expected"]["throws"]
    wrong_payoff = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0),
        AmericanExercise(TODAY, TODAY + 365),
    )
    wrong_payoff.set_pricing_engine(
        MakeMCDigitalEngine(process)
        .with_steps(4)
        .with_samples(1023)
        .with_seed(1)
        .engine()
    )
    with pytest.raises(LibraryException, match="wrong payoff given"):
        wrong_payoff.npv()

    assert cpp["digital_engine_european_exercise"]["expected"]["throws"]
    wrong_exercise = VanillaOption(
        CashOrNothingPayoff(OptionType.Call, 110.0, 15.0),
        EuropeanExercise(TODAY + 365),
    )
    wrong_exercise.set_pricing_engine(
        MakeMCDigitalEngine(process)
        .with_steps(4)
        .with_samples(1023)
        .with_seed(1)
        .engine()
    )
    with pytest.raises(LibraryException, match="wrong exercise given"):
        wrong_exercise.npv()


# ---------------------------------------------------------------------------
# 4. MCEuropeanHestonEngine / EuropeanHestonPathPricer / Make*
# ---------------------------------------------------------------------------


def _heston(inputs: dict[str, Any]) -> HestonProcess:
    return HestonProcess(
        risk_free_rate=_flat(inputs["r"]),
        dividend_yield=_flat(inputs["q"]),
        s0=SimpleQuote(inputs["s0"]),
        v0=inputs["v0"],
        kappa=inputs["kappa"],
        theta=inputs["theta"],
        sigma=inputs["sigma"],
        rho=inputs["rho"],
        discretization=HestonDiscretization[inputs["discretization"]],
    )


def test_heston_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCEuropeanHestonEngine case, over five discretizations.

    The default is ``QuadraticExponentialMartingale``, which is *not* a plain
    Euler step; an engine built on the inherited
    ``apply(expectation, stdDeviation*dw)`` cannot reproduce any of these.
    """
    names = _names(cpp, "heston_ps_")
    assert names
    for name in names:
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        builder = MakeMCEuropeanHestonEngine(_heston(inputs))
        if inputs["steps"] > 0:
            builder.with_steps(inputs["steps"])
        else:
            builder.with_steps_per_year(inputs["steps_per_year"])
        builder.with_antithetic_variate(inputs["antithetic"]).with_samples(
            inputs["samples"]
        ).with_seed(inputs["seed"])
        option = VanillaOption(
            PlainVanillaPayoff(_option_type(inputs["option_type"]), inputs["strike"]),
            EuropeanExercise(TODAY + inputs["expiry_days"]),
        )
        option.set_pricing_engine(builder.engine())
        _check_results(option, expected)


def test_heston_builder_rejects_steps_early(cpp: dict[str, Any]) -> None:
    """The C++ asymmetry: this builder guards inside the setters, not at conversion.

    ``MakeMCEuropeanEngine`` defers both steps guards to the conversion
    operator; ``MakeMCEuropeanHestonEngine`` rejects over-specification inside
    ``withSteps`` / ``withStepsPerYear`` and its conversion checks only the
    "not given" half. Both spellings are pinned so a port cannot normalise them
    into one.
    """
    process = _heston(cpp["heston_ps_call_atm"]["inputs"])

    assert cpp["heston_make_no_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps not given"):
        MakeMCEuropeanHestonEngine(process).with_samples(1023).engine()

    assert cpp["heston_make_steps_then_stepsperyear"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps already set"):
        MakeMCEuropeanHestonEngine(process).with_steps(4).with_steps_per_year(12)

    assert cpp["heston_make_stepsperyear_then_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps per year already set"):
        MakeMCEuropeanHestonEngine(process).with_steps_per_year(12).with_steps(4)

    assert cpp["heston_make_samples_after_tolerance"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="tolerance already set"):
        MakeMCEuropeanHestonEngine(process).with_absolute_tolerance(0.02).with_samples(
            1023
        )

    assert cpp["heston_make_tolerance_lowdiscrepancy"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="does not allow an error estimate"):
        MakeMCEuropeanHestonEngine(process, LowDiscrepancy).with_absolute_tolerance(
            0.02
        )

    assert cpp["heston_engine_non_plain_payoff"]["expected"]["throws"]
    binary = VanillaOption(
        CashOrNothingPayoff(OptionType.Call, 100.0, 10.0),
        EuropeanExercise(TODAY + 365),
    )
    binary.set_pricing_engine(
        MakeMCEuropeanHestonEngine(process)
        .with_steps(4)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="non-plain payoff given"):
        binary.npv()


# ---------------------------------------------------------------------------
# 5. MCEuropeanGJRGARCHEngine / EuropeanGJRGARCHPathPricer / Make*
# ---------------------------------------------------------------------------


def _gjr(inputs: dict[str, Any]) -> GJRGARCHProcess:
    return GJRGARCHProcess(
        risk_free_rate=_flat(inputs["r"]),
        dividend_yield=_flat(inputs["q"]),
        s0=SimpleQuote(inputs["s0"]),
        v0=inputs["v0"],
        omega=inputs["omega"],
        alpha=inputs["alpha"],
        beta=inputs["beta"],
        gamma=inputs["gamma"],
        lambda_=inputs["lambda"],
        days_per_year=inputs["days_per_year"],
        discretization=GjrDiscretization[inputs["discretization"]],
    )


def test_gjrgarch_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCEuropeanGJRGARCHEngine case, over three discretizations."""
    names = _names(cpp, "gjrgarch_ps_")
    assert names
    for name in names:
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        builder = MakeMCEuropeanGJRGARCHEngine(_gjr(inputs))
        if inputs["steps"] > 0:
            builder.with_steps(inputs["steps"])
        else:
            builder.with_steps_per_year(inputs["steps_per_year"])
        builder.with_antithetic_variate(inputs["antithetic"]).with_samples(
            inputs["samples"]
        ).with_seed(inputs["seed"])
        option = VanillaOption(
            PlainVanillaPayoff(_option_type(inputs["option_type"]), inputs["strike"]),
            EuropeanExercise(TODAY + inputs["expiry_days"]),
        )
        option.set_pricing_engine(builder.engine())
        _check_results(option, expected)


def test_gjrgarch_builder_validation(cpp: dict[str, Any]) -> None:
    process = _gjr(cpp["gjrgarch_ps_call_atm"]["inputs"])

    assert cpp["gjrgarch_make_no_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps not given"):
        MakeMCEuropeanGJRGARCHEngine(process).with_samples(1023).engine()

    assert cpp["gjrgarch_make_steps_then_stepsperyear"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps already set"):
        MakeMCEuropeanGJRGARCHEngine(process).with_steps(4).with_steps_per_year(12)

    assert cpp["gjrgarch_make_stepsperyear_then_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps per year already set"):
        MakeMCEuropeanGJRGARCHEngine(process).with_steps_per_year(12).with_steps(4)

    assert cpp["gjrgarch_make_tolerance_lowdiscrepancy"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="does not allow an error estimate"):
        MakeMCEuropeanGJRGARCHEngine(process, LowDiscrepancy).with_absolute_tolerance(
            0.02
        )

    assert cpp["gjrgarch_engine_non_plain_payoff"]["expected"]["throws"]
    binary = VanillaOption(
        CashOrNothingPayoff(OptionType.Call, 100.0, 10.0),
        EuropeanExercise(TODAY + 365),
    )
    binary.set_pricing_engine(
        MakeMCEuropeanGJRGARCHEngine(process)
        .with_steps(4)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="non-plain payoff given"):
        binary.npv()


# ---------------------------------------------------------------------------
# 6. MCHestonHullWhiteEngine / HestonHullWhitePathPricer / Make*
#    (+ HybridHestonHullWhiteProcess)
# ---------------------------------------------------------------------------


def _hhw(inputs: dict[str, Any]) -> HybridHestonHullWhiteProcess:
    risk_free = _flat(inputs["r"])
    heston = HestonProcess(
        risk_free_rate=risk_free,
        dividend_yield=_flat(inputs["q"]),
        s0=SimpleQuote(inputs["s0"]),
        v0=inputs["v0"],
        kappa=inputs["kappa"],
        theta=inputs["theta"],
        sigma=inputs["heston_sigma"],
        rho=inputs["rho"],
    )
    hull_white = HullWhiteForwardProcess(risk_free, inputs["hw_a"], inputs["hw_sigma"])
    hull_white.set_forward_measure_time(inputs["forward_measure_time"])
    scheme = (
        HhwDiscretization.Euler
        if inputs.get("hhw_discretization") == "Euler"
        else HhwDiscretization.BSMHullWhite
    )
    return HybridHestonHullWhiteProcess(
        heston, hull_white, inputs["corr_equity_short_rate"], scheme
    )


def test_hhw_process_primitives(cpp: dict[str, Any]) -> None:
    """``HybridHestonHullWhiteProcess`` itself, before any engine wraps it.

    ``drift`` is checked at LOOSE rather than TIGHT for one reason, recorded
    here rather than waved through: ``HestonProcess.drift`` computes its
    instantaneous forward rates over a 1e-4 window instead of at a point (a
    pre-existing divergence documented in that module), which shifts
    ``drift[0]`` by ~1e-10 relative. ``evolve`` does *not* go through
    ``drift`` — it uses the real step — so every priced number below is
    unaffected and is asserted at TIGHT.
    """
    inputs, expected = (
        cpp["hhw_process_primitives"]["inputs"],
        cpp["hhw_process_primitives"]["expected"],
    )
    process = _hhw(inputs)
    assert process.size() == expected["size"]
    assert process.factors() == expected["factors"]
    tolerance.tight(process.eta(), expected["eta"])

    x0 = process.initial_values()
    for i in range(3):
        tolerance.tight(float(x0[i]), expected[f"x0_{i}"])

    drift = process.drift(inputs["t0"], x0)
    for i in range(3):
        tolerance.loose(float(drift[i]), expected[f"drift_{i}"])

    diffusion = process.diffusion(inputs["t0"], x0)
    for row, col in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)):
        tolerance.tight(
            float(diffusion[row][col]), expected[f"diffusion_{row}{col}"]
        )

    dw = np.array([inputs["dw0"], inputs["dw1"], inputs["dw2"]], dtype=np.float64)
    evolved = process.evolve(inputs["t0"], x0, inputs["dt"], dw)
    for i in range(3):
        tolerance.tight(float(evolved[i]), expected[f"evolve_bsmhw_{i}"])

    euler_inputs = dict(inputs)
    euler_inputs["hhw_discretization"] = "Euler"
    euler = _hhw(euler_inputs).evolve(inputs["t0"], x0, inputs["dt"], dw)
    for i in range(3):
        tolerance.tight(float(euler[i]), expected[f"evolve_euler_{i}"])
    # The two schemes must actually disagree, or the enum is decorative.
    assert expected["evolve_bsmhw_0"] != expected["evolve_euler_0"]

    tolerance.tight(process.numeraire(1.0, evolved), expected["numeraire_at_1"])
    tolerance.tight(process.numeraire(0.5, evolved), expected["numeraire_at_0p5"])


def test_hhw_cases(cpp: dict[str, Any]) -> None:
    """Every MCHestonHullWhiteEngine case, control variate on and off.

    The control-variate cases are the demanding ones: they exercise
    ``controlPathGenerator()``, which drives a *second*
    ``HybridHestonHullWhiteProcess`` (built with ``corrEquityShortRate = 0``)
    from the *same* seed as the pricing generator, and
    ``controlPricingEngine()``, which is
    ``AnalyticHestonHullWhiteEngine(..., 144)``. A port that re-uses the
    pricing generator for the control leg, or re-seeds it, reproduces the
    non-CV cases and misses every CV one.
    """
    names = _names(cpp, "hhw_")
    assert names
    assert any(cpp[n]["inputs"].get("control_variate") for n in names)
    for name in names:
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        builder = (
            MakeMCHestonHullWhiteEngine(_hhw(inputs))
            .with_steps(inputs["steps"])
            .with_samples(inputs["samples"])
            .with_seed(inputs["seed"])
            .with_antithetic_variate(inputs["antithetic"])
            .with_control_variate(inputs["control_variate"])
        )
        option = VanillaOption(
            PlainVanillaPayoff(_option_type(inputs["option_type"]), inputs["strike"]),
            EuropeanExercise(TODAY + inputs["expiry_days"]),
        )
        option.set_pricing_engine(builder.engine())
        _check_results(option, expected)


def test_hhw_builder_and_process_validation(cpp: dict[str, Any]) -> None:
    inputs = cpp["hhw_bsmhw_put_atm"]["inputs"]
    process = _hhw(inputs)

    assert cpp["hhw_make_no_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps not given"):
        MakeMCHestonHullWhiteEngine(process).with_samples(1023).engine()

    assert cpp["hhw_make_both_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps overspecified"):
        MakeMCHestonHullWhiteEngine(process).with_steps(4).with_steps_per_year(
            12
        ).with_samples(1023).engine()

    assert cpp["hhw_make_tolerance_lowdiscrepancy"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="does not allow an error estimate"):
        MakeMCHestonHullWhiteEngine(process, LowDiscrepancy).with_absolute_tolerance(
            0.02
        )

    risk_free = _flat(inputs["r"])
    hull_white = HullWhiteForwardProcess(risk_free, inputs["hw_a"], inputs["hw_sigma"])
    hull_white.set_forward_measure_time(inputs["forward_measure_time"])

    assert cpp["hhw_process_corr_not_positive_definite"]["expected"]["throws"]
    steep = HestonProcess(
        risk_free_rate=risk_free,
        dividend_yield=_flat(inputs["q"]),
        s0=SimpleQuote(inputs["s0"]),
        v0=inputs["v0"],
        kappa=inputs["kappa"],
        theta=inputs["theta"],
        sigma=inputs["heston_sigma"],
        rho=0.8,
    )
    with pytest.raises(
        LibraryException, match="correlation matrix is not positive definite"
    ):
        HybridHestonHullWhiteProcess(steep, hull_white, 0.8)

    # C++ builds the zero-vol HullWhiteForwardProcess happily (its constructor
    # validates nothing) and it is HybridHestonHullWhiteProcess that raises
    # "positive vol of Hull White process is required". PQuantLib's
    # HullWhiteForwardProcess rejects sigma == 0 first, through the positive
    # constraint on the underlying model parameter, so the exception surfaces a
    # step earlier with a different message. Both refuse the configuration,
    # which is what the probe pins; the earlier rejection point is a
    # pre-existing divergence in HullWhiteForwardProcess, outside this cluster.
    assert cpp["hhw_process_zero_hw_sigma"]["expected"]["throws"]

    def _zero_vol_hybrid() -> HybridHestonHullWhiteProcess:
        zero_vol = HullWhiteForwardProcess(risk_free, inputs["hw_a"], 0.0)
        zero_vol.set_forward_measure_time(inputs["forward_measure_time"])
        return HybridHestonHullWhiteProcess(process.heston_process(), zero_vol, -0.4)

    with pytest.raises(LibraryException):
        _zero_vol_hybrid()

    assert cpp["hhw_engine_american_exercise"]["expected"]["throws"]
    american = VanillaOption(
        PlainVanillaPayoff(OptionType.Put, 100.0),
        AmericanExercise(TODAY, TODAY + 365),
    )
    american.set_pricing_engine(
        MakeMCHestonHullWhiteEngine(process)
        .with_steps(4)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="only european exercise is supported"):
        american.npv()
