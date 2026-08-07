"""Cross-validate the v1.43 Monte Carlo *discrete-average Asian* engines vs C++.

Reference: ``migration-harness/references/v143/pe/mcasian`` (probe at
``migration-harness/cpp/probes/v143_pe_mcasian/probe.cpp``).

Covers :class:`MCDiscreteAveragingAsianEngineBase` and its
:class:`PastFixingsOnly` error, :class:`MCDiscreteArithmeticAPEngine` /
:class:`ArithmeticAPOPathPricer` / :class:`MakeMCDiscreteArithmeticAPEngine`,
:class:`MCDiscreteGeometricAPEngine` / :class:`GeometricAPOPathPricer` /
:class:`MakeMCDiscreteGeometricAPEngine`,
:class:`MCDiscreteArithmeticASEngine` / :class:`ArithmeticASOPathPricer` /
:class:`MakeMCDiscreteArithmeticASEngine`,
:class:`MCDiscreteArithmeticAPHestonEngine` /
:class:`ArithmeticAPOHestonPathPricer` /
:class:`MakeMCDiscreteArithmeticAPHestonEngine`, and
:class:`MCDiscreteGeometricAPHestonEngine` /
:class:`GeometricAPOHestonPathPricer` /
:class:`MakeMCDiscreteGeometricAPHestonEngine`.

Why the assertions are exact
----------------------------
Every engine here is deterministic once the seed is fixed: ``PseudoRandom`` is
``InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
InverseCumulativeNormal>`` and ``LowDiscrepancy`` is
``InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>``; neither consults
the clock for a nonzero seed, and every case below pins an explicit nonzero
seed. A correct port therefore reproduces the C++ NPV *and* the C++ error
estimate, so the tier is TIGHT (1e-14 abs / 1e-12 rel) rather than a
confidence band. A band would pass with the wrong fixing count, the wrong
seasoning handling, or the wrong antithetic pairing — which is the entire
failure mode this file exists to catch.

Each case carries its whole market description in ``inputs``, so the tests
rebuild the setup rather than restating constants.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOption
from pquantlib.instruments.average_type import AverageType
from pquantlib.math.randomnumbers.rng_traits import LowDiscrepancy, PseudoRandom
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import CashOrNothingPayoff, OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.mc_discr_arith_av_price import (
    ArithmeticAPOPathPricer,
    MakeMCDiscreteArithmeticAPEngine,
)
from pquantlib.pricingengines.asian.mc_discr_arith_av_price_heston import (
    ArithmeticAPOHestonPathPricer,
    MakeMCDiscreteArithmeticAPHestonEngine,
    MCDiscreteArithmeticAPHestonEngine,
)
from pquantlib.pricingengines.asian.mc_discr_arith_av_strike import (
    MakeMCDiscreteArithmeticASEngine,
)
from pquantlib.pricingengines.asian.mc_discr_geom_av_price import (
    GeometricAPOPathPricer,
    MakeMCDiscreteGeometricAPEngine,
)
from pquantlib.pricingengines.asian.mc_discr_geom_av_price_heston import (
    GeometricAPOHestonPathPricer,
    MakeMCDiscreteGeometricAPHestonEngine,
    MCDiscreteGeometricAPHestonEngine,
)
from pquantlib.pricingengines.asian.mc_discrete_asian_engine_base import (
    PastFixingsOnly,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# probe.cpp — ``const Date kToday(15, May, 2025);`` and
# ``Settings::instance().evaluationDate() = kToday;`` in ``main()``.
TODAY = Date.from_ymd(15, Month.May, 2025)

_DC = Actual365Fixed()
_CAL = NullCalendar()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/mcasian")


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


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(
        reference_date=TODAY, forward_rate=rate, day_counter=_DC
    )


def _bsm(inputs: dict[str, Any]) -> GeneralizedBlackScholesProcess:
    """Rebuild the probe's ``BlackScholesMertonProcess`` from ``inputs``."""
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(inputs["s0"]),
        dividend_ts=_flat(inputs["q"]),
        risk_free_ts=_flat(inputs["r"]),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY,
            calendar=_CAL,
            day_counter=_DC,
            volatility=inputs["vol"],
        ),
    )


def _heston(inputs: dict[str, Any]) -> HestonProcess:
    """Rebuild the probe's ``HestonProcess`` from ``inputs``.

    The probe leaves the discretization at C++'s default,
    ``QuadraticExponentialMartingale`` — not Euler.
    """
    assert inputs["discretization"] == "QuadraticExponentialMartingale"
    return HestonProcess(
        risk_free_rate=_flat(inputs["r"]),
        dividend_yield=_flat(inputs["q"]),
        s0=SimpleQuote(inputs["s0"]),
        v0=inputs["v0"],
        kappa=inputs["kappa"],
        theta=inputs["theta"],
        sigma=inputs["sigma"],
        rho=inputs["rho"],
    )


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


def _average_type(name: str) -> AverageType:
    return AverageType.Arithmetic if name == "Arithmetic" else AverageType.Geometric


def _traits(name: str) -> type[PseudoRandom] | type[LowDiscrepancy]:
    return PseudoRandom if name == "pseudo" else LowDiscrepancy


def _fixings(inputs: dict[str, Any]) -> list[Date]:
    return [TODAY + int(d) for d in inputs["fixing_offsets"]]


def _option(inputs: dict[str, Any]) -> DiscreteAveragingAsianOption:
    return DiscreteAveragingAsianOption(
        _average_type(inputs["average_type"]),
        inputs["running_accumulator"],
        inputs["past_fixings"],
        _fixings(inputs),
        PlainVanillaPayoff(_option_type(inputs["option_type"]), inputs["strike"]),
        EuropeanExercise(TODAY + int(inputs["expiry_days"])),
    )


# ---------------------------------------------------------------------------
# Engine construction, driven entirely by ``inputs["engine"]``
# ---------------------------------------------------------------------------

_BsBuilder = (
    MakeMCDiscreteArithmeticAPEngine
    | MakeMCDiscreteGeometricAPEngine
    | MakeMCDiscreteArithmeticASEngine
)
_HestonBuilder = (
    MakeMCDiscreteArithmeticAPHestonEngine | MakeMCDiscreteGeometricAPHestonEngine
)


def _apply_common(
    builder: _BsBuilder | _HestonBuilder, inputs: dict[str, Any]
) -> None:
    """The knobs every one of the five builders shares."""
    builder.with_antithetic_variate(inputs["antithetic"]).with_seed(inputs["seed"])
    if inputs["samples"] > 0:
        builder.with_samples(inputs["samples"])
    else:
        builder.with_absolute_tolerance(inputs["tolerance"])
    if inputs["max_samples"] > 0:
        builder.with_max_samples(inputs["max_samples"])


def _build(inputs: dict[str, Any]) -> DiscreteAveragingAsianOption:
    """Rebuild the priced option of one probe case."""
    engine_name = inputs["engine"]
    traits = _traits(inputs["rng"])
    option = _option(inputs)

    if engine_name == "MCDiscreteArithmeticAPEngine":
        ap = MakeMCDiscreteArithmeticAPEngine(_bsm(inputs), traits)
        ap.with_brownian_bridge(inputs["brownian_bridge"]).with_control_variate(
            inputs["control_variate"]
        )
        _apply_common(ap, inputs)
        option.set_pricing_engine(ap.engine())
    elif engine_name == "MCDiscreteGeometricAPEngine":
        gp = MakeMCDiscreteGeometricAPEngine(_bsm(inputs), traits)
        gp.with_brownian_bridge(inputs["brownian_bridge"])
        _apply_common(gp, inputs)
        option.set_pricing_engine(gp.engine())
    elif engine_name == "MCDiscreteArithmeticASEngine":
        as_ = MakeMCDiscreteArithmeticASEngine(_bsm(inputs), traits)
        as_.with_brownian_bridge(inputs["brownian_bridge"])
        _apply_common(as_, inputs)
        option.set_pricing_engine(as_.engine())
    elif engine_name == "MCDiscreteArithmeticAPHestonEngine":
        aph = MakeMCDiscreteArithmeticAPHestonEngine(_heston(inputs), traits)
        aph.with_control_variate(inputs["control_variate"])
        if inputs["steps"] > 0:
            aph.with_steps(inputs["steps"])
        if inputs["steps_per_year"] > 0:
            aph.with_steps_per_year(inputs["steps_per_year"])
        _apply_common(aph, inputs)
        option.set_pricing_engine(aph.engine())
    elif engine_name == "MCDiscreteGeometricAPHestonEngine":
        gph = MakeMCDiscreteGeometricAPHestonEngine(_heston(inputs), traits)
        if inputs["steps"] > 0:
            gph.with_steps(inputs["steps"])
        if inputs["steps_per_year"] > 0:
            gph.with_steps_per_year(inputs["steps_per_year"])
        _apply_common(gph, inputs)
        option.set_pricing_engine(gph.engine())
    else:  # pragma: no cover - guards against a stale reference file
        raise AssertionError(f"unknown engine in reference: {engine_name}")
    return option


def _check_results(
    option: DiscreteAveragingAsianOption, expected: dict[str, Any]
) -> None:
    """NPV, error-estimate presence/value, and the realised time grid."""
    tolerance.tight(option.npv(), expected["npv"])
    if expected["has_error_estimate"]:
        tolerance.tight(option.error_estimate(), expected["error_estimate"])
    else:
        # C++ ``if constexpr (RNG::allowsErrorEstimate)`` leaves
        # ``results_.errorEstimate`` at Null for LowDiscrepancy, and
        # ``Instrument::errorEstimate()`` throws. Pin the absence.
        with pytest.raises(LibraryException):
            option.error_estimate()

    # ``results_.additionalResults["TimeGrid"]`` — the grid is the single
    # richest witness that timeGrid() was built the C++ way.
    grid = option.additional_results()["TimeGrid"]
    assert len(grid) == expected["grid_size"]
    for i in range(expected["grid_size"]):
        tolerance.tight(grid[i], expected[f"grid_t{i}"])
    assert len(grid.mandatory_times) == expected["mandatory_size"]
    for i in range(expected["mandatory_size"]):
        tolerance.tight(grid.mandatory_times[i], expected[f"mandatory_t{i}"])


def _priced(cpp: dict[str, Any], prefix: str) -> list[str]:
    return sorted(
        k for k in cpp if k.startswith(prefix) and "npv" in cpp[k]["expected"]
    )


def _run_all(cpp: dict[str, Any], prefix: str, *, minimum: int) -> None:
    names = _priced(cpp, prefix)
    assert len(names) >= minimum, f"only {len(names)} {prefix}* cases in the reference"
    for name in names:
        case = cpp[name]
        _check_results(_build(case["inputs"]), case["expected"])


# ---------------------------------------------------------------------------
# 1. MCDiscreteArithmeticAPEngine
# ---------------------------------------------------------------------------


def test_arithmetic_average_price_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCDiscreteArithmeticAPEngine case, exactly."""
    _run_all(cpp, "arithap_", minimum=20)


def test_arithmetic_seed_actually_reaches_the_generator(cpp: dict[str, Any]) -> None:
    """Two seeds, one otherwise identical setup, two different answers.

    A port that swallows the seed (or substitutes its own — the pre-v1.43
    Python base silently mapped seed 0 to 1) passes the single-seed test and
    fails this one.
    """
    a = cpp["arithap_ps_call_atm"]["expected"]["npv"]
    b = cpp["arithap_ps_call_atm_seed7"]["expected"]["npv"]
    assert a != b
    tolerance.tight(_build(cpp["arithap_ps_call_atm"]["inputs"]).npv(), a)
    tolerance.tight(_build(cpp["arithap_ps_call_atm_seed7"]["inputs"]).npv(), b)


def test_t0_fixing_changes_the_fixing_count(cpp: dict[str, Any]) -> None:
    """A fixing on the evaluation date puts ``path[0]`` into the average.

    The two cases differ ONLY in whether the first fixing is at t=0, so a port
    that always skips (or always includes) ``path[0]`` cannot match both.
    """
    without = cpp["arithap_ps_call_atm"]
    with_t0 = cpp["arithap_ps_call_t0fix"]
    assert without["expected"]["mandatory_t0"] != 0.0
    assert with_t0["expected"]["mandatory_t0"] == 0.0
    # 12 mandatory times either way, but one extra grid point when none is at 0
    assert without["expected"]["grid_size"] == 13
    assert with_t0["expected"]["grid_size"] == 12
    _check_results(_build(without["inputs"]), without["expected"])
    _check_results(_build(with_t0["inputs"]), with_t0["expected"])


def test_control_variate_clamps_negative_values_to_zero(cpp: dict[str, Any]) -> None:
    """``calculate()`` clamps to ``max(0, value)`` when the CV is on.

    The probe searched seeds until the raw CV estimator went negative, so this
    case reaches the clamp rather than merely asserting about it.
    """
    case = cpp["arithap_ps_put_cv_clamped"]
    assert case["expected"]["clamped"] is True
    assert case["expected"]["npv"] == 0.0
    option = _build(case["inputs"])
    _check_results(option, case["expected"])
    # Exactly zero, with a strictly positive standard error: the clamp fired.
    assert option.npv() == 0.0
    assert option.error_estimate() > 0.0


def test_control_variate_reduces_the_standard_error(cpp: dict[str, Any]) -> None:
    """The geometric CV is doing real work, not just producing a number."""
    plain = cpp["arithap_ps_call_atm"]["expected"]["error_estimate"]
    with_cv = cpp["arithap_ps_call_cv"]["expected"]["error_estimate"]
    assert with_cv < plain / 5.0
    tolerance.tight(_build(cpp["arithap_ps_call_cv"]["inputs"]).error_estimate(), with_cv)


# ---------------------------------------------------------------------------
# 2. MCDiscreteGeometricAPEngine
# ---------------------------------------------------------------------------


def test_geometric_average_price_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCDiscreteGeometricAPEngine case, exactly."""
    _run_all(cpp, "geomap_", minimum=12)


def test_geometric_seasoning_uses_a_running_product(cpp: dict[str, Any]) -> None:
    """``runningAccumulator`` is a PRODUCT here and a SUM for the arithmetic.

    Both seasoned cases carry six past fixings; a port that sums the geometric
    accumulator (or multiplies the arithmetic one) mismatches immediately.
    """
    geom = cpp["geomap_ps_call_seasoned"]
    arith = cpp["arithap_ps_call_seasoned"]
    assert geom["inputs"]["running_accumulator"] > 1e11  # 103^6
    assert arith["inputs"]["running_accumulator"] == 618.0
    _check_results(_build(geom["inputs"]), geom["expected"])
    _check_results(_build(arith["inputs"]), arith["expected"])


# ---------------------------------------------------------------------------
# 3. MCDiscreteArithmeticASEngine
# ---------------------------------------------------------------------------


def test_arithmetic_average_strike_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCDiscreteArithmeticASEngine case, exactly."""
    _run_all(cpp, "arithas_", minimum=12)


def test_average_strike_appends_the_exercise_date(cpp: dict[str, Any]) -> None:
    """``includeExerciseDate`` is true only for the average-strike engine.

    With the exercise 40 days after the last fixing the AS grid grows by one
    point (14 vs 13) while the AP grid does not — and the extra point must be
    excluded from the average via ``fixingCount``.
    """
    as_case = cpp["arithas_ps_call_exercise_after_fixings"]
    ap_case = cpp["arithap_ps_call_exercise_after_fixings"]
    assert as_case["inputs"]["expiry_days"] == 400
    assert ap_case["inputs"]["expiry_days"] == 400
    assert as_case["expected"]["grid_size"] == 14
    assert as_case["expected"]["mandatory_size"] == 13
    assert ap_case["expected"]["grid_size"] == 13
    assert ap_case["expected"]["mandatory_size"] == 12
    _check_results(_build(as_case["inputs"]), as_case["expected"])
    _check_results(_build(ap_case["inputs"]), ap_case["expected"])


# ---------------------------------------------------------------------------
# 4. MCDiscreteArithmeticAPHestonEngine
# ---------------------------------------------------------------------------


def test_arithmetic_average_price_heston_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCDiscreteArithmeticAPHestonEngine case, exactly."""
    _run_all(cpp, "arithapheston_", minimum=15)


def test_heston_extra_steps_do_not_join_the_average(cpp: dict[str, Any]) -> None:
    """Extra grid points between fixings must not enter the average.

    ``withSteps(24)`` doubles the grid; the fixing set is unchanged, so the
    price moves only through the finer variance discretization. A port that
    averages the whole path instead of ``timeGrid.closestIndex(mandatory)``
    gets a visibly different number.
    """
    base = cpp["arithapheston_ps_call_atm"]
    fine = cpp["arithapheston_ps_call_steps24"]
    assert base["expected"]["grid_size"] == 13
    assert fine["expected"]["grid_size"] == 25
    assert base["expected"]["mandatory_size"] == fine["expected"]["mandatory_size"] == 12
    _check_results(_build(base["inputs"]), base["expected"])
    _check_results(_build(fine["inputs"]), fine["expected"])


def test_heston_steps_per_year_truncates(cpp: dict[str, Any]) -> None:
    """``Size(timeStepsPerYear * t)`` truncates, with ``t`` the exercise time."""
    case = cpp["arithapheston_ps_call_stepsperyear52"]
    assert case["inputs"]["steps_per_year"] == 52
    _check_results(_build(case["inputs"]), case["expected"])


# ---------------------------------------------------------------------------
# 5. MCDiscreteGeometricAPHestonEngine
# ---------------------------------------------------------------------------


def test_geometric_average_price_heston_cases(cpp: dict[str, Any]) -> None:
    """Every priced MCDiscreteGeometricAPHestonEngine case, exactly."""
    _run_all(cpp, "geomapheston_", minimum=9)


# ---------------------------------------------------------------------------
# 6. detail::PastFixingsOnly and the other engine-level guards
# ---------------------------------------------------------------------------


def _past_fixings_option(average_type: AverageType, offsets: list[int]) -> Any:
    accumulator = 1.0 if average_type == AverageType.Geometric else 600.0
    past = 0 if len(offsets) == 1 else 6
    return DiscreteAveragingAsianOption(
        average_type,
        accumulator,
        past,
        [TODAY + d for d in offsets],
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(TODAY + 360),
    )


# probe.cpp — kFixAllPast / kFixOnlyT0
_ALL_PAST = [-180, -150, -120, -90, -60, -30]
_ONLY_T0 = [0]


def test_past_fixings_only_is_a_real_exception_type(cpp: dict[str, Any]) -> None:
    """``detail::PastFixingsOnly`` — a distinct type, not just a message.

    C++ ``calculate()`` catches it by type so a future revision can compute the
    fully-determined payoff instead of failing; a port that raises a generic
    error loses that hook.
    """
    for name, offsets, average_type in (
        ("throw_past_fixings_only_all_past", _ALL_PAST, AverageType.Arithmetic),
        ("throw_past_fixings_only_single_t0", _ONLY_T0, AverageType.Arithmetic),
    ):
        expected = cpp[name]["expected"]
        assert expected["throws"] is True
        option = _past_fixings_option(average_type, offsets)
        process = _bsm(cpp["arithap_ps_call_atm"]["inputs"])
        option.set_pricing_engine(
            MakeMCDiscreteArithmeticAPEngine(process)
            .with_samples(1023)
            .with_seed(42)
            .engine()
        )
        with pytest.raises(PastFixingsOnly, match=expected["what"]):
            option.npv()
    assert issubclass(PastFixingsOnly, LibraryException)


def test_past_fixings_only_from_every_engine(cpp: dict[str, Any]) -> None:
    """The guard lives in the shared base, so all five engines raise it."""
    bs = _bsm(cpp["arithap_ps_call_atm"]["inputs"])
    hp = _heston(cpp["arithapheston_ps_call_atm"]["inputs"])

    geometric = _past_fixings_option(AverageType.Geometric, _ALL_PAST)
    geometric.set_pricing_engine(
        MakeMCDiscreteGeometricAPEngine(bs).with_samples(1023).with_seed(42).engine()
    )
    assert cpp["throw_past_fixings_only_geometric"]["expected"]["throws"] is True
    with pytest.raises(PastFixingsOnly):
        geometric.npv()

    strike = _past_fixings_option(AverageType.Arithmetic, _ALL_PAST)
    strike.set_pricing_engine(
        MakeMCDiscreteArithmeticASEngine(bs).with_samples(1023).with_seed(42).engine()
    )
    assert cpp["throw_past_fixings_only_averagestrike"]["expected"]["throws"] is True
    with pytest.raises(PastFixingsOnly):
        strike.npv()

    heston_opt = _past_fixings_option(AverageType.Arithmetic, _ALL_PAST)
    heston_opt.set_pricing_engine(
        MakeMCDiscreteArithmeticAPHestonEngine(hp)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    assert cpp["throw_past_fixings_only_heston"]["expected"]["throws"] is True
    with pytest.raises(PastFixingsOnly):
        heston_opt.npv()


def test_neither_samples_nor_tolerance(cpp: dict[str, Any]) -> None:
    case = cpp["throw_no_samples_no_tolerance"]
    assert case["expected"]["throws"] is True
    option = _option(cpp["arithap_ps_call_atm"]["inputs"])
    option.set_pricing_engine(
        MakeMCDiscreteArithmeticAPEngine(_bsm(cpp["arithap_ps_call_atm"]["inputs"]))
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match=case["expected"]["what"]):
        option.npv()


def test_tolerance_max_samples_exceeded(cpp: dict[str, Any]) -> None:
    case = cpp["throw_tolerance_maxsamples_exceeded"]
    assert case["expected"]["throws"] is True
    inputs = cpp["arithap_ps_call_atm"]["inputs"]
    option = _option(inputs)
    option.set_pricing_engine(
        # probe.cpp leaves withBrownianBridge unset here, i.e. at the builder
        # default of True.
        MakeMCDiscreteArithmeticAPEngine(_bsm(inputs))
        .with_absolute_tolerance(1e-4)
        .with_max_samples(2000)
        .with_seed(42)
        .engine()
    )
    # The C++ message interpolates the error and tolerance at default stream
    # precision, which Python spells differently; match the invariant prefix.
    assert case["expected"]["what"].startswith("max number of samples (2000) reached")
    with pytest.raises(LibraryException, match=r"max number of samples \(2000\) reached"):
        option.npv()


@pytest.mark.parametrize(
    ("case_name", "engine_kind"),
    [
        ("throw_arithap_non_plain_payoff", "arithap"),
        ("throw_geomap_non_plain_payoff", "geomap"),
        ("throw_arithapheston_non_plain_payoff", "arithapheston"),
    ],
)
def test_non_plain_payoff_rejected(
    cpp: dict[str, Any], case_name: str, engine_kind: str
) -> None:
    expected = cpp[case_name]["expected"]
    assert expected["throws"] is True
    bs_inputs = cpp["arithap_ps_call_atm"]["inputs"]
    fixings = _fixings(bs_inputs)
    payoff = CashOrNothingPayoff(OptionType.Call, 100.0, 10.0)
    exercise = EuropeanExercise(TODAY + 360)

    if engine_kind == "geomap":
        option = DiscreteAveragingAsianOption(
            AverageType.Geometric, 1.0, 0, fixings, payoff, exercise
        )
        option.set_pricing_engine(
            MakeMCDiscreteGeometricAPEngine(_bsm(bs_inputs))
            .with_samples(1023)
            .with_seed(42)
            .engine()
        )
    elif engine_kind == "arithapheston":
        option = DiscreteAveragingAsianOption(
            AverageType.Arithmetic, 0.0, 0, fixings, payoff, exercise
        )
        option.set_pricing_engine(
            MakeMCDiscreteArithmeticAPHestonEngine(
                _heston(cpp["arithapheston_ps_call_atm"]["inputs"])
            )
            .with_samples(1023)
            .with_seed(42)
            .engine()
        )
    else:
        option = DiscreteAveragingAsianOption(
            AverageType.Arithmetic, 0.0, 0, fixings, payoff, exercise
        )
        option.set_pricing_engine(
            MakeMCDiscreteArithmeticAPEngine(_bsm(bs_inputs))
            .with_samples(1023)
            .with_seed(42)
            .engine()
        )
    with pytest.raises(LibraryException, match=expected["what"]):
        option.npv()


@pytest.mark.parametrize(
    ("case_name", "engine_kind"),
    [
        ("throw_arithap_american_exercise", "arithap"),
        ("throw_arithas_american_exercise", "arithas"),
    ],
)
def test_american_exercise_rejected(
    cpp: dict[str, Any], case_name: str, engine_kind: str
) -> None:
    expected = cpp[case_name]["expected"]
    assert expected["throws"] is True
    bs_inputs = cpp["arithap_ps_call_atm"]["inputs"]
    fixings = _fixings(bs_inputs)
    last_day = 360 if engine_kind == "arithap" else 400
    strike = 100.0 if engine_kind == "arithap" else 0.0
    option = DiscreteAveragingAsianOption(
        AverageType.Arithmetic,
        0.0,
        0,
        fixings,
        PlainVanillaPayoff(OptionType.Call, strike),
        AmericanExercise(TODAY, TODAY + last_day),
    )
    builder: _BsBuilder = (
        MakeMCDiscreteArithmeticAPEngine(_bsm(bs_inputs))
        if engine_kind == "arithap"
        else MakeMCDiscreteArithmeticASEngine(_bsm(bs_inputs))
    )
    option.set_pricing_engine(builder.with_samples(1023).with_seed(42).engine())
    with pytest.raises(LibraryException, match=expected["what"]):
        option.npv()


def test_path_pricer_strike_guards(cpp: dict[str, Any]) -> None:
    """Each path pricer's own strike guard, message included.

    The arithmetic and geometric single-variate pricers use *different* wording
    for the same condition; both are pinned so a port cannot unify them.
    """
    arith = cpp["throw_arithap_pathpricer_negative_strike"]["expected"]
    geom = cpp["throw_geomap_pathpricer_negative_strike"]["expected"]
    arith_h = cpp["throw_arithapheston_pathpricer_negative_strike"]["expected"]
    geom_h = cpp["throw_geomapheston_pathpricer_negative_strike"]["expected"]
    assert arith["what"] != geom["what"]

    with pytest.raises(LibraryException, match=arith["what"]):
        ArithmeticAPOPathPricer(OptionType.Call, -1.0, 0.95)
    with pytest.raises(LibraryException, match=geom["what"]):
        GeometricAPOPathPricer(OptionType.Call, -1.0, 0.95)
    with pytest.raises(LibraryException, match=arith_h["what"]):
        ArithmeticAPOHestonPathPricer(OptionType.Call, -1.0, 0.95, [1, 2])
    with pytest.raises(LibraryException, match=geom_h["what"]):
        GeometricAPOHestonPathPricer(OptionType.Call, -1.0, 0.95, [1, 2])


def test_heston_engines_reject_both_step_specifications(cpp: dict[str, Any]) -> None:
    """The engine constructors guard even though the builders already do."""
    process = _heston(cpp["arithapheston_ps_call_atm"]["inputs"])
    for name, factory in (
        (
            "throw_arithapheston_engine_both_steps",
            lambda: MCDiscreteArithmeticAPHestonEngine(
                process, required_samples=1023, seed=42, time_steps=24,
                time_steps_per_year=24,
            ),
        ),
        (
            "throw_geomapheston_engine_both_steps",
            lambda: MCDiscreteGeometricAPHestonEngine(
                process, required_samples=1023, seed=42, time_steps=24,
                time_steps_per_year=24,
            ),
        ),
    ):
        expected = cpp[name]["expected"]
        assert expected["throws"] is True
        with pytest.raises(LibraryException, match=expected["what"]):
            factory()


# ---------------------------------------------------------------------------
# 7. MakeMC* builder validation
# ---------------------------------------------------------------------------


def _bs_builder(kind: str, process: GeneralizedBlackScholesProcess, traits: Any) -> Any:
    return {
        "arithap": MakeMCDiscreteArithmeticAPEngine,
        "geomap": MakeMCDiscreteGeometricAPEngine,
        "arithas": MakeMCDiscreteArithmeticASEngine,
    }[kind](process, traits)


def _heston_builder(kind: str, process: HestonProcess, traits: Any) -> Any:
    return {
        "arithapheston": MakeMCDiscreteArithmeticAPHestonEngine,
        "geomapheston": MakeMCDiscreteGeometricAPHestonEngine,
    }[kind](process, traits)


@pytest.mark.parametrize("kind", ["arithap", "geomap", "arithas"])
def test_black_scholes_builder_validation(cpp: dict[str, Any], kind: str) -> None:
    """The samples/tolerance interlocks, per builder."""
    process = _bsm(cpp["arithap_ps_call_atm"]["inputs"])

    after_tol = cpp[f"make_{kind}_samples_after_tolerance"]["expected"]
    assert after_tol["throws"] is True
    with pytest.raises(LibraryException, match=after_tol["what"]):
        _bs_builder(kind, process, PseudoRandom).with_absolute_tolerance(
            0.02
        ).with_samples(1023)

    after_samples = cpp[f"make_{kind}_tolerance_after_samples"]["expected"]
    assert after_samples["throws"] is True
    with pytest.raises(LibraryException, match=after_samples["what"]):
        _bs_builder(kind, process, PseudoRandom).with_samples(
            1023
        ).with_absolute_tolerance(0.02)

    ld = cpp[f"make_{kind}_tolerance_lowdiscrepancy"]["expected"]
    assert ld["throws"] is True
    with pytest.raises(LibraryException, match=ld["what"]):
        _bs_builder(kind, process, LowDiscrepancy).with_absolute_tolerance(0.02)


def test_arithmetic_ap_builder_defaults_build(cpp: dict[str, Any]) -> None:
    """``withSamples`` alone is enough — there is no steps knob to forget.

    C++ ``bool brownianBridge_ = true`` is the *builder's* member initialiser,
    the opposite of the engine constructor's default. probe.cpp's
    ``arithap_ps_put_cv_clamped`` case deliberately leaves
    ``withBrownianBridge`` unset, so rebuilding it here without touching that
    knob pins the default *behaviourally* — a port that defaults it to False
    reproduces neither the paths nor the error estimate.
    """
    assert cpp["make_arithap_defaults_ok"]["expected"]["throws"] is False
    case = cpp["arithap_ps_put_cv_clamped"]
    inputs = case["inputs"]
    assert inputs["brownian_bridge"] is True
    option = _option(inputs)
    option.set_pricing_engine(
        MakeMCDiscreteArithmeticAPEngine(_bsm(inputs))
        .with_control_variate(True)
        .with_samples(inputs["samples"])
        .with_seed(inputs["seed"])
        .engine()
    )
    tolerance.tight(option.npv(), case["expected"]["npv"])
    tolerance.tight(option.error_estimate(), case["expected"]["error_estimate"])


@pytest.mark.parametrize("kind", ["arithapheston", "geomapheston"])
def test_heston_builder_validation(cpp: dict[str, Any], kind: str) -> None:
    """Steps guards fire in the SETTERS; ``engine()`` validates nothing."""
    process = _heston(cpp["arithapheston_ps_call_atm"]["inputs"])

    steps_after = cpp[f"make_{kind}_steps_after_stepsperyear"]["expected"]
    assert steps_after["throws"] is True
    with pytest.raises(LibraryException, match=steps_after["what"]):
        _heston_builder(kind, process, PseudoRandom).with_steps_per_year(
            12
        ).with_steps(24)

    spy_after = cpp[f"make_{kind}_stepsperyear_after_steps"]["expected"]
    assert spy_after["throws"] is True
    with pytest.raises(LibraryException, match=spy_after["what"]):
        _heston_builder(kind, process, PseudoRandom).with_steps(
            24
        ).with_steps_per_year(12)

    after_tol = cpp[f"make_{kind}_samples_after_tolerance"]["expected"]
    assert after_tol["throws"] is True
    with pytest.raises(LibraryException, match=after_tol["what"]):
        _heston_builder(kind, process, PseudoRandom).with_absolute_tolerance(
            0.02
        ).with_samples(1023)

    after_samples = cpp[f"make_{kind}_tolerance_after_samples"]["expected"]
    assert after_samples["throws"] is True
    with pytest.raises(LibraryException, match=after_samples["what"]):
        _heston_builder(kind, process, PseudoRandom).with_samples(
            1023
        ).with_absolute_tolerance(0.02)

    ld = cpp[f"make_{kind}_tolerance_lowdiscrepancy"]["expected"]
    assert ld["throws"] is True
    with pytest.raises(LibraryException, match=ld["what"]):
        _heston_builder(kind, process, LowDiscrepancy).with_absolute_tolerance(0.02)


def test_heston_builder_accepts_no_steps(cpp: dict[str, Any]) -> None:
    """Null steps are legal for this family: one step per fixing.

    There is no "number of steps not given" guard here, unlike every
    ``MakeMCVanilla*`` builder. Rebuilding the pinned
    ``arithapheston_ps_call_atm`` case without touching withSteps /
    withStepsPerYear must give a 13-point grid for 12 fixings and the pinned
    NPV.
    """
    assert cpp["make_arithapheston_no_steps_ok"]["expected"]["throws"] is False
    case = cpp["arithapheston_ps_call_atm"]
    inputs = case["inputs"]
    assert inputs["steps"] == -1
    assert inputs["steps_per_year"] == -1
    option = _option(inputs)
    option.set_pricing_engine(
        MakeMCDiscreteArithmeticAPHestonEngine(_heston(inputs))
        .with_samples(inputs["samples"])
        .with_seed(inputs["seed"])
        .engine()
    )
    _check_results(option, case["expected"])
