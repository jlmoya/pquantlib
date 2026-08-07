"""MCHullWhiteCapFloorEngine + HullWhiteCapFloorPricer + the Make builder.

Reference: ``migration-harness/references/v143/pe/swaption.json``, produced by
``migration-harness/cpp/probes/v143_pe_swaption/probe.cpp``.

The Monte Carlo stream is fully deterministic given a seed, so every NPV and
error estimate below is pinned to the EXACT C++ value at TIGHT, not to a
statistical band. The one place where bit-for-bit is unattainable is
documented on ``test_rng_stream_matches_cpp``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest

from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.cap_floor import Cap, CapFloor, CapFloorArguments, Floor
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.capfloor.analytic_capfloor_engine import (
    AffineModelLike,
    AnalyticCapFloorEngine,
)
from pquantlib.pricingengines.capfloor.mc_hull_white_engine import (
    HullWhiteCapFloorPricer,
    MakeMCHullWhiteCapFloorEngine,
)
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_grid import TimeGrid
from pquantlib.time.time_unit import TimeUnit

# probe.cpp:293 / :657-659 / :686.
TODAY = Date.from_ymd(3, Month.March, 2025)
FORWARDING_RATE = 0.03
HW_A = 0.05
HW_SIGMA = 0.01
STRIKE = 0.03

# Bound on the MT19937 -> InverseCumulativeNormal -> HullWhiteForwardProcess
# path chain. The uniforms are bit-identical (MT19937 is exact integer
# arithmetic); the Acklam rational approximation that maps them to normals is
# transcribed constant-for-constant from normaldistribution.cpp, and the
# Halley refinement step there is #ifdef'd OUT in v1.43
# (normaldistribution.hpp:127-133), so the only difference is the order of
# floating-point operations in the two degree-5 polynomial evaluations. That
# costs a handful of ULPs: the largest observed deviation over the pinned
# stream is 2.5e-14 absolute on a normal variate of order 1.7, and it
# propagates to at most 6e-16 absolute on the resulting short rate and
# ~4e-15 relative on the 1023-sample mean. TIGHT (1e-12 rel / 1e-14 abs)
# covers it with two decades of margin.
_MC_TOL = tolerance.tight


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/swaption")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:293 — Settings::instance().evaluationDate() = Date(3, March, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _curve() -> FlatForward:
    return FlatForward.from_rate(
        TODAY, FORWARDING_RATE, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )


def _index(curve: FlatForward) -> Euribor:
    return Euribor(Period(6, TimeUnit.Months), curve)


def _leg(curve: FlatForward) -> list[Any]:
    # probe.cpp:660-671.
    idx = _index(curve)
    cal = TARGET()
    start = idx.fixing_calendar().advance(TODAY, idx.fixing_days(), TimeUnit.Days)
    end = cal.advance(start, 5, TimeUnit.Years)
    sched = Schedule.from_rule(
        start, end, Period(6, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    return ibor_leg(
        sched,
        idx,
        nominals=[1.0],
        payment_day_counter=Actual360(),
        payment_adjustment=BusinessDayConvention.ModifiedFollowing,
        fixing_days=idx.fixing_days(),
    )


def _instrument(is_cap: bool, curve: FlatForward) -> CapFloor:
    leg = _leg(curve)
    return Cap(leg, [STRIKE]) if is_cap else Floor(leg, [STRIKE])


def _model(curve: FlatForward) -> HullWhite:
    return HullWhite(curve, HW_A, HW_SIGMA)


def _args(is_cap: bool, curve: FlatForward) -> CapFloorArguments:
    args = CapFloorArguments()
    _instrument(is_cap, curve).setup_arguments(args)
    return args


def _time_grid(args: CapFloorArguments, curve: FlatForward) -> TimeGrid:
    dc = curve.day_counter()
    ref = curve.reference_date()
    times = [dc.year_fraction(ref, d) for d in args.fixing_dates if d > ref]
    times.append(dc.year_fraction(ref, args.end_dates[-1]))
    return TimeGrid.with_mandatory(times)


# --------------------------------------------------------------------------
# Stream + grid + path: bisection points for any NPV mismatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 12345])
def test_rng_stream_matches_cpp(cpp: dict[str, Any], seed: int) -> None:
    """PseudoRandom::make_sequence_generator(4, seed), three draws."""
    case = cpp[f"mc_rng_seed{seed}"]
    inputs, expected = case["inputs"], case["expected"]
    gen = make_pseudo_random_rsg(inputs["dimension"], seed)
    got: list[float] = []
    for _ in range(inputs["draws"]):
        got.extend(float(v) for v in gen.next_sequence().value)
    assert len(got) == len(expected["values"])
    for a, c in zip(got, expected["values"], strict=True):
        _MC_TOL(a, c)


def test_time_grid_and_arguments(cpp: dict[str, Any]) -> None:
    """Only FUTURE fixing times enter the grid, then the last end date."""
    expected = cpp["mc_time_grid"]["expected"]
    curve = _curve()
    args = _args(True, curve)
    grid = _time_grid(args, curve)
    assert len(grid) == expected["time_grid_size"]
    for a, c in zip(list(grid), expected["time_grid"], strict=True):
        _MC_TOL(a, c)
    assert [str(d) for d in args.start_dates] == expected["start_dates"]
    assert [str(d) for d in args.end_dates] == expected["end_dates"]
    assert [str(d) for d in args.fixing_dates] == expected["fixing_dates"]
    for a, c in zip(args.accrual_times, expected["accrual_times"], strict=True):
        _MC_TOL(a, c)
    for a, c in zip(args.forwards, expected["forwards"], strict=True):
        _MC_TOL(a, c)
    # The first caplet fixes ON the reference date, so it is NOT a grid
    # point — which is exactly why HullWhiteCapFloorPricer needs its
    # past_fixings offset. Assert that structural fact: 10 caplets, but only
    # 9 future fixings, plus the maturity and the implicit t = 0 -> 11 nodes.
    assert args.fixing_dates[0] == curve.reference_date()
    future_fixings = [d for d in args.fixing_dates if d > curve.reference_date()]
    assert len(future_fixings) == len(args.fixing_dates) - 1
    assert len(grid) == len(future_fixings) + 2


@pytest.mark.parametrize("seed", [42, 12345])
def test_path_generation(cpp: dict[str, Any], seed: int) -> None:
    expected = cpp[f"mc_path_seed{seed}"]["expected"]
    curve = _curve()
    args = _args(True, curve)
    grid = _time_grid(args, curve)
    dc = curve.day_counter()
    fmt = dc.year_fraction(curve.reference_date(), args.end_dates[-1])
    process = HullWhiteForwardProcess(curve, HW_A, HW_SIGMA)
    process.set_forward_measure_time(fmt)
    gen = make_pseudo_random_rsg(len(grid) - 1, seed)
    pgen = PathGenerator.with_time_grid(process, grid, gen, brownian_bridge=False)

    # PathGenerator reuses a single Path buffer across calls (matching the
    # C++ member ``next_``), so each sample must be copied out BEFORE the
    # next draw overwrites it.
    paths = {
        "path0": [float(v) for v in pgen.next().value.values],
        "antithetic0": [float(v) for v in pgen.antithetic().value.values],
        "path1": [float(v) for v in pgen.next().value.values],
    }
    for label, got in paths.items():
        for a, c in zip(got, expected[label], strict=True):
            _MC_TOL(a, c)


# --------------------------------------------------------------------------
# HullWhiteCapFloorPricer, isolated from the RNG
# --------------------------------------------------------------------------

_PATHPRICER_CASES = sorted(
    k for k in reference_reader.load("v143/pe/swaption") if k.startswith("mc_pathpricer_")
)


def test_pathpricer_case_count() -> None:
    assert len(_PATHPRICER_CASES) == 8


@pytest.mark.parametrize("name", _PATHPRICER_CASES)
def test_hull_white_capfloor_pricer(cpp: dict[str, Any], name: str) -> None:
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    is_cap = inputs["instrument"] == "Cap"
    curve = _curve()
    args = _args(is_cap, curve)
    grid = _time_grid(args, curve)
    fmt = curve.day_counter().year_fraction(
        curve.reference_date(), args.end_dates[-1]
    )
    _MC_TOL(fmt, inputs["forward_measure_time"])

    pricer = HullWhiteCapFloorPricer(args, _model(curve), fmt)
    values = np.array(inputs["path_values"], dtype=np.float64)
    assert values.size == len(grid)
    _MC_TOL(pricer(Path(grid, values)), expected["payoff"])


def test_pathpricer_distinguishes_cap_and_floor(cpp: dict[str, Any]) -> None:
    """A ramp up is worthless to the floor and valuable to the cap, and back."""
    assert cpp["mc_pathpricer_cap_v1"]["expected"]["payoff"] > 0.0
    assert cpp["mc_pathpricer_cap_v2"]["expected"]["payoff"] == 0.0
    assert cpp["mc_pathpricer_floor_v2"]["expected"]["payoff"] > 0.0


# --------------------------------------------------------------------------
# Full engine
# --------------------------------------------------------------------------

_ENGINE_CASES = sorted(
    k
    for k, v in reference_reader.load("v143/pe/swaption").items()
    if v["inputs"].get("engine") == "MCHullWhiteCapFloorEngine"
)


def test_engine_case_count() -> None:
    assert len(_ENGINE_CASES) == 9


@pytest.mark.parametrize("name", _ENGINE_CASES)
def test_mc_hull_white_capfloor_engine(cpp: dict[str, Any], name: str) -> None:
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    curve = _curve()
    instrument = _instrument(inputs["instrument"] == "Cap", curve)
    instrument.set_pricing_engine(
        MakeMCHullWhiteCapFloorEngine(_model(curve))
        .with_brownian_bridge(inputs["brownian_bridge"])
        .with_antithetic_variate(inputs["antithetic_variate"])
        .with_samples(inputs["required_samples"])
        .with_seed(inputs["seed"])
        .engine()
    )
    assert not expected["throws"]
    _MC_TOL(instrument.npv(), expected["npv"])
    _MC_TOL(instrument.error_estimate(), expected["error_estimate"])


def test_tolerance_driven_sampling(cpp: dict[str, Any]) -> None:
    """withAbsoluteTolerance drives the adaptive sample-growth loop."""
    case = cpp["mc_cap_tolerance_1em4_seed42"]
    inputs, expected = case["inputs"], case["expected"]
    curve = _curve()
    cap = _instrument(True, curve)
    cap.set_pricing_engine(
        MakeMCHullWhiteCapFloorEngine(_model(curve))
        .with_absolute_tolerance(inputs["required_tolerance"])
        .with_max_samples(inputs["max_samples"])
        .with_seed(inputs["seed"])
        .engine()
    )
    _MC_TOL(cap.npv(), expected["npv"])
    _MC_TOL(cap.error_estimate(), expected["error_estimate"])
    assert cap.error_estimate() < inputs["required_tolerance"]


def test_each_knob_moves_the_answer(cpp: dict[str, Any]) -> None:
    """Antithetic / brownian bridge / seed / sample count must all matter."""
    base = cpp["mc_cap_n1023_seed42"]["expected"]
    for other in (
        "mc_cap_n4095_seed42",
        "mc_cap_n1023_seed12345",
        "mc_cap_n1023_seed42_antithetic",
        "mc_cap_n1023_seed42_bridge",
        "mc_cap_n1023_seed42_bridge_antithetic",
    ):
        assert cpp[other]["expected"]["npv"] != base["npv"], other
    # Antithetic sampling halves the standard error.
    assert (
        cpp["mc_cap_n1023_seed42_antithetic"]["expected"]["error_estimate"]
        < base["error_estimate"]
    )


@pytest.mark.parametrize("is_cap", [True, False])
def test_mc_agrees_with_the_analytic_engine(cpp: dict[str, Any], is_cap: bool) -> None:
    """Independent check that the MC prices the right instrument."""
    key = "mc_analytic_benchmark_cap" if is_cap else "mc_analytic_benchmark_floor"
    expected = cpp[key]["expected"]
    curve = _curve()
    instrument = _instrument(is_cap, curve)
    instrument.set_pricing_engine(
        AnalyticCapFloorEngine(cast("AffineModelLike", _model(curve)), curve)
    )
    _MC_TOL(instrument.npv(), expected["npv"])

    mc_key = "mc_cap_n4095_seed42" if is_cap else "mc_floor_n4095_seed42"
    mc = cpp[mc_key]["expected"]
    # 4095 antithetic-free samples: the MC mean must sit within 3 standard
    # errors of the analytic price. This is a sanity band on the C++ numbers
    # themselves, not the cross-validation (which is exact above).
    assert abs(mc["npv"] - expected["npv"]) < 3.0 * mc["error_estimate"]


# --------------------------------------------------------------------------
# MakeMCHullWhiteCapFloorEngine guards
# --------------------------------------------------------------------------


def test_samples_then_tolerance_rejected(cpp: dict[str, Any]) -> None:
    assert cpp["mc_make_samples_then_tolerance_throws"]["expected"]["throws"]
    maker = MakeMCHullWhiteCapFloorEngine(_model(_curve())).with_samples(1023)
    with pytest.raises(LibraryException, match=r"already set"):
        maker.with_absolute_tolerance(1e-4)


def test_tolerance_then_samples_rejected(cpp: dict[str, Any]) -> None:
    assert cpp["mc_make_tolerance_then_samples_throws"]["expected"]["throws"]
    maker = MakeMCHullWhiteCapFloorEngine(_model(_curve())).with_absolute_tolerance(1e-4)
    with pytest.raises(LibraryException, match="already set"):
        maker.with_samples(1023)


def test_no_termination_criterion_rejected(cpp: dict[str, Any]) -> None:
    assert cpp["mc_make_no_criterion_throws"]["expected"]["throws"]
    curve = _curve()
    cap = _instrument(True, curve)
    cap.set_pricing_engine(
        MakeMCHullWhiteCapFloorEngine(_model(curve)).with_seed(42).engine()
    )
    with pytest.raises(LibraryException, match="neither tolerance nor number of samples"):
        cap.npv()


def test_builder_has_no_control_variate(cpp: dict[str, Any]) -> None:
    """v1.43 fact: no control variate exists on this engine.

    ``MCHullWhiteCapFloorEngine`` passes ``controlVariate = false`` to
    ``McSimulation`` (mchullwhiteengine.hpp:81) and the builder exposes no
    ``withControlVariate``. Pinned so a port does not invent one — and so a
    later C++ version that adds one shows up as a probe diff.
    """
    expected = cpp["mc_make_no_control_variate_builder"]["expected"]
    assert expected["has_with_control_variate"] is False
    assert expected["engine_control_variate_flag"] is False
    maker = MakeMCHullWhiteCapFloorEngine(_model(_curve()))
    assert not hasattr(maker, "with_control_variate")
    expected_names = {
        "withBrownianBridge": "with_brownian_bridge",
        "withSamples": "with_samples",
        "withAbsoluteTolerance": "with_absolute_tolerance",
        "withMaxSamples": "with_max_samples",
        "withSeed": "with_seed",
        "withAntitheticVariate": "with_antithetic_variate",
    }
    assert set(expected["named_parameters"]) == set(expected_names)
    for py_name in expected_names.values():
        assert callable(getattr(maker, py_name))
    engine = maker.with_samples(1023).with_seed(42).engine()
    assert getattr(engine, "_control_variate") is False  # structural pin on the C++ flag  # noqa: B009
