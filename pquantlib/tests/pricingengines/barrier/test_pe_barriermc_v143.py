"""Cross-validate the v1.43 lattice + MC single-barrier engine cluster against C++.

Reference: ``migration-harness/references/v143/pe/barriermc`` (probe at
``migration-harness/cpp/probes/v143_pe_barriermc/probe.cpp``).

Covers :class:`DiscretizedBarrierOption`,
:class:`DiscretizedDermanKaniBarrierOption`, :class:`BinomialBarrierEngine`,
:class:`BarrierPathPricer`, :class:`BiasedBarrierPathPricer`,
:class:`MCBarrierEngine` and :class:`MakeMCBarrierEngine`.

Why the assertions are exact
----------------------------
Nothing in this cluster consults the clock.  The lattice engines are
deterministic by construction, and the MC engines are deterministic once the
seed is fixed: ``PseudoRandom`` is
``InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
InverseCumulativeNormal>`` and ``LowDiscrepancy`` is
``InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>``.  Every MC case
below pins an explicit nonzero seed, so a correct port reproduces the C++ NPV
*and* the C++ error estimate and the tier is TIGHT (1e-14 abs / 1e-12 rel)
rather than a confidence band.  A band would pass with a missing
Brownian-bridge continuity correction, the wrong uniform stream for it, the
wrong antithetic pairing, or a knock-out rebate discounted at the wrong node —
which is the entire failure mode this file exists to catch.

The three layers are asserted separately on purpose: the discretized-asset
cases pin ``check_barrier`` and the Derman-Kani interpolation node by node, so
a binomial-engine failure localises to the barrier rule, the tree, or the
Boyle-Lau step correction rather than only showing a wrong NPV.

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
from pquantlib.exercise import AmericanExercise, EuropeanExercise, Exercise
from pquantlib.instruments.barrier_option import (
    BarrierOption,
    BarrierOptionArguments,
    BarrierType,
)
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_sequence_generator import (
    RandomSequenceGenerator,
)
from pquantlib.math.randomnumbers.rng_traits import LowDiscrepancy, PseudoRandom
from pquantlib.methods.lattices.binomial_tree import (
    AdditiveEQPBinomialTree,
    CoxRossRubinstein,
    JarrowRudd,
    Joshi4,
    LeisenReimer,
    Tian,
    Trigeorgis,
)
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice
from pquantlib.methods.montecarlo.path import Path
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import CashOrNothingPayoff, OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.binomial_barrier_engine import (
    BarrierDiscretization,
    BarrierTreeBuilder,
    BinomialBarrierEngine,
)
from pquantlib.pricingengines.barrier.discretized_barrier_option import (
    DiscretizedBarrierOption,
    DiscretizedDermanKaniBarrierOption,
)
from pquantlib.pricingengines.barrier.mc_barrier_engine import (
    BRIDGE_UNIFORM_SEED,
    BarrierPathPricer,
    BiasedBarrierPathPricer,
    MakeMCBarrierEngine,
    MCBarrierEngine,
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
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid

# probe.cpp — ``const Date kToday(15, May, 2025);`` and
# ``Settings::instance().evaluationDate() = kToday;`` in ``main()``
# (v143_pe_barriermc/probe.cpp, main()).
TODAY = Date.from_ymd(15, Month.May, 2025)

_DC = Actual365Fixed()
_CAL = NullCalendar()

#: probe.cpp — ``dispatchBinomialEngine`` maps these strings to the template
#: argument ``T``.
_TREES: dict[str, BarrierTreeBuilder] = {
    "JarrowRudd": JarrowRudd,
    "CoxRossRubinstein": CoxRossRubinstein,
    "AdditiveEQPBinomialTree": AdditiveEQPBinomialTree,
    "Trigeorgis": Trigeorgis,
    "Tian": Tian,
    "LeisenReimer": LeisenReimer,
    "Joshi4": Joshi4,
}

#: probe.cpp — the template argument ``D``.
_DISCRETIZATIONS: dict[str, BarrierDiscretization] = {
    "DiscretizedBarrierOption": DiscretizedBarrierOption,
    "DiscretizedDermanKaniBarrierOption": DiscretizedDermanKaniBarrierOption,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/barriermc")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


# ---------------------------------------------------------------------------
# Market / instrument reconstruction
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


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


def _traits(name: str) -> type[PseudoRandom] | type[LowDiscrepancy]:
    return PseudoRandom if name == "pseudo" else LowDiscrepancy


def _exercise(inputs: dict[str, Any]) -> Exercise:
    if inputs["exercise"] == "American":
        return AmericanExercise(TODAY, TODAY + inputs["expiry_days"])
    return EuropeanExercise(TODAY + inputs["expiry_days"])


def _barrier_option(inputs: dict[str, Any]) -> BarrierOption:
    return BarrierOption(
        BarrierType[inputs["barrier_type"]],
        inputs["barrier"],
        inputs["rebate"],
        PlainVanillaPayoff(_option_type(inputs["option_type"]), inputs["strike"]),
        _exercise(inputs),
    )


def _arguments(inputs: dict[str, Any]) -> BarrierOptionArguments:
    args = BarrierOptionArguments()
    _barrier_option(inputs).setup_arguments(args)
    args.validate()
    return args


def _array(expected: dict[str, Any], prefix: str) -> list[float]:
    """Rebuild an ``Obj::a``-encoded array (``<prefix>_n`` + ``<prefix>_i``)."""
    return [expected[f"{prefix}_{i}"] for i in range(expected[f"{prefix}_n"])]


def _check_array(actual: Any, expected: dict[str, Any], prefix: str) -> None:
    want = _array(expected, prefix)
    assert len(actual) == len(want), f"{prefix}: length {len(actual)} != {len(want)}"
    for i, value in enumerate(want):
        tolerance.tight(float(actual[i]), value, reason=f"{prefix}[{i}]")


def _names(cpp: dict[str, Any], prefix: str, key: str) -> list[str]:
    """Case names starting with ``prefix`` whose ``expected`` carries ``key``."""
    return sorted(k for k in cpp if k.startswith(prefix) and key in cpp[k]["expected"])


# ---------------------------------------------------------------------------
# 1. DiscretizedBarrierOption / DiscretizedDermanKaniBarrierOption
# ---------------------------------------------------------------------------


def _lattice_setup(
    inputs: dict[str, Any], args: BarrierOptionArguments
) -> tuple[GeneralizedBlackScholesProcess, BlackScholesLattice, TimeGrid, float]:
    """Rebuild the probe's ``makeSetup`` — the engine's own lattice construction.

    Mirrors ``BinomialBarrierEngine::calculate`` (binomialbarrierengine.hpp:99-164)
    without the Boyle-Lau branch, which the ``binom_boylelau_*`` cases pin
    separately.  probe.cpp ``makeSetup``.
    """
    process = _bsm(inputs)
    rfdc = process.risk_free_rate().day_counter()
    divdc = process.dividend_yield().day_counter()
    voldc = process.black_volatility().day_counter()
    volcal = process.black_volatility().calendar()

    s0 = process.state_variable().value()
    assert args.exercise is not None
    maturity_date = args.exercise.last_date()
    v = process.black_volatility().black_vol(maturity_date, s0)
    r = process.risk_free_rate().zero_rate(
        maturity_date,
        Compounding.Continuous,
        Frequency.NoFrequency,
        result_day_counter=rfdc,
    ).rate()
    q = process.dividend_yield().zero_rate(
        maturity_date,
        Compounding.Continuous,
        Frequency.NoFrequency,
        result_day_counter=divdc,
    ).rate()
    reference_date = process.risk_free_rate().reference_date()
    maturity = rfdc.year_fraction(reference_date, maturity_date)

    bs = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(s0),
        dividend_ts=FlatForward.from_rate(
            reference_date=reference_date, forward_rate=q, day_counter=divdc
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=reference_date, forward_rate=r, day_counter=rfdc
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=reference_date,
            calendar=volcal,
            day_counter=voldc,
            volatility=v,
        ),
    )
    steps = inputs["steps"]
    payoff = args.payoff
    assert isinstance(payoff, PlainVanillaPayoff)
    tree = CoxRossRubinstein(bs, maturity, steps, payoff.strike())
    lattice = BlackScholesLattice(tree, r, maturity, steps)
    return process, lattice, TimeGrid.regular(end=maturity, steps=steps), maturity


def test_discretized_barrier_option_internals(cpp: dict[str, Any]) -> None:
    """``check_barrier`` and the rollback, node by node, for both variants.

    ``plain_check_ones`` is the sharp one: ``check_barrier`` is applied to an
    all-ones synthetic array over the lattice grid at maturity, so which
    entries survive as 1.0 and which are overwritten (with the rebate, the
    payoff, or the nested vanilla value) is visible without a rollback in the
    way.  A port that gets the ``<=`` / ``>=`` direction, the knocked/live
    side, or the ``end_time`` gate wrong fails here rather than three layers up.
    """
    names = _names(cpp, "disc_", "plain_pv")
    assert names, "no discretized-asset cases in the reference"
    for name in names:
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        args = _arguments(inputs)
        steps = inputs["steps"]
        mid = steps // 2

        # --- plain DiscretizedBarrierOption ------------------------------
        process, lattice, grid, maturity = _lattice_setup(inputs, args)
        option = DiscretizedBarrierOption(args, process, grid)
        option.initialize(lattice, maturity)

        grid_values = lattice.grid(maturity)
        _check_array(grid_values, expected, "grid")
        _check_array(option.values, expected, "plain_init")
        _check_array(option.vanilla(), expected, "plain_vanilla_init")

        synthetic = np.ones(len(grid_values), dtype=np.float64)
        option.check_barrier(synthetic, grid_values)
        _check_array(synthetic, expected, "plain_check_ones")

        option.rollback(grid[mid])
        tolerance.tight(grid[mid], expected["plain_mid_t"])
        _check_array(option.values, expected, "plain_mid")
        _check_array(option.vanilla(), expected, "plain_mid_vanilla")

        option.rollback(grid[2])
        _check_array(option.values, expected, "plain_s2")
        option.rollback(grid[1])
        _check_array(option.values, expected, "plain_s1")
        option.rollback(0.0)
        tolerance.tight(option.present_value(), expected["plain_pv"])

        # --- DiscretizedDermanKaniBarrierOption --------------------------
        process, lattice, grid, maturity = _lattice_setup(inputs, args)
        dk = DiscretizedDermanKaniBarrierOption(args, process, grid)
        dk.initialize(lattice, maturity)
        _check_array(dk.values, expected, "dk_init")

        dk.rollback(grid[mid])
        _check_array(dk.values, expected, "dk_mid")
        dk.rollback(grid[2])
        _check_array(dk.values, expected, "dk_s2")
        dk.rollback(grid[1])
        _check_array(dk.values, expected, "dk_s1")
        dk.rollback(0.0)
        tolerance.tight(dk.present_value(), expected["dk_pv"])


def test_derman_kani_actually_differs_from_plain(cpp: dict[str, Any]) -> None:
    """The two discretizations must disagree, or ``D`` is decorative.

    The barrier is deliberately off-grid in every case, so the Derman-Kani
    interpolation has something to do.  Asserted on the C++ numbers first (the
    difference is a property of v1.43, not of the port) and then reproduced.
    """
    names = _names(cpp, "disc_", "plain_pv")
    assert names
    differing = [n for n in names if cpp[n]["expected"]["plain_pv"] != cpp[n]["expected"]["dk_pv"]]
    assert len(differing) == len(names), (
        f"cases where Derman-Kani made no difference: "
        f"{sorted(set(names) - set(differing))}"
    )


def test_discretized_empty_exercise_is_unreachable(cpp: dict[str, Any]) -> None:
    """C++ cannot build a ``BarrierOption`` with an empty exercise date list.

    The ``QL_REQUIRE(!args.exercise->dates().empty())`` guard in
    ``DiscretizedBarrierOption`` is therefore dead through the public
    instrument API, which the probe records rather than fakes.  The port keeps
    the guard anyway.
    """
    assert (
        cpp["disc_empty_exercise_unreachable"]["expected"][
            "reachable_through_public_api"
        ]
        is False
    )


# ---------------------------------------------------------------------------
# 2. BinomialBarrierEngine
# ---------------------------------------------------------------------------


def _build_binomial(inputs: dict[str, Any]) -> BarrierOption:
    option = _barrier_option(inputs)
    option.set_pricing_engine(
        BinomialBarrierEngine(
            _bsm(inputs),
            inputs["steps"],
            inputs["max_steps"],
            tree=_TREES[inputs["tree"]],
            discretization=_DISCRETIZATIONS[inputs["discretization"]],
        )
    )
    return option


def _check_binomial(option: BarrierOption, expected: dict[str, Any]) -> None:
    tolerance.tight(option.npv(), expected["npv"])
    tolerance.tight(option.delta(), expected["delta"])
    tolerance.tight(option.gamma(), expected["gamma"])
    tolerance.tight(option.theta(), expected["theta"])


def test_binomial_tree_sweep(cpp: dict[str, Any]) -> None:
    """All seven C++ tree builders, both discretizations, NPV + all three Greeks.

    ``theta`` is the discriminating one: this engine reports
    ``(p2m - p0) / grid[2]``, a finite difference off the middle node of the
    third-last slice, and *not* the Black-Scholes-PDE theta that
    ``BinomialVanillaEngine`` computes.
    """
    names = _names(cpp, "binom_sweep_", "npv")
    assert len(names) == 14, f"expected 7 trees x 2 discretizations, got {names}"
    for name in names:
        case = cpp[name]
        _check_binomial(_build_binomial(case["inputs"]), case["expected"])


def test_binomial_barrier_types(cpp: dict[str, Any]) -> None:
    """All four barrier types, with and without a rebate, both discretizations."""
    names = _names(cpp, "binom_type_", "npv")
    assert len(names) == 16, f"expected 4 types x 2 rebates x 2 disc, got {names}"
    for name in names:
        case = cpp[name]
        _check_binomial(_build_binomial(case["inputs"]), case["expected"])


def test_binomial_american_exercise(cpp: dict[str, Any]) -> None:
    """American exercise drives ``check_barrier``'s RANGE stopping-time branch."""
    names = _names(cpp, "binom_amer_", "npv")
    assert names
    for name in names:
        case = cpp[name]
        _check_binomial(_build_binomial(case["inputs"]), case["expected"])


def test_binomial_boyle_lau_step_correction(cpp: dict[str, Any]) -> None:
    """The Boyle-Lau promotion, its clamp, its off-switch, and its tree guard.

    With s0=100, B=95, vol=0.20, T=1 the divisor is ``log(100/95)**2`` and the
    truncated scan gives 15, 60, 136 for i = 1, 2, 3.  So 40 requested steps
    promote to 60 — proven here by *equality* with an explicit 60-step run, not
    by asserting the intermediate count:

    * ``binom_boylelau_crr_default`` (maxSteps=0 -> heuristic 1000) == the
      explicit 60-step run;
    * ``binom_boylelau_crr_clamped`` (maxSteps=50) differs from both;
    * ``binom_boylelau_crr_disabled`` (maxSteps == steps) differs again;
    * ``Trigeorgis`` is a *sibling* of ``CoxRossRubinstein`` under
      ``EqualJumpsBinomialTree``, not a subclass, so ``is_base_of`` is false and
      the promotion never fires — its default and disabled runs must be equal.
    """
    names = _names(cpp, "binom_boylelau_", "npv")
    assert names
    for name in names:
        case = cpp[name]
        _check_binomial(_build_binomial(case["inputs"]), case["expected"])

    expected = {n: cpp[n]["expected"]["npv"] for n in names}
    # Promotion 40 -> 60.
    assert (
        expected["binom_boylelau_crr_default"]
        == expected["binom_boylelau_crr_explicit60"]
    )
    # The clamp and the off-switch each land somewhere else.
    assert (
        expected["binom_boylelau_crr_clamped"]
        != expected["binom_boylelau_crr_default"]
    )
    assert (
        expected["binom_boylelau_crr_disabled"]
        != expected["binom_boylelau_crr_clamped"]
    )
    # Trigeorgis never promotes.
    assert (
        expected["binom_boylelau_trigeorgis_default"]
        == expected["binom_boylelau_trigeorgis_disabled"]
    )


def test_binomial_engine_guards(cpp: dict[str, Any]) -> None:
    """The four ``QL_REQUIRE``s of ``BinomialBarrierEngine``."""
    process = _bsm(cpp["binom_sweep_CoxRossRubinstein_plain"]["inputs"])

    assert cpp["binom_engine_zero_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="timeSteps must be positive"):
        BinomialBarrierEngine(
            process,
            0,
            0,
            tree=CoxRossRubinstein,
            discretization=DiscretizedBarrierOption,
        )

    assert cpp["binom_engine_maxsteps_below_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="maxTimeSteps must be zero or"):
        BinomialBarrierEngine(
            process,
            40,
            39,
            tree=CoxRossRubinstein,
            discretization=DiscretizedBarrierOption,
        )

    # Positive control: the probe pins this one as NOT throwing.
    assert not cpp["binom_engine_maxsteps_equal_ok"]["expected"]["throws"]
    BinomialBarrierEngine(
        process, 40, 40, tree=CoxRossRubinstein, discretization=DiscretizedBarrierOption
    )

    assert cpp["binom_engine_barrier_touched"]["expected"]["throws"]
    touched = BarrierOption(
        BarrierType.DownOut,
        105.0,
        0.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(TODAY + 365),
    )
    touched.set_pricing_engine(
        BinomialBarrierEngine(
            process,
            41,
            41,
            tree=CoxRossRubinstein,
            discretization=DiscretizedBarrierOption,
        )
    )
    with pytest.raises(LibraryException, match="barrier touched"):
        touched.npv()

    assert cpp["binom_engine_zero_strike"]["expected"]["throws"]
    zero_strike = BarrierOption(
        BarrierType.DownOut,
        95.0,
        0.0,
        PlainVanillaPayoff(OptionType.Call, 0.0),
        EuropeanExercise(TODAY + 365),
    )
    zero_strike.set_pricing_engine(
        BinomialBarrierEngine(
            process,
            41,
            41,
            tree=CoxRossRubinstein,
            discretization=DiscretizedBarrierOption,
        )
    )
    with pytest.raises(LibraryException, match="strike must be positive"):
        zero_strike.npv()


# ---------------------------------------------------------------------------
# 3. BarrierPathPricer / BiasedBarrierPathPricer
# ---------------------------------------------------------------------------


def _paths(inputs: dict[str, Any]) -> tuple[TimeGrid, list[Path]]:
    grid = TimeGrid.regular(end=1.0, steps=inputs["steps"])
    n = inputs["steps"] + 1
    for k in range(n):
        tolerance.tight(grid[k], inputs[f"grid_t{k}"])
    paths = [
        Path(grid, np.array([inputs[f"path{p}_{k}"] for k in range(n)], dtype=np.float64))
        for p in range(inputs["n_paths"])
    ]
    return grid, paths


def _bridge_generator(grid: TimeGrid) -> RandomSequenceGenerator[MersenneTwisterUniformRng]:
    """# C++ parity: ``PseudoRandom::ursg_type(grid.size()-1,
    # PseudoRandom::urng_type(5))`` (mcbarrierengine.hpp:256)."""
    return RandomSequenceGenerator(len(grid) - 1, MersenneTwisterUniformRng(5))


def test_path_pricers(cpp: dict[str, Any]) -> None:
    """Both path pricers on explicit paths that genuinely knock.

    Four paths per case: one that never crosses, one that crosses mid-path, one
    that crosses on the last node, and a **near miss** whose every node stays
    on the live side of the barrier — the case the biased pricer structurally
    cannot see, and which ends in the money so the two pricers cannot agree by
    accident on a zero payoff.

    Three things are pinned per case:

    * ``unbiased_p*`` — all four paths through ONE pricer instance, so the
      seed-5 uniform generator's *statefulness* across paths is pinned;
    * ``unbiased_fresh_p*`` — a fresh pricer per path, i.e. every path sees the
      first uniform block.  Where these disagree with ``unbiased_p*``, the
      bridge draw is genuinely load-bearing;
    * ``biased_p*`` — the same paths with no bridge at all.
    """
    names = _names(cpp, "pathpricer_", "unbiased_p0")
    assert names
    for name in names:
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        process = _bsm(inputs)
        grid, paths = _paths(inputs)
        discounts = [
            process.risk_free_rate().discount(grid[i]) for i in range(len(grid))
        ]
        for i, discount in enumerate(discounts):
            tolerance.tight(discount, inputs[f"discount_{i}"])

        barrier_type = BarrierType[inputs["barrier_type"]]
        option_type = _option_type(inputs["option_type"])

        # One instance, three calls in order.
        pricer = BarrierPathPricer(
            barrier_type,
            inputs["barrier"],
            inputs["rebate"],
            option_type,
            inputs["strike"],
            discounts,
            process,
            _bridge_generator(grid),
        )
        for k, path in enumerate(paths):
            tolerance.tight(pricer(path), expected[f"unbiased_p{k}"], reason=name)

        # A fresh instance per path.
        for k, path in enumerate(paths):
            fresh = BarrierPathPricer(
                barrier_type,
                inputs["barrier"],
                inputs["rebate"],
                option_type,
                inputs["strike"],
                discounts,
                process,
                _bridge_generator(grid),
            )
            tolerance.tight(
                fresh(path), expected[f"unbiased_fresh_p{k}"], reason=name
            )

        biased = BiasedBarrierPathPricer(
            barrier_type,
            inputs["barrier"],
            inputs["rebate"],
            option_type,
            inputs["strike"],
            discounts,
        )
        for k, path in enumerate(paths):
            tolerance.tight(biased(path), expected[f"biased_p{k}"], reason=name)


def test_bridge_correction_actually_changes_something(cpp: dict[str, Any]) -> None:
    """The near-miss path must price differently biased vs unbiased, every case.

    Asserted on the C++ numbers, so it is a statement about v1.43 rather than
    about the port: if a case ever stopped exercising the continuity
    correction, the corresponding port test would silently become vacuous.
    Path 3 is the near miss, so it is the one that must differ; a port with no
    bridge at all reproduces every other path and fails only here.
    """
    names = _names(cpp, "pathpricer_", "unbiased_p0")
    assert names
    for name in names:
        expected = cpp[name]["expected"]
        near_miss = cpp[name]["inputs"]["n_paths"] - 1
        assert (
            expected[f"unbiased_p{near_miss}"] != expected[f"biased_p{near_miss}"]
        ), f"{name}: the bridge correction never fired on the near-miss path"


def test_bridge_uniform_seed_is_pinned(cpp: dict[str, Any]) -> None:
    """The bridge generator's seed is a constant of the algorithm, not a knob."""
    assert cpp["pathpricer_downout"]["inputs"]["bridge_uniform_seed"] == (
        BRIDGE_UNIFORM_SEED
    )
    assert cpp["mc_ps_downout_unbiased"]["inputs"]["bridge_uniform_seed"] == (
        BRIDGE_UNIFORM_SEED
    )


def test_path_pricer_guards(cpp: dict[str, Any]) -> None:
    """The constructor ``QL_REQUIRE``s plus the empty-path guard."""
    inputs = cpp["pathpricer_downout"]["inputs"]
    process = _bsm(inputs)
    grid, _ = _paths(inputs)
    discounts = [process.risk_free_rate().discount(grid[i]) for i in range(len(grid))]

    assert cpp["pathpricer_negative_strike"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="strike less than zero"):
        BarrierPathPricer(
            BarrierType.DownOut,
            90.0,
            0.0,
            OptionType.Call,
            -1.0,
            discounts,
            process,
            _bridge_generator(grid),
        )

    assert cpp["pathpricer_zero_barrier"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="barrier less/equal zero"):
        BarrierPathPricer(
            BarrierType.DownOut,
            0.0,
            0.0,
            OptionType.Call,
            100.0,
            discounts,
            process,
            _bridge_generator(grid),
        )

    assert cpp["biased_pathpricer_negative_strike"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="strike less than zero"):
        BiasedBarrierPathPricer(
            BarrierType.DownOut, 90.0, 0.0, OptionType.Call, -1.0, discounts
        )

    assert cpp["biased_pathpricer_zero_barrier"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="barrier less/equal zero"):
        BiasedBarrierPathPricer(
            BarrierType.DownOut, 0.0, 0.0, OptionType.Call, 100.0, discounts
        )

    assert cpp["pathpricer_single_point_path"]["expected"]["throws"]
    single = Path(TimeGrid.with_mandatory([0.0]), np.array([100.0], dtype=np.float64))
    pricer = BiasedBarrierPathPricer(
        BarrierType.DownOut, 90.0, 0.0, OptionType.Call, 100.0, discounts
    )
    with pytest.raises(LibraryException, match="the path cannot be empty"):
        pricer(single)


# ---------------------------------------------------------------------------
# 4. MCBarrierEngine / MakeMCBarrierEngine
# ---------------------------------------------------------------------------


def _build_mc(inputs: dict[str, Any]) -> BarrierOption:
    builder = MakeMCBarrierEngine(_bsm(inputs), _traits(inputs["rng"]))
    if inputs["steps"] > 0:
        builder.with_steps(inputs["steps"])
    if inputs["steps_per_year"] > 0:
        builder.with_steps_per_year(inputs["steps_per_year"])
    builder.with_brownian_bridge(inputs["brownian_bridge"]).with_antithetic_variate(
        inputs["antithetic"]
    ).with_bias(inputs["biased"]).with_seed(inputs["seed"])
    if inputs["samples"] > 0:
        builder.with_samples(inputs["samples"])
    else:
        builder.with_absolute_tolerance(inputs["tolerance"])
    if inputs["max_samples"] > 0:
        builder.with_max_samples(inputs["max_samples"])

    option = _barrier_option(inputs)
    option.set_pricing_engine(builder.engine())
    return option


def _check_mc(option: BarrierOption, expected: dict[str, Any]) -> None:
    tolerance.tight(option.npv(), expected["npv"])
    if expected["has_error_estimate"]:
        tolerance.tight(option.error_estimate(), expected["error_estimate"])
    else:
        # C++ ``if constexpr (RNG::allowsErrorEstimate)`` leaves
        # ``results_.errorEstimate`` at Null for LowDiscrepancy, and
        # ``Instrument::errorEstimate()`` throws. Pin the absence.
        with pytest.raises(LibraryException):
            option.error_estimate()


def test_mc_barrier_cases(cpp: dict[str, Any]) -> None:
    """Every priced ``MCBarrierEngine`` case, exactly.

    The 8-steps-over-a-year cases are the demanding ones: on that grid the
    Brownian-bridge continuity correction dominates rather than merely refines,
    so a port that drops it (or draws its uniforms from the engine's own
    generator instead of the hard-coded seed-5 one) misses these by a wide
    margin while still looking plausible on the 52-step case.
    """
    names = _names(cpp, "mc_ps_", "npv") + _names(cpp, "mc_ld_", "npv")
    assert names
    for name in sorted(names):
        case = cpp[name]
        _check_mc(_build_mc(case["inputs"]), case["expected"])


def test_mc_biased_and_unbiased_disagree(cpp: dict[str, Any]) -> None:
    """``isBiased_`` selects a different pricer, on all four barrier types."""
    for kind in ("downout", "downin", "upout", "upin"):
        unbiased = cpp[f"mc_ps_{kind}_unbiased"]["expected"]["npv"]
        biased = cpp[f"mc_ps_{kind}_biased"]["expected"]["npv"]
        assert unbiased != biased, f"{kind}: bias flag made no difference"


def test_mc_seed_actually_reaches_the_generator(cpp: dict[str, Any]) -> None:
    """Two seeds, one otherwise identical setup, two different answers.

    A port that swallows the seed (or substitutes its own, the way
    ``MCDoubleBarrierEngine`` does for seed 0) passes the single-seed test and
    fails this one.
    """
    a = cpp["mc_ps_downout_unbiased"]["expected"]["npv"]
    b = cpp["mc_ps_downout_unbiased_seed7"]["expected"]["npv"]
    assert a != b
    tolerance.tight(_build_mc(cpp["mc_ps_downout_unbiased"]["inputs"]).npv(), a)
    tolerance.tight(_build_mc(cpp["mc_ps_downout_unbiased_seed7"]["inputs"]).npv(), b)


def test_mc_make_defaults_match_the_explicit_twin(cpp: dict[str, Any]) -> None:
    """``brownianBridge_ = antithetic_ = biased_ = false`` by default.

    ``mc_make_defaults`` leaves every knob alone; ``mc_ps_downout_unbiased``
    sets them all explicitly to the same values.  Equal answers prove the
    defaults, and a mismatch localises to whichever default drifted.
    """
    expected = cpp["mc_make_defaults"]["expected"]
    assert expected["npv"] == cpp["mc_ps_downout_unbiased"]["expected"]["npv"]
    inputs = cpp["mc_make_defaults"]["inputs"]
    option = _barrier_option(inputs)
    option.set_pricing_engine(
        MakeMCBarrierEngine(_bsm(inputs))
        .with_steps(inputs["steps"])
        .with_samples(inputs["samples"])
        .with_seed(inputs["seed"])
        .engine()
    )
    _check_mc(option, expected)


def test_mc_make_bias_default_argument_is_true(cpp: dict[str, Any]) -> None:
    """``with_bias()`` with no argument selects the biased pricer."""
    case = cpp["mc_make_bias_default_is_true"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["npv_default_arg"] == expected["npv_explicit_true"]
    # And it must be the biased answer, not the unbiased one.
    assert expected["npv_default_arg"] == cpp["mc_ps_downout_biased"]["expected"]["npv"]

    def price(*, explicit: bool) -> float:
        builder = MakeMCBarrierEngine(_bsm(inputs))
        builder.with_steps(inputs["steps"]).with_samples(
            inputs["samples"]
        ).with_seed(inputs["seed"])
        builder.with_bias(True) if explicit else builder.with_bias()
        option = _barrier_option(inputs)
        option.set_pricing_engine(builder.engine())
        return option.npv()

    tolerance.tight(price(explicit=False), expected["npv_default_arg"])
    tolerance.tight(price(explicit=True), expected["npv_explicit_true"])


def test_mc_builder_validation(cpp: dict[str, Any]) -> None:
    """``MakeMCBarrierEngine``'s guards, each pinned by the probe."""
    process = _bsm(cpp["mc_ps_downout_unbiased"]["inputs"])

    assert cpp["mc_make_no_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps not given"):
        MakeMCBarrierEngine(process).with_samples(1023).engine()

    assert cpp["mc_make_both_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of steps overspecified"):
        MakeMCBarrierEngine(process).with_steps(4).with_steps_per_year(
            12
        ).with_samples(1023).engine()

    assert cpp["mc_make_samples_after_tolerance"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="tolerance already set"):
        MakeMCBarrierEngine(process).with_absolute_tolerance(0.02).with_samples(1023)

    assert cpp["mc_make_tolerance_after_samples"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="number of samples already set"):
        MakeMCBarrierEngine(process).with_samples(1023).with_absolute_tolerance(0.02)

    assert cpp["mc_make_tolerance_lowdiscrepancy"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="does not allow an error estimate"):
        MakeMCBarrierEngine(process, LowDiscrepancy).with_absolute_tolerance(0.02)

    # Positive control: the probe pins this one as NOT throwing.
    assert not cpp["mc_make_steps_only_ok"]["expected"]["throws"]
    MakeMCBarrierEngine(process).with_steps(4).with_samples(1023).engine()


def test_mc_engine_guards(cpp: dict[str, Any]) -> None:
    """The ``MCBarrierEngine`` / ``McSimulation`` / path-pricer guards."""
    inputs = cpp["mc_ps_downout_unbiased"]["inputs"]
    process = _bsm(inputs)

    assert cpp["mc_engine_zero_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="timeSteps must be positive"):
        MCBarrierEngine(process, time_steps=0, required_samples=1023, seed=42)

    assert cpp["mc_engine_zero_steps_per_year"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="timeStepsPerYear must be positive"):
        MCBarrierEngine(process, time_steps_per_year=0, required_samples=1023, seed=42)

    assert cpp["mc_engine_no_steps_at_all"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="no time steps provided"):
        MCBarrierEngine(process, required_samples=1023, seed=42)

    assert cpp["mc_engine_both_steps"]["expected"]["throws"]
    with pytest.raises(LibraryException, match="both time steps and time steps"):
        MCBarrierEngine(
            process,
            time_steps=4,
            time_steps_per_year=12,
            required_samples=1023,
            seed=42,
        )

    assert cpp["mc_engine_no_samples_no_tolerance"]["expected"]["throws"]
    option = BarrierOption(
        BarrierType.DownOut,
        90.0,
        0.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(TODAY + 365),
    )
    option.set_pricing_engine(MCBarrierEngine(process, time_steps=8, seed=42))
    with pytest.raises(LibraryException, match="neither tolerance nor number"):
        option.npv()

    assert cpp["mc_engine_barrier_touched"]["expected"]["throws"]
    touched = BarrierOption(
        BarrierType.DownOut,
        105.0,
        0.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(TODAY + 365),
    )
    touched.set_pricing_engine(
        MakeMCBarrierEngine(process)
        .with_steps(8)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="barrier touched"):
        touched.npv()

    assert cpp["mc_engine_non_plain_payoff"]["expected"]["throws"]
    binary = BarrierOption(
        BarrierType.DownOut,
        90.0,
        0.0,
        CashOrNothingPayoff(OptionType.Call, 100.0, 10.0),
        EuropeanExercise(TODAY + 365),
    )
    binary.set_pricing_engine(
        MakeMCBarrierEngine(process)
        .with_steps(8)
        .with_samples(1023)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="non-plain payoff given"):
        binary.npv()


def test_mc_tolerance_max_samples_exceeded(cpp: dict[str, Any]) -> None:
    """``QL_REQUIRE(sampleNumber < maxSamples)`` inside the tolerance loop."""
    assert cpp["mc_tol_maxsamples_exceeded"]["expected"]["throws"]
    process = _bsm(cpp["mc_ps_downout_unbiased"]["inputs"])
    option = BarrierOption(
        BarrierType.DownOut,
        90.0,
        0.0,
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(TODAY + 365),
    )
    option.set_pricing_engine(
        MakeMCBarrierEngine(process)
        .with_steps(8)
        .with_absolute_tolerance(1e-4)
        .with_max_samples(2000)
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match="max number of samples"):
        option.npv()
