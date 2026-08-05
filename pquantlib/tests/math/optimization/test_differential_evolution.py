"""Cross-validate ``DifferentialEvolution`` against C++ v1.43.

Reference: ``migration-harness/references/v143/math/optimization.json``,
block E. All 7 mutation strategies x 3 crossover types, plus a randomly
initialised population, an adaptive-crossover run and a bounds-disabled run.

Differential evolution is stochastic, and reproducing it is entirely a
question of consuming QuantLib's Mersenne twister in exactly the same order:
the Fisher-Yates shuffles (``next_int32() % (i+1)``, running i = n-1 down to
1), the jitter/dither draws, the per-component crossover mask, the
out-of-bounds mirror draw and the Brest self-adaptation draws all pull from
one stream. The function-evaluation count therefore doubles as a checksum on
the whole stream: 20 members x 30 generations = 600 evaluations, and it
matches for every one of the 24 strategy/crossover combinations.

``EndCriteria.Type`` and the evaluation counters are asserted exactly; the
best-ever point and its cost at TIGHT, because the probe's objective is
compiled with FP contraction and 30 generations of arithmetic carry the
resulting last-bit difference forward.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.constraint import BoundaryConstraint, NoConstraint
from pquantlib.math.optimization.differential_evolution import (
    Candidate,
    Configuration,
    CrossoverType,
    DifferentialEvolution,
    Strategy,
)
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.testing import reference_reader, tolerance
from tests.math.optimization._cpp_cost_functions import (
    RosenbrockAnalytic,
    WeightedQuadratic,
)

_STRATEGIES: dict[str, Strategy] = {
    "rand1standard": Strategy.Rand1Standard,
    "bestjitter": Strategy.BestMemberWithJitter,
    "currtobest2": Strategy.CurrentToBest2Diffs,
    "pervectordither": Strategy.Rand1DiffWithPerVectorDither,
    "dither": Strategy.Rand1DiffWithDither,
    "eitheror": Strategy.EitherOrWithOptimalRecombination,
    "selfadaptrot": Strategy.Rand1SelfadaptiveWithRotation,
}
_CROSSOVERS: dict[str, CrossoverType] = {
    "normal": CrossoverType.Normal,
    "binomial": CrossoverType.Binomial,
    "exponential": CrossoverType.Exponential,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def _fixed_population(cpp: dict[str, Any]) -> list[npt.NDArray[np.float64]]:
    flat: list[float] = cpp["de_fixed_population"]
    members: int = cpp["de_fixed_population_members"]
    dim: int = cpp["de_fixed_population_dim"]
    return [
        np.array(flat[dim * i : dim * i + dim], dtype=np.float64)
        for i in range(members)
    ]


def _grid_config(
    cpp: dict[str, Any], strategy: Strategy, crossover: CrossoverType
) -> Configuration:
    return (
        Configuration()
        .with_strategy(strategy)
        .with_crossover_type(crossover)
        .with_seed(42)
        .with_stepsize_weight(0.4)
        .with_crossover_probability(0.9)
        .with_bounds(True)
        .with_lower_bound(np.array([-5.0, -5.0]))
        .with_upper_bound(np.array([5.0, 5.0]))
        .with_initial_population(_fixed_population(cpp))
    )


def _run(name: str, cpp: dict[str, Any]) -> tuple[Problem, int]:
    parts = name.split("_")
    if parts[1] in _STRATEGIES:
        problem = Problem(
            RosenbrockAnalytic(), NoConstraint(), np.array([-1.2, 1.0])
        )
        conf = _grid_config(cpp, _STRATEGIES[parts[1]], _CROSSOVERS[parts[2]])
        ec = EndCriteria(30, 20, 1e-12, 1e-12, 1e-12)
    elif name == "de_randominit_boundaryconstraint":
        problem = Problem(
            RosenbrockAnalytic(), BoundaryConstraint(-5.0, 5.0), np.array([-1.2, 1.0])
        )
        conf = (
            Configuration()
            .with_strategy(Strategy.BestMemberWithJitter)
            .with_crossover_type(CrossoverType.Normal)
            .with_seed(7)
            .with_population_members(16)
            .with_stepsize_weight(0.2)
            .with_crossover_probability(0.9)
            .with_bounds(True)
        )
        ec = EndCriteria(25, 20, 1e-12, 1e-12, 1e-12)
    elif name == "de_adaptive_crossover":
        problem = Problem(
            WeightedQuadratic(np.array([1.5, -0.5]), np.array([1.0, 3.0])),
            BoundaryConstraint(-5.0, 5.0),
            np.array([0.0, 0.0]),
        )
        conf = (
            Configuration()
            .with_strategy(Strategy.Rand1Standard)
            .with_crossover_type(CrossoverType.Binomial)
            .with_seed(11)
            .with_stepsize_weight(0.5)
            .with_crossover_probability(0.8)
            .with_adaptive_crossover(True)
            .with_bounds(True)
            .with_initial_population(_fixed_population(cpp))
        )
        ec = EndCriteria(25, 20, 1e-12, 1e-12, 1e-12)
    elif name == "de_no_bounds":
        problem = Problem(
            WeightedQuadratic(np.array([1.5, -0.5]), np.array([1.0, 3.0])),
            NoConstraint(),
            np.array([0.0, 0.0]),
        )
        conf = (
            Configuration()
            .with_strategy(Strategy.Rand1Standard)
            .with_crossover_type(CrossoverType.Normal)
            .with_seed(13)
            .with_stepsize_weight(0.5)
            .with_crossover_probability(0.9)
            .with_bounds(False)
            .with_lower_bound(np.array([-5.0, -5.0]))
            .with_upper_bound(np.array([5.0, 5.0]))
            .with_initial_population(_fixed_population(cpp))
        )
        ec = EndCriteria(25, 20, 1e-12, 1e-12, 1e-12)
    else:
        raise KeyError(name)
    return problem, int(DifferentialEvolution(conf).minimize(problem, ec))


def test_case_list_covers_every_strategy_and_crossover(cpp: dict[str, Any]) -> None:
    names: list[str] = cpp["block_e_cases"]
    assert len(names) == 7 * 3 + 3


@pytest.mark.parametrize(
    "name",
    reference_reader.load("v143/math/optimization")["block_e_cases"],
)
def test_differential_evolution_matches_cpp(name: str, cpp: dict[str, Any]) -> None:
    problem, ec_type = _run(name, cpp)

    # The evaluation count is a checksum on the entire RNG stream.
    assert problem.function_evaluation == cpp[f"{name}_nfev"]
    assert problem.gradient_evaluation == cpp[f"{name}_ngev"] == 0
    assert ec_type == cpp[f"{name}_ec"], f"{name}: C++ says {cpp[f'{name}_ec_name']}"

    for got, expected in zip(problem.current_value, cpp[f"{name}_x"], strict=True):
        tolerance.tight(float(got), float(expected))
    tolerance.tight(problem.function_value, float(cpp[f"{name}_f"]))


def test_differential_evolution_trajectory_matches_cpp(cpp: dict[str, Any]) -> None:
    """Best-ever point after k = 3..12 generations, bit-exact.

    With the fixed initial population the generation loop is deterministic
    from the first draw, so the truncated runs pin the mutant construction and
    the crossover mask generation-by-generation.
    """
    ks: list[int] = cpp["de_traj_k"]
    dim: int = cpp["de_traj_n"]
    for step, k in enumerate(ks):
        problem = Problem(
            RosenbrockAnalytic(), NoConstraint(), np.array([-1.2, 1.0])
        )
        conf = _grid_config(cpp, Strategy.Rand1Standard, CrossoverType.Normal)
        ec_type = DifferentialEvolution(conf).minimize(
            problem, EndCriteria(int(k), 2, 1e-12, 1e-12, 1e-12)
        )
        for i in range(dim):
            tolerance.exact(
                float(problem.current_value[i]),
                float(cpp["de_traj_x"][dim * step + i]),
            )
        # TIGHT for f, EXACT for x: the probe's Rosenbrock ``value`` is
        # compiled with FP contraction (``x[i+1] - x[i]*x[i]`` -> fma), so the
        # reported objective can differ by an ulp even when the selected
        # candidate is bit-identical.
        tolerance.tight(problem.function_value, float(cpp["de_traj_f"][step]))
        assert int(ec_type) == cpp["de_traj_ec"][step]
        assert problem.function_evaluation == cpp["de_traj_nfev"][step]


# --- Configuration / Candidate behaviour ------------------------------------


def test_configuration_defaults_mirror_cpp() -> None:
    """# C++ parity: differentialevolution.hpp:84-91."""
    conf = Configuration()
    assert conf.strategy is Strategy.BestMemberWithJitter
    assert conf.crossover_type is CrossoverType.Normal
    assert conf.population_members == 100
    assert conf.stepsize_weight == 0.2
    assert conf.crossover_probability == 0.9
    assert conf.seed == 0
    assert conf.apply_bounds is True
    assert conf.crossover_is_adaptive is False
    assert conf.initial_population == []
    assert conf.upper_bound.size == 0
    assert conf.lower_bound.size == 0


def test_with_population_members_clears_the_initial_population() -> None:
    """# C++ parity: differentialevolution.hpp:110-115 — the clear is deliberate."""
    conf = Configuration().with_initial_population(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    )
    assert conf.population_members == 2
    conf.with_population_members(7)
    assert conf.population_members == 7
    assert conf.initial_population == []


def test_with_initial_population_overwrites_the_member_count() -> None:
    conf = Configuration().with_population_members(50)
    conf.with_initial_population([np.array([1.0]), np.array([2.0]), np.array([3.0])])
    assert conf.population_members == 3


@pytest.mark.parametrize("p", [-0.01, 1.01])
def test_crossover_probability_is_validated(p: float) -> None:
    with pytest.raises(LibraryException, match=r"must be in \[0,1\] range"):
        Configuration().with_crossover_probability(p)


@pytest.mark.parametrize("w", [-0.01, 2.01])
def test_stepsize_weight_is_validated(w: float) -> None:
    with pytest.raises(LibraryException, match=r"must be in \[0,2\] range"):
        Configuration().with_stepsize_weight(w)


def test_population_members_must_be_positive() -> None:
    with pytest.raises(LibraryException, match="Positive number of population members"):
        Configuration().with_population_members(0)


def test_configuration_is_copied_into_the_optimizer() -> None:
    """C++ stores the Configuration by value; later mutation must not leak in."""
    conf = Configuration().with_seed(5).with_stepsize_weight(0.3)
    de = DifferentialEvolution(conf)
    conf.stepsize_weight = 1.7
    assert de.configuration.stepsize_weight == 0.3


def test_an_explicit_seed_makes_the_run_reproducible() -> None:
    """The property the cross-validation depends on.

    ``Configuration.seed`` defaults to 0, which — like C++ — routes
    ``MersenneTwisterUniformRng`` through the clock-based ``SeedGenerator``
    and is therefore NOT reproducible. An explicit non-zero seed is.
    """
    conf = Configuration().with_seed(4242).with_population_members(8)
    ec = EndCriteria(5, 2, 1e-8, 1e-8, 1e-8)
    values: list[npt.NDArray[np.float64]] = []
    for _ in range(2):
        problem = Problem(
            RosenbrockAnalytic(), BoundaryConstraint(-5.0, 5.0), np.array([-1.2, 1.0])
        )
        DifferentialEvolution(conf).minimize(problem, ec)
        values.append(problem.current_value.copy())
    for a, b in zip(values[0], values[1], strict=True):
        tolerance.exact(float(a), float(b))

def test_candidate_defaults() -> None:
    """# C++ parity: differentialevolution.hpp:76-80."""
    c = Candidate.of_size(3)
    assert c.values.size == 3
    assert float(np.max(np.abs(c.values))) == 0.0
    assert c.cost == 0.0
    assert DifferentialEvolution.Candidate is Candidate
    assert DifferentialEvolution.Configuration is Configuration


def test_candidate_copy_is_deep() -> None:
    c = Candidate(values=np.array([1.0, 2.0]), cost=3.0)
    d = c.copy()
    d.values[0] = 9.0
    assert c.values[0] == 1.0


def test_wrong_bound_size_is_rejected() -> None:
    problem = Problem(RosenbrockAnalytic(), NoConstraint(), np.array([-1.2, 1.0]))
    conf = (
        Configuration()
        .with_seed(3)
        .with_upper_bound(np.array([1.0, 2.0, 3.0]))
        .with_population_members(4)
    )
    with pytest.raises(LibraryException, match="wrong upper bound size"):
        DifferentialEvolution(conf).minimize(
            problem, EndCriteria(5, 2, 1e-8, 1e-8, 1e-8)
        )


def test_partial_sort_front_swaps_as_it_scans() -> None:
    """``std::partial_sort(b, b+1, e)`` leaves the tail in swap order.

    ``[5, 3, 1]`` becomes ``[1, 5, 3]``, not ``[1, 3, 5]``. The distinction is
    observable because the next generation shuffles this very list with a
    seeded Fisher-Yates.
    """
    pop = [
        Candidate(values=np.array([float(i)]), cost=c)
        for i, c in enumerate([5.0, 3.0, 1.0])
    ]
    DifferentialEvolution.partial_sort_front(pop)
    assert [c.cost for c in pop] == [1.0, 5.0, 3.0]


def test_differential_evolution_is_an_optimization_method() -> None:
    assert isinstance(
        DifferentialEvolution(Configuration().with_seed(1)), OptimizationMethod
    )
