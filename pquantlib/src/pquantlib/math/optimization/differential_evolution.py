"""Differential Evolution global optimizer.

# C++ parity: ql/math/optimization/differentialevolution.{hpp,cpp} (v1.43).

Price & Storn (1997), *Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces*, Journal of
Global Optimization 11, 341-359. Seven mutation strategies and three
crossover types.

The method is stochastic, but every draw comes from QuantLib's own
``MersenneTwisterUniformRng`` seeded from ``Configuration.seed``, so a
faithful port is exactly reproducible. Three things must line up for
that to hold, and each is a place a re-implementation drifts:

1. **The number and ORDER of RNG draws.** ``randomize`` is a
   Fisher-Yates shuffle running ``i = n-1 .. 1`` and drawing
   ``next_int32() % (i+1)``; each strategy calls it a fixed number of
   times before touching ``next_real()``. Getting the count right but
   the order wrong desynchronises the whole stream.
2. **The bounds mirror.** When ``apply_bounds`` is set, an out-of-range
   coordinate is not clipped: it is reflected towards the corresponding
   member of the "mirror population" by a fresh uniform draw
   (differentialevolution.cpp:282-295) — another RNG consumer.
3. **The cost bookkeeping.** The initial population is priced through
   ``Problem.cost_function.value`` (which does NOT move the evaluation
   counter) while every later generation is priced through
   ``Problem.value`` (which does). A non-finite cost, or a raised
   ``LibraryException``, becomes ``QL_MAX_REAL``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem


class Strategy(IntEnum):
    """Mutant-population construction strategy.

    # C++ parity: ``DifferentialEvolution::Strategy`` in
    # ql/math/optimization/differentialevolution.hpp:61-69 (v1.43).
    """

    Rand1Standard = 0
    BestMemberWithJitter = 1
    CurrentToBest2Diffs = 2
    Rand1DiffWithPerVectorDither = 3
    Rand1DiffWithDither = 4
    EitherOrWithOptimalRecombination = 5
    Rand1SelfadaptiveWithRotation = 6


class CrossoverType(IntEnum):
    """Crossover-probability transform.

    # C++ parity: ``DifferentialEvolution::CrossoverType`` in
    # ql/math/optimization/differentialevolution.hpp:70-74 (v1.43).
    """

    Normal = 0
    Binomial = 1
    Exponential = 2


@dataclass(slots=True)
class Candidate:
    """One population member: a parameter vector and its cost.

    # C++ parity: ``DifferentialEvolution::Candidate`` in
    # ql/math/optimization/differentialevolution.hpp:76-80 (v1.43).
    """

    values: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    cost: float = 0.0

    @staticmethod
    def of_size(size: int) -> Candidate:
        """Zero-filled candidate of the given dimension.

        # C++ parity: ``Candidate(Size size = 0) : values(size, 0.0) {}``.
        """
        return Candidate(values=np.zeros(size, dtype=np.float64), cost=0.0)

    def copy(self) -> Candidate:
        """Deep copy — C++ copies ``Candidate`` by value throughout."""
        return Candidate(values=self.values.astype(np.float64, copy=True), cost=self.cost)


@dataclass(slots=True)
class Configuration:
    """Tuning knobs for :class:`DifferentialEvolution`.

    # C++ parity: ``DifferentialEvolution::Configuration`` in
    # ql/math/optimization/differentialevolution.hpp:82-160 (v1.43).

    The ``with_*`` methods mirror the C++ fluent setters, including their
    validation and the side effect that ``with_population_members``
    CLEARS any initial population while ``with_initial_population``
    overwrites ``population_members``. They mutate and return ``self``,
    matching the C++ ``Configuration&`` returns.
    """

    strategy: Strategy = Strategy.BestMemberWithJitter
    crossover_type: CrossoverType = CrossoverType.Normal
    population_members: int = 100
    stepsize_weight: float = 0.2
    crossover_probability: float = 0.9
    seed: int = 0
    apply_bounds: bool = True
    crossover_is_adaptive: bool = False
    initial_population: list[npt.NDArray[np.float64]] = field(
        default_factory=lambda: cast("list[npt.NDArray[np.float64]]", [])
    )
    upper_bound: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    lower_bound: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )

    def with_bounds(self, b: bool = True) -> Configuration:
        """Enable/disable the post-crossover bounds mirror."""
        self.apply_bounds = b
        return self

    def with_crossover_probability(self, p: float) -> Configuration:
        """Set the crossover probability; must lie in [0, 1]."""
        qassert.require(
            0.0 <= p <= 1.0, f"Crossover probability ({p}) must be in [0,1] range"
        )
        self.crossover_probability = p
        return self

    def with_population_members(self, n: int) -> Configuration:
        """Set the population size and DISCARD any initial population."""
        qassert.require(n > 0, "Positive number of population members required")
        self.population_members = n
        self.initial_population.clear()
        return self

    def with_initial_population(
        self, c: list[npt.NDArray[np.float64]]
    ) -> Configuration:
        """Supply an explicit initial population (sets ``population_members``)."""
        self.initial_population = [v.astype(np.float64, copy=True) for v in c]
        self.population_members = len(c)
        return self

    def with_upper_bound(self, u: npt.NDArray[np.float64]) -> Configuration:
        """Override the constraint's upper bound."""
        self.upper_bound = u.astype(np.float64, copy=True)
        return self

    def with_lower_bound(self, low: npt.NDArray[np.float64]) -> Configuration:
        """Override the constraint's lower bound."""
        self.lower_bound = low.astype(np.float64, copy=True)
        return self

    def with_seed(self, s: int) -> Configuration:
        """Seed the Mersenne twister."""
        self.seed = s
        return self

    def with_adaptive_crossover(self, b: bool = True) -> Configuration:
        """Enable Brest-style self-adaptation of the crossover probability."""
        self.crossover_is_adaptive = b
        return self

    def with_stepsize_weight(self, w: float) -> Configuration:
        """Set the differential weight F; must lie in [0, 2]."""
        qassert.require(0.0 <= w <= 2.0, f"Step size weight ({w}) must be in [0,2] range")
        self.stepsize_weight = w
        return self

    def with_crossover_type(self, t: CrossoverType) -> Configuration:
        """Select the crossover-probability transform."""
        self.crossover_type = t
        return self

    def with_strategy(self, s: Strategy) -> Configuration:
        """Select the mutation strategy."""
        self.strategy = s
        return self


class DifferentialEvolution(OptimizationMethod):
    """Differential Evolution optimizer.

    # C++ parity: ``class DifferentialEvolution`` in
    # ql/math/optimization/differentialevolution.{hpp,cpp} (v1.43).

    ``Configuration.seed`` defaults to 0, which — in C++ and in pquantlib
    alike — routes ``MersenneTwisterUniformRng`` through the clock-based
    ``SeedGenerator``. A default-configured optimizer is therefore NOT
    reproducible; supply an explicit non-zero seed via
    ``Configuration().with_seed(...)`` when the result has to be.
    """

    Strategy = Strategy
    CrossoverType = CrossoverType
    Candidate = Candidate
    Configuration = Configuration

    __slots__ = (
        "_best_member_ever",
        "_configuration",
        "_curr_gen_crossover",
        "_curr_gen_size_weights",
        "_lower_bound",
        "_rng",
        "_upper_bound",
    )

    def __init__(self, configuration: Configuration | None = None) -> None:
        # C++ parity: differentialevolution.hpp:163-164 — the configuration
        # is stored BY VALUE (so later mutation of the caller's object is
        # invisible) and the RNG is seeded from it.
        conf = copy.deepcopy(Configuration() if configuration is None else configuration)
        self._configuration: Configuration = conf
        self._rng: MersenneTwisterUniformRng = MersenneTwisterUniformRng(conf.seed)
        self._upper_bound: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._lower_bound: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._curr_gen_size_weights: npt.NDArray[np.float64] = np.zeros(
            0, dtype=np.float64
        )
        self._curr_gen_crossover: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._best_member_ever: Candidate = Candidate.of_size(0)

    @property
    def configuration(self) -> Configuration:
        """The configuration this optimizer was built with.

        # C++ parity: differentialevolution.hpp:168-170.
        """
        return self._configuration

    # --- internals -------------------------------------------------------

    def _randomize(self, items: list[Candidate]) -> None:
        """In-place Fisher-Yates shuffle driven by ``next_int32``.

        # C++ parity: differentialevolution.cpp:36-43 — ``randomize``.
        """
        n = len(items)
        for i in range(n - 1, 0, -1):
            j = self._rng.next_int32() % (i + 1)
            items[i], items[j] = items[j], items[i]

    def _rotate_array(self, a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Shuffle the components of ``a``.

        # C++ parity: differentialevolution.cpp:351-354 — ``rotateArray``
        # takes the Array BY VALUE and shuffles it with the same
        # Fisher-Yates loop, so the copy is part of the contract.
        """
        out = a.astype(np.float64, copy=True)
        n = out.size
        for i in range(n - 1, 0, -1):
            j = self._rng.next_int32() % (i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    def _adapt_size_weights(self) -> None:
        """Brest-style self-adaptation of the differential weights.

        # C++ parity: differentialevolution.cpp:356-368 — note both the
        # gate draw and the value draw come from the same stream, and the
        # new weight is ``0.1 + u * 0.9`` (so its range is [0.1, 1.0),
        # despite the variable being called ``sizeWeightUpperBound``).
        """
        size_weight_lower_bound = 0.1
        size_weight_upper_bound = 0.9
        size_weight_change_prob = 0.1
        for i in range(self._curr_gen_size_weights.size):
            if self._rng.next_real() < size_weight_change_prob:
                self._curr_gen_size_weights[i] = (
                    size_weight_lower_bound + self._rng.next_real() * size_weight_upper_bound
                )

    def _adapt_crossover(self) -> None:
        """Self-adaptation of the crossover probabilities.

        # C++ parity: differentialevolution.cpp:370-376.
        """
        crossover_change_prob = 0.1
        for i in range(self._curr_gen_crossover.size):
            if self._rng.next_real() < crossover_change_prob:
                self._curr_gen_crossover[i] = self._rng.next_real()

    def _fill_initial_population(
        self, population: list[Candidate], problem: Problem
    ) -> None:
        """Seed member 0 from ``current_value``, the rest uniformly in the box.

        # C++ parity: differentialevolution.cpp:378-395. Costs here go
        # through ``costFunction().value`` and therefore do NOT move
        # ``Problem``'s evaluation counter.
        """
        population[0].values = problem.current_value.astype(np.float64, copy=True)
        population[0].cost = problem.cost_function.value(population[0].values)
        for j in range(1, len(population)):
            for i in range(problem.current_value.size):
                low = float(self._lower_bound[i])
                up = float(self._upper_bound[i])
                population[j].values[i] = low + (up - low) * self._rng.next_real()
            population[j].cost = problem.cost_function.value(population[j].values)
            if not np.isfinite(population[j].cost):
                population[j].cost = QL_MAX_REAL

    def _get_mutation_probabilities(
        self, population: list[Candidate]
    ) -> npt.NDArray[np.float64]:
        """Per-member crossover probability after the ``CrossoverType`` transform.

        # C++ parity: differentialevolution.cpp:323-349.
        """
        mutation_probabilities = self._curr_gen_crossover.astype(np.float64, copy=True)
        dim = population[0].values.size
        if self._configuration.crossover_type == CrossoverType.Normal:
            pass
        elif self._configuration.crossover_type == CrossoverType.Binomial:
            mutation_probabilities = (
                self._curr_gen_crossover * (1.0 - 1.0 / dim) + 1.0 / dim
            )
        elif self._configuration.crossover_type == CrossoverType.Exponential:
            for i in range(self._curr_gen_crossover.size):
                c = float(self._curr_gen_crossover[i])
                mutation_probabilities[i] = (1.0 - c**dim) / (dim * (1.0 - c))
        else:
            qassert.fail(
                f"Unknown crossover type ({int(self._configuration.crossover_type)})"
            )
        return mutation_probabilities

    def _get_crossover_mask(
        self,
        crossover_mask: list[npt.NDArray[np.float64]],
        inv_crossover_mask: list[npt.NDArray[np.float64]],
        mutation_probabilities: npt.NDArray[np.float64],
    ) -> None:
        """Fill the complementary 0/1 masks, one uniform draw per component.

        # C++ parity: differentialevolution.cpp:308-321. Note the
        # probability index is the MEMBER index while the loop also runs
        # over components, so every member uses one probability for all of
        # its coordinates.
        """
        for cm_iter in range(len(crossover_mask)):
            for mem_iter in range(crossover_mask[cm_iter].size):
                if self._rng.next_real() < mutation_probabilities[cm_iter]:
                    inv_crossover_mask[cm_iter][mem_iter] = 0.0
                else:
                    crossover_mask[cm_iter][mem_iter] = 0.0

    def _crossover(
        self,
        old_population: list[Candidate],
        population: list[Candidate],
        mutant_population: list[Candidate],
        mirror_population: list[Candidate],
        problem: Problem,
    ) -> None:
        """Recombine, mirror out-of-bounds coordinates, and price.

        # C++ parity: differentialevolution.cpp:259-306.

        C++ passes the SAME vector as ``population`` and
        ``mutantPopulation`` ("in order to avoid unnecessary copying"), so
        each member is overwritten while later members still read their
        own (unmodified) mutant entry. The Python port keeps that aliasing.
        """
        if self._configuration.crossover_is_adaptive:
            self._adapt_crossover()

        mutation_probabilities = self._get_mutation_probabilities(population)

        dim = population[0].values.size
        crossover_mask = [np.ones(dim, dtype=np.float64) for _ in population]
        inv_crossover_mask = [np.ones(dim, dtype=np.float64) for _ in population]
        self._get_crossover_mask(
            crossover_mask, inv_crossover_mask, mutation_probabilities
        )

        for pop_iter in range(len(population)):
            population[pop_iter].values = (
                old_population[pop_iter].values * inv_crossover_mask[pop_iter]
                + mutant_population[pop_iter].values * crossover_mask[pop_iter]
            )
            if self._configuration.apply_bounds:
                for mem_iter in range(population[pop_iter].values.size):
                    if population[pop_iter].values[mem_iter] > self._upper_bound[mem_iter]:
                        population[pop_iter].values[mem_iter] = self._upper_bound[
                            mem_iter
                        ] + self._rng.next_real() * (
                            mirror_population[pop_iter].values[mem_iter]
                            - self._upper_bound[mem_iter]
                        )
                    if population[pop_iter].values[mem_iter] < self._lower_bound[mem_iter]:
                        population[pop_iter].values[mem_iter] = self._lower_bound[
                            mem_iter
                        ] + self._rng.next_real() * (
                            mirror_population[pop_iter].values[mem_iter]
                            - self._lower_bound[mem_iter]
                        )
            try:
                population[pop_iter].cost = problem.value(population[pop_iter].values)
            except LibraryException:
                population[pop_iter].cost = QL_MAX_REAL
            if not np.isfinite(population[pop_iter].cost):
                population[pop_iter].cost = QL_MAX_REAL

    def _calculate_next_generation(  # noqa: PLR0915
        self, population: list[Candidate], problem: Problem
    ) -> None:
        """Build the mutant population, then cross it with the old one.

        # C++ parity: differentialevolution.cpp:109-257.
        """
        old_population = [c.copy() for c in population]
        conf = self._configuration
        mirror_population: list[Candidate]

        if conf.strategy == Strategy.Rand1Standard:
            self._randomize(population)
            shuffled1 = [c.copy() for c in population]
            self._randomize(population)
            shuffled2 = [c.copy() for c in population]
            self._randomize(population)
            mirror_population = [c.copy() for c in shuffled1]
            for i in range(len(population)):
                population[i].values = population[i].values + conf.stepsize_weight * (
                    shuffled1[i].values - shuffled2[i].values
                )

        elif conf.strategy == Strategy.BestMemberWithJitter:
            self._randomize(population)
            shuffled1 = [c.copy() for c in population]
            self._randomize(population)
            jitter = np.zeros(population[0].values.size, dtype=np.float64)
            for i in range(len(population)):
                for j in range(jitter.size):
                    jitter[j] = self._rng.next_real()
                population[i].values = self._best_member_ever.values + (
                    shuffled1[i].values - population[i].values
                ) * (0.0001 * jitter + conf.stepsize_weight)
            mirror_population = [
                self._best_member_ever.copy() for _ in range(len(population))
            ]

        elif conf.strategy == Strategy.CurrentToBest2Diffs:
            self._randomize(population)
            shuffled1 = [c.copy() for c in population]
            self._randomize(population)
            for i in range(len(population)):
                population[i].values = (
                    old_population[i].values
                    + conf.stepsize_weight
                    * (self._best_member_ever.values - old_population[i].values)
                    + conf.stepsize_weight
                    * (population[i].values - shuffled1[i].values)
                )
            mirror_population = shuffled1

        elif conf.strategy == Strategy.Rand1DiffWithPerVectorDither:
            self._randomize(population)
            shuffled1 = [c.copy() for c in population]
            self._randomize(population)
            shuffled2 = [c.copy() for c in population]
            self._randomize(population)
            mirror_population = [c.copy() for c in shuffled1]
            f_weight = np.zeros(population[0].values.size, dtype=np.float64)
            for j in range(f_weight.size):
                f_weight[j] = (
                    1.0 - conf.stepsize_weight
                ) * self._rng.next_real() + conf.stepsize_weight
            for i in range(len(population)):
                population[i].values = population[i].values + f_weight * (
                    shuffled1[i].values - shuffled2[i].values
                )

        elif conf.strategy == Strategy.Rand1DiffWithDither:
            self._randomize(population)
            shuffled1 = [c.copy() for c in population]
            self._randomize(population)
            shuffled2 = [c.copy() for c in population]
            self._randomize(population)
            mirror_population = [c.copy() for c in shuffled1]
            f_weight_scalar = (
                1.0 - conf.stepsize_weight
            ) * self._rng.next_real() + conf.stepsize_weight
            for i in range(len(population)):
                population[i].values = population[i].values + f_weight_scalar * (
                    shuffled1[i].values - shuffled2[i].values
                )

        elif conf.strategy == Strategy.EitherOrWithOptimalRecombination:
            self._randomize(population)
            shuffled1 = [c.copy() for c in population]
            self._randomize(population)
            shuffled2 = [c.copy() for c in population]
            self._randomize(population)
            mirror_population = [c.copy() for c in shuffled1]
            prob_f_weight = 0.5
            if self._rng.next_real() < prob_f_weight:
                for i in range(len(population)):
                    population[i].values = old_population[
                        i
                    ].values + conf.stepsize_weight * (
                        shuffled1[i].values - shuffled2[i].values
                    )
            else:
                # Invariant with respect to the probF weight used.
                k = 0.5 * (conf.stepsize_weight + 1)
                for i in range(len(population)):
                    population[i].values = old_population[i].values + k * (
                        shuffled1[i].values
                        - shuffled2[i].values
                        - 2.0 * population[i].values
                    )

        elif conf.strategy == Strategy.Rand1SelfadaptiveWithRotation:
            self._randomize(population)
            shuffled1 = [c.copy() for c in population]
            self._randomize(population)
            shuffled2 = [c.copy() for c in population]
            self._randomize(population)
            mirror_population = [c.copy() for c in shuffled1]
            self._adapt_size_weights()
            for i in range(len(population)):
                if self._rng.next_real() < 0.1:
                    population[i].values = self._rotate_array(
                        self._best_member_ever.values
                    )
                else:
                    population[i].values = self._best_member_ever.values + float(
                        self._curr_gen_size_weights[i]
                    ) * (shuffled1[i].values - shuffled2[i].values)

        else:
            qassert.fail(f"Unknown strategy ({int(conf.strategy)})")

        # C++ passes ``population`` as BOTH the destination and the mutants.
        self._crossover(
            old_population, population, population, mirror_population, problem
        )

    # --- OptimizationMethod ---------------------------------------------

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
        # C++ parity: differentialevolution.cpp:47-107.
        # C++ leaves ``ecType`` default-initialised; it is always written by
        # one of the two checks below before being returned.
        ec_type = Type.None_
        problem.reset()

        conf = self._configuration
        if conf.upper_bound.size == 0:
            self._upper_bound = problem.constraint.upper_bound(problem.current_value)
        else:
            qassert.require(
                conf.upper_bound.size == problem.current_value.size,
                "wrong upper bound size in differential evolution configuration",
            )
            self._upper_bound = conf.upper_bound
        if conf.lower_bound.size == 0:
            self._lower_bound = problem.constraint.lower_bound(problem.current_value)
        else:
            qassert.require(
                conf.lower_bound.size == problem.current_value.size,
                "wrong lower bound size in differential evolution configuration",
            )
            self._lower_bound = conf.lower_bound

        self._curr_gen_size_weights = np.full(
            conf.population_members, conf.stepsize_weight, dtype=np.float64
        )
        self._curr_gen_crossover = np.full(
            conf.population_members, conf.crossover_probability, dtype=np.float64
        )

        population: list[Candidate]
        if conf.initial_population:
            population = [Candidate.of_size(0) for _ in conf.initial_population]
            for i in range(len(population)):
                population[i].values = conf.initial_population[i].astype(
                    np.float64, copy=True
                )
                qassert.require(
                    population[i].values.size == problem.current_value.size,
                    "wrong values size in initial population",
                )
                population[i].cost = problem.cost_function.value(population[i].values)
        else:
            population = [
                Candidate.of_size(int(problem.current_value.size))
                for _ in range(conf.population_members)
            ]
            self._fill_initial_population(population, problem)

        # ``std::partial_sort(begin, begin+1, end)`` only guarantees the
        # FIRST element; Python's ``min`` by cost is the same observable.
        self.partial_sort_front(population)
        self._best_member_ever = population[0].copy()
        fx_old = population[0].cost
        iteration = 0
        stationary_point_iteration = 0

        # Main loop — calculate consecutive emerging populations.
        while True:
            hit = end_criteria.check_max_iterations(iteration)
            iteration += 1
            if hit is not None:
                ec_type = hit
                break
            self._calculate_next_generation(population, problem)
            self.partial_sort_front(population)
            if population[0].cost < self._best_member_ever.cost:
                self._best_member_ever = population[0].copy()
            fx_new = population[0].cost
            stationary_point_iteration, hit = (
                end_criteria.check_stationary_function_value(
                    fx_old, fx_new, stationary_point_iteration
                )
            )
            if hit is not None:
                ec_type = hit
                break
            fx_old = fx_new

        problem.set_current_value(self._best_member_ever.values)
        problem.set_function_value(self._best_member_ever.cost)
        return ec_type

    @staticmethod
    def partial_sort_front(population: list[Candidate]) -> None:
        """Bring the cheapest candidate to index 0, swapping as it scans.

        # C++ parity: ``std::partial_sort(population.begin(),
        # population.begin() + 1, population.end(), sort_by_cost())``
        # (differentialevolution.cpp:85-86 and :94-95).

        The standard only guarantees the FIRST element; the permutation of
        the rest is unspecified — and it is observable here, because the
        next generation runs a seeded Fisher-Yates shuffle over exactly
        this array. Both libc++ and libstdc++ reduce to the same loop when
        the sorted range has length one: make a one-element heap, then for
        every later element that compares smaller, SWAP it with the front
        (the sift-down and the final sort-heap are no-ops on a heap of
        one). So ``[5, 3, 1]`` becomes ``[1, 5, 3]``, not ``[1, 3, 5]``.
        """
        for i in range(1, len(population)):
            if population[i].cost < population[0].cost:
                population[0], population[i] = population[i], population[0]
