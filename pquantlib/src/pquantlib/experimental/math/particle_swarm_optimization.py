"""Particle Swarm Optimization (PSO) global optimizer.

# C++ parity: ql/experimental/math/particleswarmoptimization.{hpp,cpp}
# (v1.42.1) (Copyright 2015 Andres Hernandez).

Implements the canonical PSO with constriction factor (PSO-Co) of
Clerc & Kennedy (2002), "The particle swarm-explosion, stability and
convergence in a multidimensional complex space", *IEEE Transactions
on Evolutionary Computation* 6(2): 58-73, plus the inertia-factor
variant (PSO-In).

``M`` particles explore the parameter space; each particle's velocity
is pulled toward its personal best ``P`` and the swarm/neighbourhood
best ``G`` with stochastic coefficients. The ``Inertia`` and
``Topology`` strategy objects (ported as ABCs) customise the velocity
damping and the definition of "social best" respectively.

This is a **stochastic** optimizer: with a fixed seed it is fully
deterministic, but the contract is convergence into the global basin,
not an exact value. Particles are seeded from a Sobol sequence (matching
C++), so the initial population is low-discrepancy-deterministic.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.isotropic_random_walk import IsotropicRandomWalk
from pquantlib.experimental.math.levy_flight_distribution import LevyFlightDistribution
from pquantlib.experimental.math.std_random import (
    StdMt19937,
    StdUniformIntDistribution,
)
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem

_REAL_MAX: float = sys.float_info.max


class Inertia(ABC):
    """Base inertia strategy — alters the PSO velocity state per iteration.

    # C++ parity: ``ParticleSwarmOptimization::Inertia`` in
    # particleswarmoptimization.hpp:127-152 (v1.42.1).

    The C++ version holds raw pointers into the owning PSO's state
    arrays; the Python port stores a reference to the ``ParticleSwarm
    Optimization`` instance and reaches into its public-to-the-strategy
    attributes (``_v`` etc.) through ``init``.
    """

    def __init__(self) -> None:
        self._pso: ParticleSwarmOptimization | None = None

    @abstractmethod
    def set_size(self, m: int, n: int, c0: float, end_criteria: EndCriteria) -> None:
        """Initialize strategy state for the current problem."""
        ...

    @abstractmethod
    def set_values(self) -> None:
        """Apply this iteration's inertia change to the velocity state."""
        ...

    def init(self, pso: ParticleSwarmOptimization) -> None:
        """Bind to the owning PSO (gives access to its state arrays).

        # C++ parity: ``Inertia::init`` particleswarmoptimization.hpp:142-151.
        """
        self._pso = pso


class TrivialInertia(Inertia):
    """Constant inertia: velocity is scaled by ``c0`` each iteration.

    # C++ parity: ``class TrivialInertia`` particleswarmoptimization.hpp:156-172.
    """

    def __init__(self) -> None:
        super().__init__()
        self._c0: float = 0.0
        self._m: int = 0

    def set_size(self, m: int, n: int, c0: float, end_criteria: EndCriteria) -> None:
        self._c0 = c0
        self._m = m

    def set_values(self) -> None:
        assert self._pso is not None
        v = self._pso.velocities
        for i in range(self._m):
            v[i] *= self._c0


class SimpleRandomInertia(Inertia):
    """Inertia multiplied by a uniform random in ``(threshold, 1)``.

    # C++ parity: ``class SimpleRandomInertia`` particleswarmoptimization.hpp:178-203.
    """

    def __init__(self, threshold: float = 0.5, seed: int = 1) -> None:
        super().__init__()
        qassert.require(
            0.0 <= threshold < 1.0, "Threshold must be a Real in [0, 1)"
        )
        self._threshold: float = threshold
        self._rng: MersenneTwisterUniformRng = MersenneTwisterUniformRng(seed)
        self._c0: float = 0.0
        self._m: int = 0

    def set_size(self, m: int, n: int, c0: float, end_criteria: EndCriteria) -> None:
        self._m = m
        self._c0 = c0

    def set_values(self) -> None:
        assert self._pso is not None
        v = self._pso.velocities
        for i in range(self._m):
            val = self._c0 * (self._threshold + (1.0 - self._threshold) * self._rng.next_real())
            v[i] *= val


class DecreasingInertia(Inertia):
    """Inertia decreases linearly to ``threshold`` at the last iteration.

    # C++ parity: ``class DecreasingInertia`` particleswarmoptimization.hpp:209-235.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        qassert.require(
            0.0 <= threshold < 1.0, "Threshold must be a Real in [0, 1)"
        )
        self._threshold: float = threshold
        self._c0: float = 0.0
        self._m: int = 0
        self._n: int = 0
        self._max_iterations: int = 0
        self._iteration: int = 0

    def set_size(self, m: int, n: int, c0: float, end_criteria: EndCriteria) -> None:
        self._m = m
        self._n = n
        self._c0 = c0
        self._iteration = 0
        self._max_iterations = end_criteria.max_iterations

    def set_values(self) -> None:
        assert self._pso is not None
        # C++ parity note: C++ never advances ``iteration_`` inside
        # ``setValues`` (a latent bug — the linear decay is frozen at
        # iteration 0). Reproduced faithfully.
        c0 = self._c0 * (
            self._threshold
            + (1.0 - self._threshold) * (self._max_iterations - self._iteration) / self._max_iterations
        )
        v = self._pso.velocities
        for i in range(self._m):
            v[i] *= c0


class AdaptiveInertia(Inertia):
    """Inertia halved when the swarm stalls, doubled when it improves.

    # C++ parity: ``class AdaptiveInertia`` particleswarmoptimization.hpp:233-254
    # + ``AdaptiveInertia::setValues`` particleswarmoptimization.cpp:179-206.

    From Lukic, *Approximating Kinetic Parameters Using Particle Swarm
    Optimization*. A counter tracks consecutive non-improvements of the
    swarm-wide best: above ``sh`` the inertia is halved (intensify), below
    ``sl`` it is doubled (diversify), and either way it is clamped into
    ``[min_inertia, max_inertia]``.

    Parameters
    ----------
    min_inertia, max_inertia:
        Clamp bounds applied after every rescale.
    sh:
        Stall count above which the inertia is halved.
    sl:
        Stall count below which the inertia is doubled.

    # C++ parity note: ``adaptiveCounter`` is a ``Size`` (unsigned) and is
    # decremented on improvement without a floor, so an improvement on the
    # very first scored iteration wraps it to ``SIZE_MAX``. The port keeps the
    # unsigned wrap explicitly (mask to 64 bits) rather than letting Python's
    # unbounded ints go negative, because ``counter > sh`` and ``counter < sl``
    # both read completely differently after a wrap — that wrap is why the
    # pinned schedule halves on its second step.
    """

    _COUNTER_MASK: int = 0xFFFFFFFFFFFFFFFF

    def __init__(
        self, min_inertia: float, max_inertia: float, sh: int = 5, sl: int = 2
    ) -> None:
        super().__init__()
        self._min_inertia: float = min_inertia
        self._max_inertia: float = max_inertia
        self._sh: int = sh
        self._sl: int = sl
        self._c0: float = 0.0
        self._m: int = 0
        self._best: float = _REAL_MAX
        self._adaptive_counter: int = 0
        self._started: bool = False

    def set_size(self, m: int, n: int, c0: float, end_criteria: EndCriteria) -> None:
        self._m = m
        self._c0 = c0
        self._adaptive_counter = 0
        self._best = _REAL_MAX
        self._started = False

    def set_values(self) -> None:
        assert self._pso is not None
        pbf = self._pso.personal_best_f
        curr_best = pbf[0]
        for i in range(1, self._m):
            curr_best = min(curr_best, pbf[i])
        if self._started:
            # First iteration leaves the inertia unchanged (C++ comment).
            if curr_best < self._best:
                self._best = curr_best
                self._adaptive_counter = (self._adaptive_counter - 1) & self._COUNTER_MASK
            else:
                self._adaptive_counter = (self._adaptive_counter + 1) & self._COUNTER_MASK
            if self._adaptive_counter > self._sh:
                self._c0 = max(self._min_inertia, min(self._max_inertia, self._c0 * 0.5))
            elif self._adaptive_counter < self._sl:
                self._c0 = max(self._min_inertia, min(self._max_inertia, self._c0 * 2.0))
        else:
            self._best = curr_best
            self._started = True
        v = self._pso.velocities
        for i in range(self._m):
            v[i] *= self._c0

    @property
    def inertia(self) -> float:
        """Current inertia weight ``c0`` (inspector; the C++ field is private)."""
        return self._c0


class LevyFlightInertia(Inertia):
    """Random inertia that switches to a Lévy flight for stalled particles.

    # C++ parity: ``class LevyFlightInertia``
    # particleswarmoptimization.hpp:262-312.

    While a particle keeps improving its personal best the inertia behaves
    like :class:`SimpleRandomInertia` with a fixed 0.5 threshold. Once a
    particle has gone ``threshold`` iterations without improving, its velocity
    is *replaced* (not scaled) by an isotropic Lévy-flight step ``~u^{-1/alpha}``.

    Parameters
    ----------
    alpha:
        Lévy tail exponent.
    threshold:
        Consecutive stalled iterations before the flight kicks in.
    seed:
        Seeds three independent streams, exactly as C++ does: the QuantLib-MT
        ``rng_`` used for the simple-random branch, the ``std::mt19937``
        ``generator_`` that drives the Lévy radius, and the
        ``IsotropicRandomWalk`` angle stream.

    # C++ parity note: C++ passes the same ``seed`` to all three, so the angle
    # stream and the simple-random stream are the *same* sequence consumed at
    # different points. Reproduced verbatim.
    """

    def __init__(self, alpha: float, threshold: int, seed: int = 1) -> None:
        super().__init__()
        self._rng: MersenneTwisterUniformRng = MersenneTwisterUniformRng(seed)
        self._generator: StdMt19937 = StdMt19937(seed)
        self._flight: IsotropicRandomWalk[StdMt19937] = IsotropicRandomWalk(
            engine=self._generator,
            distribution=LevyFlightDistribution(1.0, alpha),
            dim=1,
            weights=np.ones(1, dtype=np.float64),
            seed=seed,
        )
        self._threshold: int = threshold
        self._c0: float = 0.0
        self._m: int = 0
        self._n: int = 0
        self._adaptive_counter: list[int] = []
        self._personal_best_f: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

    def set_size(self, m: int, n: int, c0: float, end_criteria: EndCriteria) -> None:
        self._m = m
        self._n = n
        self._c0 = c0
        self._adaptive_counter = [0] * m

    def init(self, pso: ParticleSwarmOptimization) -> None:
        """Bind to the PSO, snapshot personal bests and size the flight.

        # C++ parity: ``LevyFlightInertia::init``
        # particleswarmoptimization.hpp:297-301.
        """
        super().init(pso)
        self._personal_best_f = pso.personal_best_f.astype(np.float64, copy=True)
        self._flight.set_dimension_bounded(self._n, pso.lower_bound, pso.upper_bound)

    def set_values(self) -> None:
        assert self._pso is not None
        pbf = self._pso.personal_best_f
        v = self._pso.velocities
        for i in range(self._m):
            if pbf[i] < self._personal_best_f[i]:
                self._personal_best_f[i] = pbf[i]
                self._adaptive_counter[i] = 0
            else:
                self._adaptive_counter[i] += 1
            if self._adaptive_counter[i] <= self._threshold:
                # Simple Random Inertia
                v[i] *= self._c0 * (0.5 + 0.5 * self._rng.next_real())
            else:
                # Stalled for threshold_ iterations: replace the velocity with
                # a Lévy-flight step.
                self._flight.next_real(v[i])


class Topology(ABC):
    """Base topology strategy — determines each particle's social best.

    # C++ parity: ``ParticleSwarmOptimization::Topology`` in
    # particleswarmoptimization.hpp:309-331 (v1.42.1).
    """

    def __init__(self) -> None:
        self._pso: ParticleSwarmOptimization | None = None

    @abstractmethod
    def set_size(self, m: int) -> None:
        """Initialize strategy state for the current swarm size."""
        ...

    @abstractmethod
    def find_social_best(self) -> None:
        """Populate the global-best arrays for this iteration."""
        ...

    def init(self, pso: ParticleSwarmOptimization) -> None:
        """Bind to the owning PSO."""
        self._pso = pso


class GlobalTopology(Topology):
    """Global topology — social best is the swarm-wide best particle.

    # C++ parity: ``class GlobalTopology`` particleswarmoptimization.hpp:337-360.

    # C++ parity note: the C++ ``findSocialBest`` searches with
    # ``if (bestF < pBF[i])`` which keeps the *largest* personal-best
    # value (a sign-flip bug — it propagates the swarm's worst point as
    # "best"). PQuantLib reproduces this verbatim; the LOOSE convergence
    # contract still holds because the velocity pull toward the personal
    # best dominates, and the final result is selected from the true
    # global minimum tracked separately in ``minimize``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._m: int = 0

    def set_size(self, m: int) -> None:
        self._m = m

    def find_social_best(self) -> None:
        assert self._pso is not None
        pbf = self._pso.personal_best_f
        pbx = self._pso.personal_best_x
        gbx = self._pso.global_best_x
        gbf = self._pso.global_best_f
        best_f = pbf[0]
        best_p = 0
        for i in range(1, self._m):
            # C++ parity: ``bestF < pBF[i]`` (keeps worst — verbatim).
            if best_f < pbf[i]:
                best_f = pbf[i]
                best_p = i
        x = pbx[best_p]
        for i in range(self._m):
            if i != best_p:
                gbx[i] = x.copy()
                gbf[i] = best_f


class KNeighbors(Topology):
    """K-neighbour topology — social best from the ``[i-K, i+K]`` ring.

    # C++ parity: ``class KNeighbors`` particleswarmoptimization.hpp:366-381
    # + ``KNeighbors::findSocialBest`` particleswarmoptimization.cpp:225-260.
    """

    def __init__(self, k: int = 1) -> None:
        super().__init__()
        qassert.require(k > 0, "Neighbors need to be larger than 0")
        self._k: int = k
        self._m: int = 0

    def set_size(self, m: int) -> None:
        self._m = m
        qassert.require(
            self._k < m, "Number of neighbors need to be smaller than total particles in swarm"
        )

    def find_social_best(self) -> None:
        assert self._pso is not None
        pbf = self._pso.personal_best_f
        pbx = self._pso.personal_best_x
        gbx = self._pso.global_best_x
        gbf = self._pso.global_best_f
        m = self._m
        k = self._k
        for i in range(m):
            best_f = pbf[i]
            best_x = 0
            upper = min(i + k, m)
            lower = max(i, k + 1) - k - 1
            for j in range(lower, upper):
                if pbf[j] < best_f:
                    best_f = pbf[j]
                    best_x = j
            if i + k >= m:  # loop around if i+K >= M
                for j in range(i + k - m):
                    if pbf[j] < best_f:
                        best_f = pbf[j]
                        best_x = j
            elif i < k:  # loop around from above
                for j in range(m - (k - i) - 1, m):
                    if pbf[j] < best_f:
                        best_f = pbf[j]
                        best_x = j
            gbx[i] = pbx[best_x].copy()
            gbf[i] = best_f


class ClubsTopology(Topology):
    """Adaptive clubs topology — social best drawn from club membership.

    # C++ parity: ``class ClubsTopology`` particleswarmoptimization.hpp:401-423
    # + particleswarmoptimization.cpp:243-400.

    From Emara, *Adaptive Clubs-based Particle Swarm Optimization*. Each
    particle starts in ``default_clubs`` of the ``total_clubs`` available; its
    social best is the best particle across the clubs it belongs to.
    A particle that is best in every one of its clubs leaves one at random
    (narrowing its influences); one that is worst in every one of its clubs
    joins one at random (widening them). Every ``reset_iteration`` iterations,
    memberships that drifted away from ``default_clubs`` are nudged back.

    .. warning::
       **Upstream defect in C++ v1.43.** ``ClubsTopology::setSize`` populates
       the initial memberships with

       .. code-block:: cpp

          std::uniform_int_distribution<Size> distribution_(1, totalClubs_);
          Size index = distribution_(generator_);
          clubSet[index] = true;              // clubSet.size() == totalClubs_
          particles4clubs_[index][i] = true;  // outer size == totalClubs_

       The distribution's range is *closed*, so ``index`` runs ``1``..
       ``totalClubs_``: club 0 is unreachable and ``index == totalClubs_`` is
       one past the end of both containers. ``std::vector`` does no bounds
       checking, so this is undefined behaviour; in practice
       ``particles4clubs_[totalClubs_]`` yields a garbage ``std::vector<bool>``
       and the process segfaults. Verified against the reference build:
       ``ClubsTopology(3, 6, 5, 2, 4, 909).setSize(8)`` crashes in
       ``ClubsTopology::setSize``. Probe block ``G`` of
       ``v143/experimental/pso.json`` pins the draw sequence that shows
       ``totalClubs_`` being produced.

       This branch is only reached when ``default_clubs < total_clubs``; when
       they are equal ``setSize`` takes the "everyone in every club" path and
       never draws, which is well-defined and is what the cross-validated
       tests exercise.

       The port does not invent a fix. It transcribes the arithmetic and
       raises :class:`LibraryException` at the point where C++ would read out
       of range, so the defect surfaces as a diagnosable error instead of
       silent corruption. Change this only if upstream changes.

    Parameters
    ----------
    default_clubs:
        Clubs each particle joins initially.
    total_clubs:
        Size of the club pool.
    max_clubs, min_clubs:
        Membership bounds (declared and validated by C++, but note the C++
        join/leave paths do not actually enforce them — reproduced as-is).
    reset_iteration:
        Iteration period at which memberships are nudged back toward
        ``default_clubs``.
    seed:
        Seed for the ``std::mt19937`` membership stream.
    """

    def __init__(
        self,
        default_clubs: int,
        total_clubs: int,
        max_clubs: int,
        min_clubs: int,
        reset_iteration: int,
        seed: int = 1,
    ) -> None:
        super().__init__()
        qassert.require(
            total_clubs >= default_clubs,
            "Total number of clubs must be larger or equal than default clubs",
        )
        qassert.require(
            default_clubs >= min_clubs,
            "Number of default clubs must be larger or equal than minimum clubs",
        )
        qassert.require(
            max_clubs >= default_clubs,
            "Number of maximum clubs must be larger or equal than default clubs",
        )
        qassert.require(
            total_clubs >= max_clubs,
            "Total number of clubs must be larger or equal than maximum clubs",
        )
        self._total_clubs: int = total_clubs
        self._max_clubs: int = max_clubs
        self._min_clubs: int = min_clubs
        self._default_clubs: int = default_clubs
        self._reset_iteration: int = reset_iteration
        self._iteration: int = 0
        self._m: int = 0
        self._clubs4particles: list[list[bool]] = []
        self._particles4clubs: list[list[bool]] = []
        self._best_by_club: list[int] = [0] * total_clubs
        self._worst_by_club: list[int] = [0] * total_clubs
        self._generator: StdMt19937 = StdMt19937(seed)
        # C++ ctor: distribution_(1, totalClubs_) — the closed range that
        # causes the out-of-range write documented in the class docstring.
        self._distribution: StdUniformIntDistribution = StdUniformIntDistribution(
            1, total_clubs
        )

    def _check_index(self, index: int) -> int:
        if index >= self._total_clubs:
            msg = (
                f"ClubsTopology drew club index {index} for a pool of "
                f"{self._total_clubs} clubs (valid 0..{self._total_clubs - 1}). "
                "This is the v1.43 upstream off-by-one: "
                "uniform_int_distribution(1, totalClubs_) is a closed range. "
                "C++ writes out of bounds here (undefined behaviour, observed "
                "as a segfault); only default_clubs == total_clubs is runnable."
            )
            raise LibraryException(msg)
        return index

    def set_size(self, m: int) -> None:
        """Assign particles to clubs.

        # C++ parity: ``ClubsTopology::setSize``
        # particleswarmoptimization.cpp:262-284.
        """
        self._m = m
        if self._default_clubs < self._total_clubs:
            self._clubs4particles = [
                [False] * self._total_clubs for _ in range(m)
            ]
            self._particles4clubs = [[False] * m for _ in range(self._total_clubs)]
            # Assign particles to clubs randomly.
            for i in range(m):
                club_set = self._clubs4particles[i]
                for _ in range(self._default_clubs):
                    index = self._check_index(self._distribution(self._generator))
                    while club_set[index]:
                        index = self._check_index(self._distribution(self._generator))
                    club_set[index] = True
                    self._particles4clubs[index][i] = True
        else:
            # total_clubs == default_clubs: everyone is in every club.
            self._clubs4particles = [[True] * self._total_clubs for _ in range(m)]
            self._particles4clubs = [[True] * m for _ in range(self._total_clubs)]

    def _rank_clubs(self, pbf: npt.NDArray[np.float64]) -> None:
        """Record the best and worst particle in each club.

        # C++ parity: the first loop of ``ClubsTopology::findSocialBest``
        # particleswarmoptimization.cpp:295-316.
        """
        for i in range(self._total_clubs):
            best_by_club = _REAL_MAX
            worst_by_club = -_REAL_MAX
            best_p = 0
            worst_p = 0
            particles_set = self._particles4clubs[i]
            for j in range(self._m):
                if particles_set[j]:
                    # C++ parity: this is an if/else-if chain, so a particle
                    # that is the club best is never also considered for the
                    # club worst. Verbatim.
                    if best_by_club > pbf[j]:
                        best_by_club = pbf[j]
                        best_p = j
                    elif worst_by_club < pbf[j]:
                        worst_by_club = pbf[j]
                        worst_p = j
            self._best_by_club[i] = best_p
            self._worst_by_club[i] = worst_p

    def find_social_best(self) -> None:
        """Update memberships, then each particle's social best.

        # C++ parity: ``ClubsTopology::findSocialBest``
        # particleswarmoptimization.cpp:286-367.
        """
        assert self._pso is not None
        pbf = self._pso.personal_best_f
        pbx = self._pso.personal_best_x
        gbx = self._pso.global_best_x
        gbf = self._pso.global_best_f

        self._iteration += 1
        reset = False
        if self._iteration == self._reset_iteration:
            self._iteration = 0
            reset = True

        self._rank_clubs(pbf)

        # Update clubs and each particle's global best.
        for i in range(self._m):
            club_set = self._clubs4particles[i]
            best = True
            worst = True
            current_clubs = 0
            for j in range(self._total_clubs):
                if club_set[j]:
                    if best and i != self._best_by_club[j]:
                        best = False
                    if worst and i != self._worst_by_club[j]:
                        worst = False
                    current_clubs += 1

            if best:
                self._leave_random_club(i, current_clubs)
            elif worst:
                self._join_random_club(i, current_clubs)
            elif reset and current_clubs != self._default_clubs:
                if current_clubs < self._default_clubs:
                    self._join_random_club(i, current_clubs)
                else:
                    self._leave_random_club(i, current_clubs)

            best_neighbor_f = _REAL_MAX
            best_neighbor_x = 0
            for j in range(self._total_clubs):
                if club_set[j] and best_neighbor_f > pbf[self._best_by_club[j]]:
                    best_neighbor_f = pbf[self._best_by_club[j]]
                    best_neighbor_x = j
            # C++ parity: indexes pBX_ by the CLUB index j, not by the
            # particle ``bestByClub_[j]`` that achieved the best value. That
            # is a latent upstream bug (a club index is used as a particle
            # index), but it is well-defined whenever totalClubs_ <= M, and
            # it is what the pinned trajectories reflect. Verbatim.
            gbx[i] = pbx[best_neighbor_x].copy()
            gbf[i] = best_neighbor_f

    def _leave_random_club(self, particle: int, current_clubs: int) -> None:
        # C++ parity: particleswarmoptimization.cpp:369-383.
        rand_index = self._distribution(self._generator, 1, current_clubs)
        index = 1
        club_set = self._clubs4particles[particle]
        for j in range(self._total_clubs):
            if club_set[j]:
                if index == rand_index:
                    club_set[j] = False
                    self._particles4clubs[j][particle] = False
                    break
                index += 1

    def _join_random_club(self, particle: int, current_clubs: int) -> None:
        # C++ parity: particleswarmoptimization.cpp:385-400.
        rand_index = (
            1
            if self._total_clubs == current_clubs
            else self._distribution(self._generator, 1, self._total_clubs - current_clubs)
        )
        index = 1
        club_set = self._clubs4particles[particle]
        for j in range(self._total_clubs):
            if not club_set[j]:
                if index == rand_index:
                    club_set[j] = True
                    self._particles4clubs[j][particle] = True
                    break
                index += 1


class ParticleSwarmOptimization(OptimizationMethod):
    """Particle Swarm Optimization constrained global optimizer.

    # C++ parity: ``class ParticleSwarmOptimization`` in
    # particleswarmoptimization.hpp:91-124 + the ``.cpp`` driver.

    Parameters
    ----------
    m:
        Number of particles (swarm size).
    topology:
        Social-best strategy (e.g. ``Topology``, ``KNeighbors``).
    inertia:
        Velocity-damping strategy (e.g. ``TrivialInertia``).
    c1, c2:
        Self- and social-recognition coefficients. Default 2.05 each
        (Clerc-Kennedy recommendation; phi = c1 + c2 = 4.1).
    omega:
        If supplied, selects the PSO-In inertia-factor variant with this
        inertia weight (and ``c1``/``c2`` used directly). If omitted, the
        PSO-Co constriction factor is computed from ``phi``.
    seed:
        Seed for the velocity-update Mersenne-Twister stream.

    Strategy state arrays are exposed as properties (``velocities``,
    ``personal_best_x`` etc.) so the ``Inertia`` / ``Topology`` objects
    can read and mutate them — mirroring the C++ friend-pointer wiring.
    """

    def __init__(
        self,
        m: int,
        topology: Topology,
        inertia: Inertia,
        c1: float = 2.05,
        c2: float = 2.05,
        omega: float | None = None,
        seed: int = 1,
    ) -> None:
        self._m: int = m
        self._n: int = 0
        self._rng: MersenneTwisterUniformRng = MersenneTwisterUniformRng(seed)
        self._topology: Topology = topology
        self._inertia: Inertia = inertia

        if omega is None:
            # PSO-Co: compute constriction factor from phi.
            phi = c1 + c2
            # C++ uses QL_ENSURE (postcondition); PQuantLib maps both
            # QL_REQUIRE and QL_ENSURE onto qassert.require.
            qassert.require(phi * phi - 4.0 * phi != 0.0, "Invalid phi")
            self._c0: float = 2.0 / abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi))
            self._c1: float = self._c0 * c1
            self._c2: float = self._c0 * c2
        else:
            # PSO-In: inertia factor used directly.
            self._c0 = omega
            self._c1 = c1
            self._c2 = c2

        # Per-problem state (populated by start_state).
        self._x: list[npt.NDArray[np.float64]] = []
        self._v: list[npt.NDArray[np.float64]] = []
        self._pbx: list[npt.NDArray[np.float64]] = []
        self._gbx: list[npt.NDArray[np.float64]] = []
        self._pbf: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._gbf: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._lx: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._ux: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

    # -- strategy-facing state accessors (mirror C++ friend pointers) -----

    @property
    def constriction_factor(self) -> float:
        """The velocity-damping weight ``c0`` handed to the inertia strategy.

        Read-only inspector; the C++ field ``c0_`` is protected and has no
        accessor. For the PSO-Co constructor this is the Clerc-Kennedy
        constriction factor derived from ``phi``; for PSO-In it is ``omega``.
        """
        return self._c0

    @property
    def velocities(self) -> list[npt.NDArray[np.float64]]:
        """Per-particle velocity arrays."""
        return self._v

    @property
    def personal_best_x(self) -> list[npt.NDArray[np.float64]]:
        """Per-particle personal-best position arrays."""
        return self._pbx

    @property
    def personal_best_f(self) -> npt.NDArray[np.float64]:
        """Per-particle personal-best function values."""
        return self._pbf

    @property
    def global_best_x(self) -> list[npt.NDArray[np.float64]]:
        """Per-particle social-best position arrays."""
        return self._gbx

    @property
    def global_best_f(self) -> npt.NDArray[np.float64]:
        """Per-particle social-best function values."""
        return self._gbf

    @property
    def lower_bound(self) -> npt.NDArray[np.float64]:
        """Lower parameter bound."""
        return self._lx

    @property
    def upper_bound(self) -> npt.NDArray[np.float64]:
        """Upper parameter bound."""
        return self._ux

    # -- driver -----------------------------------------------------------

    def start_state(self, problem: Problem, end_criteria: EndCriteria) -> None:
        """Seed the swarm from a Sobol sequence and evaluate personal bests.

        # C++ parity: ``ParticleSwarmOptimization::startState``
        # particleswarmoptimization.cpp:53-90.
        """
        self._n = problem.current_value.size
        self._topology.set_size(self._m)
        self._inertia.set_size(self._m, self._n, self._c0, end_criteria)

        self._x = []
        self._v = []
        self._pbx = []
        self._gbx = []
        self._pbf = np.empty(self._m, dtype=np.float64)
        self._gbf = np.empty(self._m, dtype=np.float64)
        self._ux = problem.constraint.upper_bound(problem.current_value)
        self._lx = problem.constraint.lower_bound(problem.current_value)
        bounds = self._ux - self._lx

        # Random initialization via a Sobol sequence (dim 2N: position +
        # velocity components).
        sobol = SobolRsg(self._n * 2)

        for i in range(self._m):
            sample = sobol.next_sequence()
            x = np.zeros(self._n, dtype=np.float64)
            v = np.zeros(self._n, dtype=np.float64)
            self._gbx.append(np.zeros(self._n, dtype=np.float64))
            for j in range(self._n):
                # X = lb + (ub - lb) * random
                x[j] = self._lx[j] + bounds[j] * sample[2 * j]
                # V = (ub - lb) * (2*random - 1) in (lb-ub, ub-lb)
                v[j] = bounds[j] * (2.0 * sample[2 * j + 1] - 1.0)
            self._x.append(x)
            self._v.append(v)
            self._pbx.append(x.copy())
            self._pbf[i] = problem.value(x)

        self._topology.init(self)
        self._inertia.init(self)

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
        """Run PSO; return the termination outcome.

        # C++ parity: ``ParticleSwarmOptimization::minimize``
        # particleswarmoptimization.cpp:92-180.
        """
        # C++ parity: ``QL_REQUIRE(!P.constraint().empty(), ...)``. Python
        # constraints are never "empty" (they always expose bounds), so
        # the empty-guard is dropped; PSO reads the constraint's box
        # bounds in ``start_state`` regardless.
        ec_type = Type.None_
        problem.reset()
        iteration = 0
        iteration_stat = 0
        max_iteration = end_criteria.max_iterations
        max_i_stationary = end_criteria.max_stationary_state
        best_value = _REAL_MAX
        best_position = 0

        self.start_state(problem, end_criteria)
        for i in range(self._m):
            if self._pbf[i] < best_value:
                best_value = self._pbf[i]
                best_position = i

        while True:
            iteration += 1
            iteration_stat += 1
            if iteration > max_iteration or iteration_stat > max_i_stationary:
                break

            self._topology.find_social_best()
            self._inertia.set_values()

            for i in range(self._m):
                x = self._x[i]
                pb = self._pbx[i]
                gb = self._gbx[i]
                v = self._v[i]
                for j in range(self._n):
                    v[j] += self._c1 * self._rng.next_real() * (pb[j] - x[j]) + self._c2 * self._rng.next_real() * (
                        gb[j] - x[j]
                    )
                    x[j] += v[j]
                    if x[j] < self._lx[j]:
                        x[j] = self._lx[j]
                        v[j] = 0.0
                    elif x[j] > self._ux[j]:
                        x[j] = self._ux[j]
                        v[j] = 0.0
                f = problem.value(x)
                if f < self._pbf[i]:
                    self._pbf[i] = f
                    # C++ assigns ``pB = x`` (copy into the personal-best
                    # buffer). NumPy slice-assign keeps the buffer identity.
                    pb[:] = x
                    if f < best_value:
                        best_value = f
                        best_position = i
                        iteration_stat = 0

        ec_type = Type.MaxIterations if iteration > max_iteration else Type.StationaryPoint

        problem.set_current_value(self._pbx[best_position])
        problem.set_function_value(best_value)
        return ec_type
