"""Firefly Algorithm global optimizer (with hybrid differential evolution).

# C++ parity: ql/experimental/math/fireflyalgorithm.{hpp,cpp}
# (v1.42.1) (Copyright 2015 Andres Hernandez).

Implementation based on Yang (2009), "Firefly Algorithm, Levy Flights
and Global Optimization", *Research and Development in Intelligent
Systems XXVI*, pp 209-218, extended with the hybrid firefly/DE operator
of Abdullah et al. (2012).

``M`` fireflies explore the space; each is attracted toward brighter
fireflies (lower function value) by an ``Intensity`` kernel, plus a
``RandomWalk`` perturbation. Optionally a ``Mde``-sized subpopulation is
updated by a differential-evolution operator instead, turning this into
a fully-fledged DE optimizer when ``Mde == M``.

Stochastic optimizer, LOOSE convergence contract. Fireflies seed from a
Sobol sequence (matching C++). The random walk is driven by an
``IsotropicRandomWalk`` over the bit-identical MT angle stream.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.experimental.math.isotropic_random_walk import IsotropicRandomWalk
from pquantlib.experimental.math.levy_flight_distribution import LevyFlightDistribution
from pquantlib.math.distributions.inverse_cumulative_normal import InverseCumulativeNormal
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem

_EPSILON: float = sys.float_info.epsilon
_REAL_MAX: float = sys.float_info.max


class _GaussianRadius:
    """Radius distribution drawing N(0, sigma) via inverse transform.

    Stands in for the C++ ``std::normal_distribution<Real>(0, sigma)``
    used as the IsotropicRandomWalk radius source. Distributionally
    faithful; the exact stream differs from C++ std::normal_distribution
    (documented — firefly is stochastic with a LOOSE contract).
    """

    __slots__ = ("_sigma",)

    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    def __call__(self, engine: MersenneTwisterUniformRng) -> float:
        return self._sigma * InverseCumulativeNormal.standard_value(engine.next_real())


class Intensity(ABC):
    """Base intensity kernel — accumulates the attraction toward brighter peers.

    # C++ parity: ``FireflyAlgorithm::Intensity``
    # fireflyalgorithm.hpp:101-138.
    """

    def __init__(self) -> None:
        self._fa: FireflyAlgorithm | None = None

    @abstractmethod
    def _intensity_impl(self, value_x: float, value_y: float, distance: float) -> float:
        """The intensity kernel ``I(value_x, value_y, distance)``."""
        ...

    def _distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        # C++ parity: squared Euclidean distance, fireflyalgorithm.hpp:127-134.
        assert self._fa is not None
        d = 0.0
        for i in range(self._fa.n):
            diff = x[i] - y[i]
            d += diff * diff
        return d

    def init(self, fa: FireflyAlgorithm) -> None:
        """Bind to the owning FireflyAlgorithm."""
        self._fa = fa

    def find_brightest(self) -> None:
        """Populate the per-firefly intensity-pull vectors.

        # C++ parity: ``FireflyAlgorithm::Intensity::findBrightest``
        # fireflyalgorithm.cpp:232-256.
        """
        assert self._fa is not None
        fa = self._fa
        x_all = fa.x
        xi_all = fa.xi
        values = fa.values
        n = fa.n
        mfa = fa.mfa

        # Brightest ignores all others.
        xi0 = xi_all[values[0][1]]
        for j in range(n):
            xi0[j] = 0.0

        for i in range(1, mfa):
            index = values[i][1]
            x = x_all[index]
            xi = xi_all[index]
            for j in range(n):
                xi[j] = 0.0
            value_x = values[i][0]
            # C++ parity: inner loop runs k in [0, i-1) (note: i-1, not i),
            # so the immediately-brighter neighbour is skipped — verbatim.
            for k in range(i - 1):
                y = x_all[values[k][1]]
                value_y = values[k][0]
                intensity = self._intensity_impl(value_x, value_y, self._distance(x, y))
                for j in range(n):
                    xi[j] += intensity * (y[j] - x[j])


class ExponentialIntensity(Intensity):
    """Exponentially-decreasing intensity ``(b0-bmin)exp(-gamma d)+bmin``.

    # C++ parity: ``class ExponentialIntensity`` fireflyalgorithm.hpp:142-154.
    """

    def __init__(self, beta0: float, beta_min: float, gamma: float) -> None:
        super().__init__()
        self._beta0 = beta0
        self._beta_min = beta_min
        self._gamma = gamma

    def _intensity_impl(self, value_x: float, value_y: float, distance: float) -> float:
        return (self._beta0 - self._beta_min) * math.exp(-self._gamma * distance) + self._beta_min


class InverseLawSquareIntensity(Intensity):
    """Inverse-square intensity ``(b0-bmin)/(d+eps)+bmin``.

    # C++ parity: ``class InverseLawSquareIntensity`` fireflyalgorithm.hpp:158-168.
    """

    def __init__(self, beta0: float, beta_min: float) -> None:
        super().__init__()
        self._beta0 = beta0
        self._beta_min = beta_min

    def _intensity_impl(self, value_x: float, value_y: float, distance: float) -> float:
        return (self._beta0 - self._beta_min) / (distance + _EPSILON) + self._beta_min


class RandomWalk(ABC):
    """Base random-walk perturbation kernel.

    # C++ parity: ``FireflyAlgorithm::RandomWalk``
    # fireflyalgorithm.hpp:172-198.
    """

    def __init__(self) -> None:
        self._fa: FireflyAlgorithm | None = None

    @abstractmethod
    def _walk_impl(self, x_rw: npt.NDArray[np.float64]) -> None:
        """Perturb the walk buffer ``x_rw`` in place."""
        ...

    def init(self, fa: FireflyAlgorithm) -> None:
        """Bind to the owning FireflyAlgorithm."""
        self._fa = fa

    def walk(self) -> None:
        """Apply the random walk to every firefly's walk buffer.

        # C++ parity: ``RandomWalk::walk`` fireflyalgorithm.hpp:177-181.
        """
        assert self._fa is not None
        fa = self._fa
        for i in range(fa.mfa):
            self._walk_impl(fa.x_rw[fa.values[i][1]])


class DistributionRandomWalk(RandomWalk):
    """Random walk via an IsotropicRandomWalk over a radius distribution.

    # C++ parity: ``template<Distribution> class DistributionRandomWalk``
    # fireflyalgorithm.hpp:204-221.
    """

    def __init__(
        self,
        distribution: object,
        delta: float = 0.9,
        seed: int = 1,
    ) -> None:
        super().__init__()
        # The radius distribution is consumed by an internal MT engine.
        self._walk_random = IsotropicRandomWalk(
            engine=MersenneTwisterUniformRng(seed),
            distribution=distribution,  # type: ignore[arg-type]
            dim=1,
            weights=np.ones(1, dtype=np.float64),
            seed=seed,
        )
        self._delta = delta

    def _walk_impl(self, x_rw: npt.NDArray[np.float64]) -> None:
        self._walk_random.next_real(x_rw)
        x_rw *= self._delta

    def init(self, fa: FireflyAlgorithm) -> None:
        super().init(fa)
        self._walk_random.set_dimension_bounded(fa.n, fa.lower_bound, fa.upper_bound)


class GaussianWalk(DistributionRandomWalk):
    """Gaussian random walk.

    # C++ parity: ``class GaussianWalk`` fireflyalgorithm.hpp:225-234.
    """

    def __init__(self, sigma: float, delta: float = 0.9, seed: int = 1) -> None:
        super().__init__(_GaussianRadius(sigma), delta, seed)


class LevyFlightWalk(DistributionRandomWalk):
    """Lévy-flight random walk.

    # C++ parity: ``class LevyFlightWalk`` fireflyalgorithm.hpp:238-247.
    """

    def __init__(
        self, alpha: float, xm: float = 0.5, delta: float = 0.9, seed: int = 1
    ) -> None:
        super().__init__(LevyFlightDistribution(xm, alpha), delta, seed)


class FireflyAlgorithm(OptimizationMethod):
    """Firefly Algorithm constrained global optimizer (firefly + DE hybrid).

    # C++ parity: ``class FireflyAlgorithm`` fireflyalgorithm.hpp:74-95
    # + the ``.cpp`` driver.

    Parameters
    ----------
    m:
        Total population size.
    intensity:
        Intensity kernel (e.g. ``ExponentialIntensity``).
    random_walk:
        Random-walk perturbation (e.g. ``GaussianWalk``, ``LevyFlightWalk``).
    m_de:
        Size of the differential-evolution subpopulation (0 -> pure
        firefly; ``m`` -> pure DE).
    mutation_factor, crossover_factor:
        DE operator parameters.
    seed:
        Seed for the DE index / crossover MT stream.

    Strategy-facing state (``x``, ``xi``, ``x_rw``, ``values`` etc.) is
    exposed as properties so the ``Intensity`` / ``RandomWalk`` objects
    can read and mutate it — mirroring the C++ friend-pointer wiring.
    """

    def __init__(
        self,
        m: int,
        intensity: Intensity,
        random_walk: RandomWalk,
        m_de: int = 0,
        mutation_factor: float = 1.0,
        crossover_factor: float = 0.5,
        seed: int = 1,
    ) -> None:
        qassert.require(
            m >= m_de,
            "Differential Evolution subpopulation cannot be larger than total population",
        )
        self._mutation = mutation_factor
        self._crossover = crossover_factor
        self._m = m
        self._m_de = m_de
        self._mfa = m - m_de
        self._n = 0
        self._intensity = intensity
        self._random_walk = random_walk
        self._rng = MersenneTwisterUniformRng(seed)
        # Integer index sampler: C++ uses std::uniform_int_distribution
        # over [Mfa_, (Mde>0 ? M-1 : M)). We seed a separate MT for the
        # DE index draws + crossover dimension index.
        self._index_rng = MersenneTwisterUniformRng(seed if seed != 1 else 2)
        self._index_lo = self._mfa
        self._index_hi = (m - 1) if m_de > 0 else m  # exclusive upper used as [lo, hi)

        self._x: list[npt.NDArray[np.float64]] = []
        self._xi: list[npt.NDArray[np.float64]] = []
        self._x_rw: list[npt.NDArray[np.float64]] = []
        self._values: list[tuple[float, int]] = []
        self._lx: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._ux: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

    # -- strategy-facing accessors ----------------------------------------

    @property
    def n(self) -> int:
        """Problem dimension."""
        return self._n

    @property
    def mfa(self) -> int:
        """Firefly-subpopulation size (``M - Mde``)."""
        return self._mfa

    @property
    def x(self) -> list[npt.NDArray[np.float64]]:
        """Per-firefly position arrays."""
        return self._x

    @property
    def xi(self) -> list[npt.NDArray[np.float64]]:
        """Per-firefly intensity-pull arrays."""
        return self._xi

    @property
    def x_rw(self) -> list[npt.NDArray[np.float64]]:
        """Per-firefly random-walk arrays."""
        return self._x_rw

    @property
    def values(self) -> list[tuple[float, int]]:
        """``(function_value, original_index)`` pairs (sorted each iteration)."""
        return self._values

    @property
    def lower_bound(self) -> npt.NDArray[np.float64]:
        """Lower parameter bound."""
        return self._lx

    @property
    def upper_bound(self) -> npt.NDArray[np.float64]:
        """Upper parameter bound."""
        return self._ux

    def _draw_index(self, lo: int, hi: int) -> int:
        # Uniform integer in [lo, hi) (hi exclusive). Matches C++
        # std::uniform_int_distribution(lo, hi-1) inclusive semantics.
        span = hi - lo
        if span <= 0:
            return lo
        return lo + int(self._index_rng.next_real() * span) % span

    # -- driver -----------------------------------------------------------

    def start_state(self, problem: Problem, end_criteria: EndCriteria) -> None:
        """Seed the population from a Sobol sequence and evaluate.

        # C++ parity: ``FireflyAlgorithm::startState``
        # fireflyalgorithm.cpp:44-73.
        """
        self._n = problem.current_value.size
        self._x = []
        self._xi = []
        self._x_rw = []
        self._values = []
        self._ux = problem.constraint.upper_bound(problem.current_value)
        self._lx = problem.constraint.lower_bound(problem.current_value)
        bounds = self._ux - self._lx

        sobol = SobolRsg(self._n)
        for i in range(self._m):
            sample = sobol.next_sequence()
            x = np.zeros(self._n, dtype=np.float64)
            self._xi.append(np.zeros(self._n, dtype=np.float64))
            self._x_rw.append(np.zeros(self._n, dtype=np.float64))
            for j in range(self._n):
                x[j] = self._lx[j] + bounds[j] * sample[j]
            self._x.append(x)
            self._values.append((problem.value(x), i))

        self._intensity.init(self)
        self._random_walk.init(self)

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:  # noqa: PLR0915
        """Run the firefly/DE optimizer; return the termination outcome.

        # C++ parity: ``FireflyAlgorithm::minimize``
        # fireflyalgorithm.cpp:75-210.
        """
        ec_type = Type.None_
        problem.reset()
        iteration = 0
        iteration_stat = 0
        max_iteration = end_criteria.max_iterations
        max_i_stationary = end_criteria.max_stationary_state

        self.start_state(problem, end_criteria)

        is_fa = self._mfa > 0
        z = np.zeros(self._n, dtype=np.float64)

        best_value = self._values[0][0]
        best_position = 0
        for i in range(1, self._m):
            if self._values[i][0] < best_value:
                best_position = i
                best_value = self._values[i][0]
        best_x = self._x[best_position].astype(np.float64, copy=True)

        while True:
            iteration += 1
            iteration_stat += 1
            if iteration > max_iteration or iteration_stat > max_i_stationary:
                break

            # Sort by value (ascending) — divides into sub-populations.
            self._values.sort(key=lambda t: t[0])

            # Differential evolution on the worse subpopulation.
            if self._mfa < self._m:
                index_best = self._values[0][1]
                x_best = self._x[index_best]
                for i in range(self._mfa, self._m):
                    if not is_fa:
                        index_best = self._draw_index(self._index_lo, self._index_hi)
                        x_best = self._x[index_best]
                    index_r1 = self._draw_index(self._index_lo, self._index_hi)
                    while index_r1 == index_best:
                        index_r1 = self._draw_index(self._index_lo, self._index_hi)
                    index_r2 = self._draw_index(self._index_lo, self._index_hi)
                    while index_r2 in (index_best, index_r1):
                        index_r2 = self._draw_index(self._index_lo, self._index_hi)

                    index = self._values[i][1]
                    x = self._x[index]
                    x_r1 = self._x[index_r1]
                    x_r2 = self._x[index_r2]
                    r_index = self._draw_index(0, self._n)
                    for j in range(self._n):
                        if j == r_index or self._rng.next_real() <= self._crossover:
                            z[j] = x_best[j] + self._mutation * (x_r1[j] - x_r2[j])
                        else:
                            z[j] = x[j]
                        if z[j] < self._lx[j]:
                            z[j] = self._lx[j]
                        elif z[j] > self._ux[j]:
                            z[j] = self._ux[j]
                    val = problem.value(z)
                    # C++ parity: indexes ``values_[index]`` by the
                    # original firefly index (not the loop position i).
                    if val < self._values[index][0]:
                        x[:] = z
                        self._values[index] = (val, self._values[index][1])
                        if val < best_value:
                            best_value = val
                            best_x = x.astype(np.float64, copy=True)
                            iteration_stat = 0

            # Firefly algorithm on the better subpopulation.
            if is_fa:
                self._intensity.find_brightest()
                self._random_walk.walk()
                for i in range(self._mfa):
                    index = self._values[i][1]
                    x = self._x[index]
                    xi = self._xi[index]
                    x_rw = self._x_rw[index]
                    for j in range(self._n):
                        z[j] = x[j] + xi[j] + x_rw[j]
                        if z[j] < self._lx[j]:
                            z[j] = self._lx[j]
                        elif z[j] > self._ux[j]:
                            z[j] = self._ux[j]
                    val = problem.value(z)
                    if not math.isnan(val):
                        x[:] = z
                        self._values[index] = (val, self._values[index][1])
                        if val < best_value:
                            best_value = val
                            best_x = x.astype(np.float64, copy=True)
                            iteration_stat = 0

        ec_type = Type.MaxIterations if iteration > max_iteration else Type.StationaryPoint
        problem.set_current_value(best_x)
        problem.set_function_value(best_value)
        return ec_type
