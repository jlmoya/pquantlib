"""Sampler / Probability / Temperature / Reannealing functors for HSA.

# C++ parity: ql/experimental/math/hybridsimulatedannealingfunctors.hpp
# (v1.42.1) (Copyright 2015 Andres Hernandez).

These small callable families parametrise ``HybridSimulatedAnnealing``:

* **Sampler** ``(new_point, current_point, temp) -> None`` — proposes a
  new candidate point in place.
* **Probability** ``(current_value, new_value, temp) -> bool`` — the
  Metropolis acceptance test.
* **Temperature** ``(new_temp, current_temp, steps) -> None`` — the
  annealing schedule ``T(k)`` per dimension.
* **Reannealing** ``(steps, current_point, current_value, current_temp)
  -> None`` plus ``set_problem(problem)`` — optional VFSR reannealing.

Divergence (documented): the C++ Gaussian / lognormal samplers draw
from ``std::normal_distribution`` over ``std::mt19937``; the Python port
uses inverse-transform Gaussian draws — a uniform from the bit-identical
``MersenneTwisterUniformRng`` mapped through
``InverseCumulativeNormal.standard_value``. This is distributionally
faithful but the *exact* stream differs from C++ ``std::normal_
distribution`` (a different algorithm). HSA is stochastic with a LOOSE
convergence contract, so the choice is immaterial to correctness.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.distributions.inverse_cumulative_normal import InverseCumulativeNormal
from pquantlib.math.optimization.problem import Problem
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng


def _gaussian(rng: MersenneTwisterUniformRng) -> float:
    """Standard-normal draw via inverse transform off ``rng``."""
    return InverseCumulativeNormal.standard_value(rng.next_real())


def _cauchy(rng: MersenneTwisterUniformRng) -> float:
    """Standard-Cauchy draw via inverse transform: ``tan(pi*(u - 0.5))``."""
    return math.tan(math.pi * (rng.next_real() - 0.5))


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------


class SamplerGaussian:
    """Additive Gaussian sampler (support: whole real line).

    # C++ parity: ``class SamplerGaussian``
    # hybridsimulatedannealingfunctors.hpp:62-79.
    """

    __slots__ = ("_rng",)

    def __init__(self, seed: int = 1) -> None:
        self._rng = MersenneTwisterUniformRng(seed)

    def __call__(
        self,
        new_point: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(new_point.size == current_point.size, "Incompatible input")
        qassert.require(new_point.size == temp.size, "Incompatible input")
        for i in range(current_point.size):
            new_point[i] = current_point[i] + math.sqrt(temp[i]) * _gaussian(self._rng)


class SamplerLogNormal:
    """Multiplicative lognormal sampler (support: positive reals).

    # C++ parity: ``class SamplerLogNormal``
    # hybridsimulatedannealingfunctors.hpp:42-59.
    """

    __slots__ = ("_rng",)

    def __init__(self, seed: int = 1) -> None:
        self._rng = MersenneTwisterUniformRng(seed)

    def __call__(
        self,
        new_point: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(new_point.size == current_point.size, "Incompatible input")
        qassert.require(new_point.size == temp.size, "Incompatible input")
        for i in range(current_point.size):
            new_point[i] = current_point[i] * math.exp(math.sqrt(temp[i]) * _gaussian(self._rng))


class SamplerMirrorGaussian:
    """Gaussian sampler reflected back inside ``[lower, upper]``.

    # C++ parity: ``class SamplerMirrorGaussian``
    # hybridsimulatedannealingfunctors.hpp:138-167.
    """

    __slots__ = ("_lower", "_rng", "_upper")

    def __init__(
        self,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
        seed: int = 1,
    ) -> None:
        self._lower = lower.astype(np.float64, copy=True)
        self._upper = upper.astype(np.float64, copy=True)
        self._rng = MersenneTwisterUniformRng(seed)

    def __call__(
        self,
        new_point: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(new_point.size == current_point.size, "Incompatible input")
        qassert.require(new_point.size == temp.size, "Incompatible input")
        for i in range(current_point.size):
            new_point[i] = current_point[i] + math.sqrt(temp[i]) * _gaussian(self._rng)
            while new_point[i] < self._lower[i] or new_point[i] > self._upper[i]:
                if new_point[i] < self._lower[i]:
                    new_point[i] = self._lower[i] + self._lower[i] - new_point[i]
                else:
                    new_point[i] = self._upper[i] + self._upper[i] - new_point[i]


class SamplerRingGaussian:
    """Gaussian sampler wrapped (circled) inside ``[lower, upper]``.

    # C++ parity: ``class SamplerRingGaussian``
    # hybridsimulatedannealingfunctors.hpp:86-115.
    """

    __slots__ = ("_lower", "_rng", "_upper")

    def __init__(
        self,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
        seed: int = 1,
    ) -> None:
        self._lower = lower.astype(np.float64, copy=True)
        self._upper = upper.astype(np.float64, copy=True)
        self._rng = MersenneTwisterUniformRng(seed)

    def __call__(
        self,
        new_point: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(new_point.size == current_point.size, "Incompatible input")
        qassert.require(new_point.size == temp.size, "Incompatible input")
        for i in range(current_point.size):
            new_point[i] = current_point[i] + math.sqrt(temp[i]) * _gaussian(self._rng)
            while new_point[i] < self._lower[i] or new_point[i] > self._upper[i]:
                if new_point[i] < self._lower[i]:
                    new_point[i] = self._upper[i] + new_point[i] - self._lower[i]
                else:
                    new_point[i] = self._lower[i] + new_point[i] - self._upper[i]


class SamplerCauchy:
    """Additive Cauchy sampler (heavier tails than Gaussian).

    # C++ parity: ``class SamplerCauchy``
    # hybridsimulatedannealingfunctors.hpp:174-191.
    """

    __slots__ = ("_rng",)

    def __init__(self, seed: int = 1) -> None:
        self._rng = MersenneTwisterUniformRng(seed)

    def __call__(
        self,
        new_point: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(new_point.size == current_point.size, "Incompatible input")
        qassert.require(new_point.size == temp.size, "Incompatible input")
        for i in range(current_point.size):
            new_point[i] = current_point[i] + temp[i] * _cauchy(self._rng)


class SamplerVeryFastAnnealing:
    """Very-Fast-Annealing sampler (bounded; use with VFA temperature).

    # C++ parity: ``class SamplerVeryFastAnnealing``
    # hybridsimulatedannealingfunctors.hpp:198-227.
    """

    __slots__ = ("_lower", "_rng", "_upper")

    def __init__(
        self,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
        seed: int = 1,
    ) -> None:
        qassert.require(lower.size == upper.size, "Incompatible input")
        self._lower = lower.astype(np.float64, copy=True)
        self._upper = upper.astype(np.float64, copy=True)
        self._rng = MersenneTwisterUniformRng(seed)

    def __call__(
        self,
        new_point: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(new_point.size == current_point.size, "Incompatible input")
        qassert.require(new_point.size == self._lower.size, "Incompatible input")
        qassert.require(new_point.size == temp.size, "Incompatible input")
        for i in range(current_point.size):
            new_point[i] = self._lower[i] - 1.0
            while new_point[i] < self._lower[i] or new_point[i] > self._upper[i]:
                draw = self._rng.next_real()
                # sign = (0.5 < draw) - (draw < 0.5)
                sign = float(int(draw > 0.5) - int(draw < 0.5))
                y = sign * temp[i] * (
                    (1.0 + 1.0 / temp[i]) ** abs(2.0 * draw - 1.0) - 1.0
                )
                new_point[i] = current_point[i] + y * (self._upper[i] - self._lower[i])


# --------------------------------------------------------------------------
# Probabilities
# --------------------------------------------------------------------------


class ProbabilityAlwaysDownhill:
    """Accept only strictly improving points.

    # C++ parity: ``struct ProbabilityAlwaysDownhill``
    # hybridsimulatedannealingfunctors.hpp:234-238.
    """

    def __call__(
        self, current_value: float, new_value: float, temp: npt.NDArray[np.float64]
    ) -> bool:
        return current_value > new_value


class ProbabilityBoltzmann:
    """Boltzmann acceptance: ``1/(1+exp((new-cur)/T)) > u``.

    # C++ parity: ``class ProbabilityBoltzmann``
    # hybridsimulatedannealingfunctors.hpp:245-258.
    """

    __slots__ = ("_rng",)

    def __init__(self, seed: int = 1) -> None:
        self._rng = MersenneTwisterUniformRng(seed)

    def __call__(
        self, current_value: float, new_value: float, temp: npt.NDArray[np.float64]
    ) -> bool:
        temperature = float(np.max(temp))
        return (1.0 / (1.0 + math.exp((new_value - current_value) / temperature))) > self._rng.next_real()


class ProbabilityBoltzmannDownhill:
    """Boltzmann acceptance, but always accept strictly improving points.

    # C++ parity: ``class ProbabilityBoltzmannDownhill``
    # hybridsimulatedannealingfunctors.hpp:263-278.
    """

    __slots__ = ("_rng",)

    def __init__(self, seed: int = 1) -> None:
        self._rng = MersenneTwisterUniformRng(seed)

    def __call__(
        self, current_value: float, new_value: float, temp: npt.NDArray[np.float64]
    ) -> bool:
        if new_value < current_value:
            return True
        m_temperature = float(np.max(temp))
        return (1.0 / (1.0 + math.exp((new_value - current_value) / m_temperature))) > self._rng.next_real()


# --------------------------------------------------------------------------
# Temperatures
# --------------------------------------------------------------------------


class TemperatureBoltzmann:
    """Boltzmann schedule ``T_i(k) = T0_i / log(k)``.

    # C++ parity: ``class TemperatureBoltzmann``
    # hybridsimulatedannealingfunctors.hpp:282-295.
    """

    __slots__ = ("_initial_temp",)

    def __init__(self, initial_temp: float, dimension: int) -> None:
        self._initial_temp = np.full(dimension, initial_temp, dtype=np.float64)

    def __call__(
        self,
        new_temp: npt.NDArray[np.float64],
        curr_temp: npt.NDArray[np.float64],
        steps: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(curr_temp.size == self._initial_temp.size, "Incompatible input")
        qassert.require(curr_temp.size == new_temp.size, "Incompatible input")
        for i in range(self._initial_temp.size):
            new_temp[i] = self._initial_temp[i] / math.log(steps[i])


class TemperatureCauchy:
    """Cauchy schedule ``T_i(k) = T0_i / k``.

    # C++ parity: ``class TemperatureCauchy``
    # hybridsimulatedannealingfunctors.hpp:299-312.
    """

    __slots__ = ("_initial_temp",)

    def __init__(self, initial_temp: float, dimension: int) -> None:
        self._initial_temp = np.full(dimension, initial_temp, dtype=np.float64)

    def __call__(
        self,
        new_temp: npt.NDArray[np.float64],
        curr_temp: npt.NDArray[np.float64],
        steps: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(curr_temp.size == self._initial_temp.size, "Incompatible input")
        qassert.require(curr_temp.size == new_temp.size, "Incompatible input")
        for i in range(self._initial_temp.size):
            new_temp[i] = self._initial_temp[i] / steps[i]


class TemperatureCauchy1D:
    """1-D-rescaled Cauchy schedule ``T_i(k) = T0_i / k^{1/n}``.

    # C++ parity: ``class TemperatureCauchy1D``
    # hybridsimulatedannealingfunctors.hpp:314-329.
    """

    __slots__ = ("_initial_temp", "_inverse_n")

    def __init__(self, initial_temp: float, dimension: int) -> None:
        self._inverse_n = 1.0 / dimension
        self._initial_temp = np.full(dimension, initial_temp, dtype=np.float64)

    def __call__(
        self,
        new_temp: npt.NDArray[np.float64],
        curr_temp: npt.NDArray[np.float64],
        steps: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(curr_temp.size == self._initial_temp.size, "Incompatible input")
        qassert.require(curr_temp.size == new_temp.size, "Incompatible input")
        for i in range(self._initial_temp.size):
            new_temp[i] = self._initial_temp[i] / steps[i] ** self._inverse_n


class TemperatureExponential:
    """Exponential schedule ``T_i(k) = T0_i * power^k``.

    # C++ parity: ``class TemperatureExponential``
    # hybridsimulatedannealingfunctors.hpp:331-345.
    """

    __slots__ = ("_initial_temp", "_power")

    def __init__(self, initial_temp: float, dimension: int, power: float = 0.95) -> None:
        self._initial_temp = np.full(dimension, initial_temp, dtype=np.float64)
        self._power = power

    def __call__(
        self,
        new_temp: npt.NDArray[np.float64],
        curr_temp: npt.NDArray[np.float64],
        steps: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(curr_temp.size == self._initial_temp.size, "Incompatible input")
        qassert.require(curr_temp.size == new_temp.size, "Incompatible input")
        for i in range(self._initial_temp.size):
            new_temp[i] = self._initial_temp[i] * self._power ** steps[i]


class TemperatureVeryFastAnnealing:
    """Very-Fast-Annealing schedule (Ingber 1989).

    # C++ parity: ``class TemperatureVeryFastAnnealing``
    # hybridsimulatedannealingfunctors.hpp:349-369.
    """

    __slots__ = ("_exponent", "_final_temp", "_initial_temp", "_inverse_n")

    def __init__(
        self, initial_temp: float, final_temp: float, max_steps: float, dimension: int
    ) -> None:
        self._inverse_n = 1.0 / dimension
        self._initial_temp = np.full(dimension, initial_temp, dtype=np.float64)
        self._final_temp = np.full(dimension, final_temp, dtype=np.float64)
        self._exponent = np.zeros(dimension, dtype=np.float64)
        coeff = max_steps ** (-self._inverse_n)
        for i in range(self._initial_temp.size):
            self._exponent[i] = -math.log(self._final_temp[i] / self._initial_temp[i]) * coeff

    def __call__(
        self,
        new_temp: npt.NDArray[np.float64],
        curr_temp: npt.NDArray[np.float64],
        steps: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(curr_temp.size == self._initial_temp.size, "Incompatible input")
        qassert.require(curr_temp.size == new_temp.size, "Incompatible input")
        for i in range(self._initial_temp.size):
            new_temp[i] = self._initial_temp[i] * math.exp(
                -self._exponent[i] * steps[i] ** self._inverse_n
            )


# --------------------------------------------------------------------------
# Reannealing
# --------------------------------------------------------------------------


class ReannealingTrivial:
    """No-op reannealing.

    # C++ parity: ``struct ReannealingTrivial``
    # hybridsimulatedannealingfunctors.hpp:373-380.
    """

    def set_problem(self, problem: Problem) -> None:
        """No problem state needed."""

    def __call__(
        self,
        steps: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        current_value: float,
        curr_temp: npt.NDArray[np.float64],
    ) -> None:
        """No reannealing performed."""


class ReannealingFiniteDifferences:
    """Finite-difference VFSR reannealing (rescales per-dimension steps).

    # C++ parity: ``class ReannealingFiniteDifferences``
    # hybridsimulatedannealingfunctors.hpp:389-451.
    """

    __slots__ = (
        "_bound",
        "_bounded",
        "_function_tol",
        "_initial_temp",
        "_lower",
        "_min_size",
        "_n",
        "_problem",
        "_step_size",
        "_upper",
    )

    def __init__(
        self,
        initial_temp: float,
        dimension: int,
        lower: npt.NDArray[np.float64] | None = None,
        upper: npt.NDArray[np.float64] | None = None,
        step_size: float = 1e-7,
        min_size: float = 1e-10,
        function_tol: float = 1e-10,
    ) -> None:
        self._step_size = step_size
        self._min_size = min_size
        self._function_tol = function_tol
        self._n = dimension
        self._lower = lower
        self._upper = upper
        self._initial_temp = np.full(dimension, initial_temp, dtype=np.float64)
        self._bounded = np.ones(dimension, dtype=np.float64)
        self._bound = False
        self._problem: Problem | None = None
        if lower is not None and upper is not None and lower.size > 0 and upper.size > 0:
            qassert.require(lower.size == self._n, "Incompatible input")
            qassert.require(upper.size == self._n, "Incompatible input")
            self._bound = True
            for i in range(self._n):
                self._bounded[i] = upper[i] - lower[i]

    def set_problem(self, problem: Problem) -> None:
        """Bind the problem (needed for the finite-difference gradient)."""
        self._problem = problem

    def __call__(
        self,
        steps: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        current_value: float,
        curr_temp: npt.NDArray[np.float64],
    ) -> None:
        qassert.require(curr_temp.size == self._n, "Incompatible input")
        qassert.require(steps.size == self._n, "Incompatible input")
        assert self._problem is not None

        finite_diffs = np.zeros(self._n, dtype=np.float64)
        finite_diff_max = 0.0
        offset_point = current_point.astype(np.float64, copy=True)
        for i in range(self._n):
            offset_point[i] += self._step_size
            finite_diffs[i] = self._bounded[i] * abs(
                (self._problem.value(offset_point) - current_value) / self._step_size
            )
            offset_point[i] -= self._step_size
            finite_diffs[i] = max(finite_diffs[i], self._min_size)
            finite_diff_max = max(finite_diff_max, finite_diffs[i])
        for i in range(self._n):
            t_ratio = self._initial_temp[i] / curr_temp[i]
            s_ratio = finite_diff_max / finite_diffs[i]
            if s_ratio * t_ratio < self._function_tol:
                steps[i] = abs(math.log(self._function_tol)) ** self._n
            else:
                steps[i] = abs(math.log(s_ratio * t_ratio)) ** self._n
