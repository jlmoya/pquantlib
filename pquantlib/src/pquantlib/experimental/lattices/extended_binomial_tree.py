"""ExtendedBinomialTree — time-dependent binomial trees.

# C++ parity: ql/experimental/lattices/extendedbinomialtree.{hpp,cpp}
#             (v1.42.1).

Unlike the plain :mod:`~pquantlib.methods.lattices.binomial_tree`
concretes (which freeze the drift / std / variance at ``t = 0`` and reuse
them across all slices), the *extended* trees recompute the per-step
drift, up-step, jump and probability at each slice's time
``stepTime = i * dt`` from the underlying process. This lets the tree
track time-dependent process coefficients.

The CRTP indirection of the C++ hierarchy

    ExtendedBinomialTree<T>
      <- ExtendedEqualProbabilitiesBinomialTree<T>  (upStep)
      <- ExtendedEqualJumpsBinomialTree<T>          (probUp / dxStep)

is collapsed to ordinary Python inheritance. The concrete trees
(ExtendedJarrowRudd / ExtendedCoxRossRubinstein / ExtendedAdditiveEQP /
ExtendedTrigeorgis / ExtendedTian / ExtendedLeisenReimer / ExtendedJoshi4)
override the per-step hooks.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.math.distributions.binomial_distribution import (
    peizer_pratt_method2_inversion,
)
from pquantlib.methods.lattices.tree import Tree

if TYPE_CHECKING:
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class ExtendedBinomialTree(Tree[float]):
    """Base time-dependent binomial tree.

    # C++ parity: ``ExtendedBinomialTree<T>`` (extendedbinomialtree.hpp:38-65).
    """

    branches: int = 2

    def __init__(self, process: StochasticProcess1D, end: float, steps: int) -> None:
        super().__init__(columns=steps + 1)
        self._x0: float = process.x0()
        self._dt: float = end / steps
        self._tree_process: StochasticProcess1D = process

    def size(self, i: int) -> int:
        return i + 1

    def descendant(self, i: int, index: int, branch: int) -> int:
        del i
        return index + branch

    # Time-dependent drift per step at the centre state.
    def _drift_step(self, drift_time: float) -> float:
        # C++ parity: ``ExtendedBinomialTree<T>::driftStep``.
        return self._tree_process.drift_1d(drift_time, self._x0) * self._dt

    @property
    def x0(self) -> float:
        return self._x0

    @property
    def dt(self) -> float:
        return self._dt


class ExtendedEqualProbabilitiesBinomialTree(ExtendedBinomialTree):
    """Base for equal-probability (pu = pd = 0.5) extended trees.

    # C++ parity: ``ExtendedEqualProbabilitiesBinomialTree<T>``.
    """

    def underlying(self, i: int, index: int) -> float:
        step_time = i * self._dt
        j = 2 * index - i
        # Exploiting the forward-value tree centering.
        return self._x0 * math.exp(
            i * self._drift_step(step_time) + j * self._up_step(step_time)
        )

    def probability(self, i: int, index: int, branch: int) -> float:
        del i, index, branch
        return 0.5

    @abstractmethod
    def _up_step(self, step_time: float) -> float:
        """Time-dependent up-move term at ``step_time``."""


class ExtendedEqualJumpsBinomialTree(ExtendedBinomialTree):
    """Base for equal-jump extended trees.

    # C++ parity: ``ExtendedEqualJumpsBinomialTree<T>``.
    """

    def underlying(self, i: int, index: int) -> float:
        step_time = i * self._dt
        j = 2 * index - i
        # Exploiting equal jump and the x0 tree centering.
        return self._x0 * math.exp(j * self._dx_step(step_time))

    def probability(self, i: int, index: int, branch: int) -> float:
        step_time = i * self._dt
        up_prob = self._prob_up(step_time)
        down_prob = 1 - up_prob
        return up_prob if branch == 1 else down_prob

    @abstractmethod
    def _prob_up(self, step_time: float) -> float:
        """Probability of an up move at ``step_time``."""

    @abstractmethod
    def _dx_step(self, step_time: float) -> float:
        """Time-dependent jump term at ``step_time``."""


class ExtendedJarrowRudd(ExtendedEqualProbabilitiesBinomialTree):
    """Time-dependent Jarrow-Rudd (multiplicative equal-probabilities).

    # C++ parity: ``ExtendedJarrowRudd`` (extendedbinomialtree.cpp:28-39).
    """

    def __init__(
        self, process: StochasticProcess1D, end: float, steps: int, strike: float
    ) -> None:
        del strike  # unused (signature parity with C++)
        super().__init__(process, end, steps)
        self._up: float = process.std_deviation_1d(0.0, self._x0, self._dt)

    def _up_step(self, step_time: float) -> float:
        return self._tree_process.std_deviation_1d(step_time, self._x0, self._dt)


class ExtendedCoxRossRubinstein(ExtendedEqualJumpsBinomialTree):
    """Time-dependent Cox-Ross-Rubinstein (multiplicative equal-jumps).

    # C++ parity: ``ExtendedCoxRossRubinstein`` (extendedbinomialtree.cpp:43-63).
    """

    def __init__(
        self, process: StochasticProcess1D, end: float, steps: int, strike: float
    ) -> None:
        del strike
        super().__init__(process, end, steps)
        dx = process.std_deviation_1d(0.0, self._x0, self._dt)
        pu = 0.5 + 0.5 * self._drift_step(0.0) / dx
        qassert.require(pu <= 1.0, "negative probability")
        qassert.require(pu >= 0.0, "negative probability")

    def _dx_step(self, step_time: float) -> float:
        return self._tree_process.std_deviation_1d(step_time, self._x0, self._dt)

    def _prob_up(self, step_time: float) -> float:
        return 0.5 + 0.5 * self._drift_step(step_time) / self._dx_step(step_time)


class ExtendedAdditiveEQPBinomialTree(ExtendedEqualProbabilitiesBinomialTree):
    """Time-dependent additive equal-probabilities tree.

    # C++ parity: ``ExtendedAdditiveEQPBinomialTree`` (cpp:66-81).
    """

    def __init__(
        self, process: StochasticProcess1D, end: float, steps: int, strike: float
    ) -> None:
        del strike
        super().__init__(process, end, steps)
        self._up: float = -0.5 * self._drift_step(0.0) + 0.5 * math.sqrt(
            4.0 * process.variance_1d(0.0, self._x0, self._dt)
            - 3.0 * self._drift_step(0.0) * self._drift_step(0.0)
        )

    def _up_step(self, step_time: float) -> float:
        return -0.5 * self._drift_step(step_time) + 0.5 * math.sqrt(
            4.0 * self._tree_process.variance_1d(step_time, self._x0, self._dt)
            - 3.0 * self._drift_step(step_time) * self._drift_step(step_time)
        )


class ExtendedTrigeorgis(ExtendedEqualJumpsBinomialTree):
    """Time-dependent Trigeorgis (additive equal-jumps).

    # C++ parity: ``ExtendedTrigeorgis`` (cpp:86-107).
    """

    def __init__(
        self, process: StochasticProcess1D, end: float, steps: int, strike: float
    ) -> None:
        del strike
        super().__init__(process, end, steps)
        dx = math.sqrt(
            process.variance_1d(0.0, self._x0, self._dt)
            + self._drift_step(0.0) * self._drift_step(0.0)
        )
        pu = 0.5 + 0.5 * self._drift_step(0.0) / dx
        qassert.require(pu <= 1.0, "negative probability")
        qassert.require(pu >= 0.0, "negative probability")

    def _dx_step(self, step_time: float) -> float:
        return math.sqrt(
            self._tree_process.variance_1d(step_time, self._x0, self._dt)
            + self._drift_step(step_time) * self._drift_step(step_time)
        )

    def _prob_up(self, step_time: float) -> float:
        return 0.5 + 0.5 * self._drift_step(step_time) / self._dx_step(step_time)


class ExtendedTian(ExtendedBinomialTree):
    """Time-dependent Tian (third-moment-matching multiplicative).

    # C++ parity: ``ExtendedTian`` (cpp:110-157).
    """

    def __init__(
        self, process: StochasticProcess1D, end: float, steps: int, strike: float
    ) -> None:
        del strike
        super().__init__(process, end, steps)
        q = math.exp(process.variance_1d(0.0, self._x0, self._dt))
        r = math.exp(self._drift_step(0.0)) * math.sqrt(q)
        disc = math.sqrt(q * q + 2 * q - 3)
        up = 0.5 * r * q * (q + 1 + disc)
        down = 0.5 * r * q * (q + 1 - disc)
        pu = (r - down) / (up - down)
        qassert.require(pu <= 1.0, "negative probability")
        qassert.require(pu >= 0.0, "negative probability")

    def underlying(self, i: int, index: int) -> float:
        step_time = i * self._dt
        q = math.exp(self._tree_process.variance_1d(step_time, self._x0, self._dt))
        r = math.exp(self._drift_step(step_time)) * math.sqrt(q)
        disc = math.sqrt(q * q + 2 * q - 3)
        up = 0.5 * r * q * (q + 1 + disc)
        down = 0.5 * r * q * (q + 1 - disc)
        return self._x0 * down ** float(i - index) * up ** float(index)

    def probability(self, i: int, index: int, branch: int) -> float:
        del index
        step_time = i * self._dt
        q = math.exp(self._tree_process.variance_1d(step_time, self._x0, self._dt))
        r = math.exp(self._drift_step(step_time)) * math.sqrt(q)
        disc = math.sqrt(q * q + 2 * q - 3)
        up = 0.5 * r * q * (q + 1 + disc)
        down = 0.5 * r * q * (q + 1 - disc)
        pu = (r - down) / (up - down)
        pd = 1.0 - pu
        return pu if branch == 1 else pd


class ExtendedLeisenReimer(ExtendedBinomialTree):
    """Time-dependent Leisen-Reimer (Peizer-Pratt method-2 inversion).

    # C++ parity: ``ExtendedLeisenReimer`` (cpp:160-210).
    """

    def __init__(
        self, process: StochasticProcess1D, end: float, steps: int, strike: float
    ) -> None:
        odd_steps = steps if steps % 2 != 0 else steps + 1
        super().__init__(process, end, odd_steps)
        qassert.require(strike > 0.0, f"strike {strike}must be positive")
        self._end: float = end
        self._odd_steps: int = odd_steps
        self._strike: float = strike

        variance = process.variance_1d(0.0, self._x0, end)
        ermqdt = math.exp(self._drift_step(0.0) + 0.5 * variance / odd_steps)
        d2 = (math.log(self._x0 / strike) + self._drift_step(0.0) * odd_steps) / math.sqrt(
            variance
        )
        pu = peizer_pratt_method2_inversion(d2, odd_steps)
        pdash = peizer_pratt_method2_inversion(d2 + math.sqrt(variance), odd_steps)
        _up = ermqdt * pdash / pu
        _ = (ermqdt - pu * _up) / (1.0 - pu)  # down (unused at construction)

    def underlying(self, i: int, index: int) -> float:
        step_time = i * self._dt
        variance = self._tree_process.variance_1d(step_time, self._x0, self._end)
        ermqdt = math.exp(self._drift_step(step_time) + 0.5 * variance / self._odd_steps)
        d2 = (
            math.log(self._x0 / self._strike) + self._drift_step(step_time) * self._odd_steps
        ) / math.sqrt(variance)
        pu = peizer_pratt_method2_inversion(d2, self._odd_steps)
        pdash = peizer_pratt_method2_inversion(d2 + math.sqrt(variance), self._odd_steps)
        up = ermqdt * pdash / pu
        down = (ermqdt - pu * up) / (1.0 - pu)
        return self._x0 * down ** float(i - index) * up ** float(index)

    def probability(self, i: int, index: int, branch: int) -> float:
        del index
        step_time = i * self._dt
        variance = self._tree_process.variance_1d(step_time, self._x0, self._end)
        d2 = (
            math.log(self._x0 / self._strike) + self._drift_step(step_time) * self._odd_steps
        ) / math.sqrt(variance)
        pu = peizer_pratt_method2_inversion(d2, self._odd_steps)
        pd = 1.0 - pu
        return pu if branch == 1 else pd


class ExtendedJoshi4(ExtendedBinomialTree):
    """Time-dependent Joshi-4 (method-4 inversion).

    # C++ parity: ``ExtendedJoshi4`` (cpp:214-282).
    """

    def __init__(
        self, process: StochasticProcess1D, end: float, steps: int, strike: float
    ) -> None:
        odd_steps = steps if steps % 2 != 0 else steps + 1
        super().__init__(process, end, odd_steps)
        qassert.require(strike > 0.0, f"strike {strike}must be positive")
        self._end: float = end
        self._odd_steps: int = odd_steps
        self._strike: float = strike

        variance = process.variance_1d(0.0, self._x0, end)
        ermqdt = math.exp(self._drift_step(0.0) + 0.5 * variance / odd_steps)
        d2 = (math.log(self._x0 / strike) + self._drift_step(0.0) * odd_steps) / math.sqrt(
            variance
        )
        pu = self._compute_up_prob((odd_steps - 1.0) / 2.0, d2)
        pdash = self._compute_up_prob((odd_steps - 1.0) / 2.0, d2 + math.sqrt(variance))
        _up = ermqdt * pdash / pu
        _ = (ermqdt - pu * _up) / (1.0 - pu)  # down (unused at construction)

    @staticmethod
    def _compute_up_prob(k: float, dj: float) -> float:
        alpha = dj / math.sqrt(8.0)
        alpha2 = alpha * alpha
        alpha3 = alpha * alpha2
        alpha5 = alpha3 * alpha2
        alpha7 = alpha5 * alpha2
        beta = -0.375 * alpha - alpha3
        gamma = (5.0 / 6.0) * alpha5 + (13.0 / 12.0) * alpha3 + (25.0 / 128.0) * alpha
        delta = -0.1025 * alpha - 0.9285 * alpha3 - 1.43 * alpha5 - 0.5 * alpha7
        p = 0.5
        rootk = math.sqrt(k)
        p += alpha / rootk
        p += beta / (k * rootk)
        p += gamma / (k * k * rootk)
        p += delta / (k * k * k * rootk)
        return p

    def underlying(self, i: int, index: int) -> float:
        step_time = i * self._dt
        variance = self._tree_process.variance_1d(step_time, self._x0, self._end)
        ermqdt = math.exp(self._drift_step(step_time) + 0.5 * variance / self._odd_steps)
        d2 = (
            math.log(self._x0 / self._strike) + self._drift_step(step_time) * self._odd_steps
        ) / math.sqrt(variance)
        pu = self._compute_up_prob((self._odd_steps - 1.0) / 2.0, d2)
        pdash = self._compute_up_prob(
            (self._odd_steps - 1.0) / 2.0, d2 + math.sqrt(variance)
        )
        up = ermqdt * pdash / pu
        down = (ermqdt - pu * up) / (1.0 - pu)
        return self._x0 * down ** float(i - index) * up ** float(index)

    def probability(self, i: int, index: int, branch: int) -> float:
        del index
        step_time = i * self._dt
        variance = self._tree_process.variance_1d(step_time, self._x0, self._end)
        d2 = (
            math.log(self._x0 / self._strike) + self._drift_step(step_time) * self._odd_steps
        ) / math.sqrt(variance)
        pu = self._compute_up_prob((self._odd_steps - 1.0) / 2.0, d2)
        pd = 1.0 - pu
        return pu if branch == 1 else pd


__all__ = [
    "ExtendedAdditiveEQPBinomialTree",
    "ExtendedBinomialTree",
    "ExtendedCoxRossRubinstein",
    "ExtendedEqualJumpsBinomialTree",
    "ExtendedEqualProbabilitiesBinomialTree",
    "ExtendedJarrowRudd",
    "ExtendedJoshi4",
    "ExtendedLeisenReimer",
    "ExtendedTian",
]
