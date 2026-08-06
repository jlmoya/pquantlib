"""Binomial trees — the full ql/methods/lattices/binomialtree family.

# C++ parity: ql/methods/lattices/binomialtree.{hpp,cpp} (v1.43).

The C++ class hierarchy, reproduced here one-for-one:

    Tree<Impl> (CRTP) <- BinomialTree<Impl>
                          <- EqualProbabilitiesBinomialTree<Impl>
                             <- JarrowRudd
                             <- AdditiveEQPBinomialTree
                          <- EqualJumpsBinomialTree<Impl>
                             <- CoxRossRubinstein
                             <- Trigeorgis
                          <- Tian
                          <- LeisenReimer
                          <- Joshi4

Only the CRTP indirection is dropped (Python's method lookup is already
dynamic); the layering is the C++ one, so the two intermediate bases own
their ``underlying`` / ``probability`` formulas exactly as in C++:

* ``EqualProbabilitiesBinomialTree`` — ``p == 0.5`` on both branches and
  a *drift-centred* underlying ``x0 * exp(i*driftPerStep + j*up)``;
* ``EqualJumpsBinomialTree`` — asymmetric ``(pu, pd)`` and an
  *x0-centred* underlying ``x0 * exp(j*dx)``;

with ``j = 2*index - i`` in both. ``Tian`` / ``LeisenReimer`` / ``Joshi4``
derive straight from ``BinomialTree`` and use the multiplicative layout
``x0 * down^(i-index) * up^index``, which is *not* centred on x0.

# C++ parity notes:
#
# * The C++ ``BinomialTree`` ctor takes ``steps`` and stores
#   ``columns = steps + 1``. Our ``columns()`` (inherited from
#   :class:`Tree`) matches.
# * ``descendant(i, index, branch) = index + branch`` for every
#   binomial tree (branch 0 = down, branch 1 = up — same orientation
#   as the L3-D BinomialVanillaEngine).
# * ``LeisenReimer`` and ``Joshi4`` round an even ``steps`` up to the
#   next odd number *before* calling the base ctor, so ``dt`` and
#   ``driftPerStep`` are computed against the rounded count.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.math.distributions.binomial_distribution import (
    peizer_pratt_method2_inversion,
)
from pquantlib.methods.lattices.tree import Tree

if TYPE_CHECKING:
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class BinomialTree(Tree[float]):
    """Base binomial tree (abstract — concretes supply the coefficients).

    # C++ parity: ``BinomialTree<T>`` (binomialtree.hpp:38-57).

    Holds the per-step drift, dt, and initial state ``x0``. Concrete
    subclasses set up/down/probability fields and override
    ``underlying`` + ``probability``.

    Subclasses inheriting from this class commit to ``branches = 2``
    (the binomial layout) and to ``descendant(i, j, b) = j + b``.
    """

    branches: int = 2

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
    ) -> None:
        # C++ parity: ``BinomialTree<T>::BinomialTree`` (binomialtree.hpp:42-47).
        # ``Tree::__init__`` records columns = steps + 1.
        super().__init__(columns=steps + 1)
        self._x0: float = process.x0()
        self._dt: float = end / steps
        self._drift_per_step: float = process.drift_1d(0.0, self._x0) * self._dt

    # --- Tree contract — partially defined ---------------------------------

    def size(self, i: int) -> int:
        """Slice ``i`` has ``i + 1`` nodes.

        # C++ parity: ``BinomialTree<T>::size`` (binomialtree.hpp:48-50 inline).
        """
        return i + 1

    def descendant(self, i: int, index: int, branch: int) -> int:
        """Index ``index + branch`` at the next slice.

        # C++ parity: ``BinomialTree<T>::descendant`` (binomialtree.hpp:51-53).

        ``branch == 0`` = down; ``branch == 1`` = up. The arity is
        not checked here — callers (``TreeLattice.stepback``) are expected
        to obey the [0, branches) convention.
        """
        del i
        return index + branch

    # --- inspectors --------------------------------------------------------

    @property
    def x0(self) -> float:
        """Initial state value.

        # C++ parity: protected ``x0_`` (binomialtree.hpp:55).
        """
        return self._x0

    @property
    def dt(self) -> float:
        """Per-step time increment.

        # C++ parity: protected ``dt_`` (binomialtree.hpp:56).
        """
        return self._dt

    @property
    def drift_per_step(self) -> float:
        """Drift per step at the centre state.

        # C++ parity: protected ``driftPerStep_`` (binomialtree.hpp:55).
        """
        return self._drift_per_step


class EqualProbabilitiesBinomialTree(BinomialTree):
    """Base for equal-probabilities (``p = 0.5``) binomial trees.

    # C++ parity: ``EqualProbabilitiesBinomialTree<T>``
    # (binomialtree.hpp:62-78).

    The tree is centred on the *forward* value: with ``j = 2*index - i``
    the underlying is ``x0 * exp(i*driftPerStep + j*up)``. Concrete
    subclasses set ``_up``.
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
    ) -> None:
        # C++ parity: binomialtree.hpp:65-69 — forwards to BinomialTree;
        # ``up_`` is left for the concrete ctor (protected member,
        # binomialtree.hpp:77).
        super().__init__(process, end, steps)
        self._up: float = 0.0

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: ``EqualProbabilitiesBinomialTree<T>::underlying``
        # (binomialtree.hpp:70-74) — exploits the forward-value centring.
        j = 2 * index - i
        return self._x0 * math.exp(i * self._drift_per_step + j * self._up)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: ``EqualProbabilitiesBinomialTree<T>::probability``
        # (binomialtree.hpp:75) — always 0.5.
        del i, index, branch
        return 0.5


class EqualJumpsBinomialTree(BinomialTree):
    """Base for equal-jumps binomial trees (constant log-step ``dx``).

    # C++ parity: ``EqualJumpsBinomialTree<T>`` (binomialtree.hpp:83-101).

    The tree is centred on ``x0``: with ``j = 2*index - i`` the underlying
    is ``x0 * exp(j*dx)``, and the asymmetry of the process is carried by
    ``(pu, pd)`` instead. Concrete subclasses set ``_dx`` / ``_pu`` / ``_pd``.
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
    ) -> None:
        # C++ parity: binomialtree.hpp:86-90 — forwards to BinomialTree;
        # ``dx_`` / ``pu_`` / ``pd_`` are left for the concrete ctor
        # (protected members, binomialtree.hpp:100).
        super().__init__(process, end, steps)
        self._dx: float = 0.0
        self._pu: float = 0.0
        self._pd: float = 0.0

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: ``EqualJumpsBinomialTree<T>::underlying``
        # (binomialtree.hpp:91-95) — equal jump, x0 centring.
        j = 2 * index - i
        return self._x0 * math.exp(j * self._dx)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: ``EqualJumpsBinomialTree<T>::probability``
        # (binomialtree.hpp:96-98).
        del i, index
        return self._pu if branch == 1 else self._pd


class JarrowRudd(EqualProbabilitiesBinomialTree):
    """Jarrow-Rudd equal-probabilities multiplicative binomial tree.

    # C++ parity: ``class JarrowRudd`` (binomialtree.hpp:106-112 +
    # binomialtree.cpp:28-34).

    The drift is removed by the forward centring, so the only coefficient
    is ``up = stdDeviation(0, x0, dt)``.
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,  # kept for C++ signature parity (unused in this builder)
    ) -> None:
        # C++ parity: binomialtree.cpp:28-34. ``strike`` is ignored, exactly
        # as in C++ (the parameter is unnamed there).
        del strike
        super().__init__(process, end, steps)
        # C++: ``up_ = process->stdDeviation(0.0, x0_, dt_)`` — "drift removed".
        self._up = process.std_deviation_1d(0.0, self._x0, self._dt)


class AdditiveEQPBinomialTree(EqualProbabilitiesBinomialTree):
    """Additive equal-probabilities binomial tree.

    # C++ parity: ``class AdditiveEQPBinomialTree`` (binomialtree.hpp:129-137 +
    # binomialtree.cpp:51-59).

    Third-order-accurate up step::

        up = -0.5*drift + 0.5*sqrt(4*variance - 3*drift^2)
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,  # kept for C++ signature parity (unused in this builder)
    ) -> None:
        # C++ parity: binomialtree.cpp:51-59.
        del strike
        super().__init__(process, end, steps)
        self._up = -0.5 * self._drift_per_step + 0.5 * math.sqrt(
            4.0 * process.variance_1d(0.0, self._x0, self._dt)
            - 3.0 * self._drift_per_step * self._drift_per_step
        )


class CoxRossRubinstein(EqualJumpsBinomialTree):
    """CRR multiplicative equal-jumps binomial tree.

    # C++ parity: ``class CoxRossRubinstein`` (binomialtree.hpp:117-124 +
    # binomialtree.cpp:37-48)::

        dx = stdDeviation(0, x0, dt)
        pu = 0.5 + 0.5 * (drift / dx)
        pd = 1 - pu
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,  # kept for C++ signature parity (unused in this builder)
    ) -> None:
        # C++ parity: binomialtree.cpp:37-48. ``strike`` is ignored —
        # the CRR coefficients do not depend on it (matches C++).
        del strike
        super().__init__(process, end, steps)
        self._dx = process.std_deviation_1d(0.0, self._x0, self._dt)
        self._pu = 0.5 + 0.5 * self._drift_per_step / self._dx
        self._pd = 1.0 - self._pu
        qassert.require(self._pu <= 1.0, "negative probability")
        qassert.require(self._pu >= 0.0, "negative probability")


class Trigeorgis(EqualJumpsBinomialTree):
    """Trigeorgis additive-equal-jumps binomial tree.

    # C++ parity: ``class Trigeorgis`` (binomialtree.hpp:142-148 +
    # binomialtree.cpp:62-74)::

        dx = sqrt(variance + drift^2)
        pu = 0.5 + 0.5 * (drift / dx)
        pd = 1 - pu

    Same probability form as CRR; the difference is the jump size, which
    folds the drift into the second moment instead of leaving it out.
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,  # kept for C++ signature parity (unused in this builder)
    ) -> None:
        # C++ parity: binomialtree.cpp:62-74.
        del strike
        super().__init__(process, end, steps)
        self._dx = math.sqrt(
            process.variance_1d(0.0, self._x0, self._dt)
            + self._drift_per_step * self._drift_per_step
        )
        self._pu = 0.5 + 0.5 * self._drift_per_step / self._dx
        self._pd = 1.0 - self._pu
        qassert.require(self._pu <= 1.0, "negative probability")
        qassert.require(self._pu >= 0.0, "negative probability")


class Tian(BinomialTree):
    """Tian third-moment-matching multiplicative binomial tree.

    # C++ parity: ``class Tian`` (binomialtree.hpp:153-168 +
    # binomialtree.cpp:77-96).

    Multiplicative up/down with explicit ``pu/pd``::

        q  = exp(variance_per_step)
        rr = exp(drift_per_step) * sqrt(q)
        up   = 0.5 * rr * q * (q+1 + sqrt(q^2 + 2q - 3))
        down = 0.5 * rr * q * (q+1 - sqrt(q^2 + 2q - 3))
        pu = (rr - down) / (up - down)
        pd = 1 - pu

    Underlying is ``x0 * down^(i - index) * up^index`` (asymmetric
    layout — the tree is not centred on x0).
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,  # kept for C++ signature parity (unused in this builder)
    ) -> None:
        # C++ parity: binomialtree.cpp:77-96.
        del strike
        super().__init__(process, end, steps)
        # C++ parity: q = exp(variance), r = exp(drift) * sqrt(q).
        q = math.exp(process.variance_1d(0.0, self._x0, self._dt))
        rr = math.exp(self._drift_per_step) * math.sqrt(q)
        disc = math.sqrt(q * q + 2.0 * q - 3.0)
        self._up: float = 0.5 * rr * q * (q + 1.0 + disc)
        self._down: float = 0.5 * rr * q * (q + 1.0 - disc)
        self._pu: float = (rr - self._down) / (self._up - self._down)
        self._pd: float = 1.0 - self._pu
        qassert.require(self._pu <= 1.0, "negative probability")
        qassert.require(self._pu >= 0.0, "negative probability")

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: binomialtree.hpp:159-162 — ``x0 * down^(i - index) * up^index``.
        return self._x0 * (self._down ** (i - index)) * (self._up**index)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: binomialtree.hpp:163-165.
        del i, index
        return self._pu if branch == 1 else self._pd


class LeisenReimer(BinomialTree):
    """Leisen-Reimer Peizer-Pratt method-2 inversion binomial tree.

    # C++ parity: ``class LeisenReimer`` (binomialtree.hpp:172-187 +
    # binomialtree.cpp:99-117).

    Forces an odd number of steps (so ``log(K/S0) / d2`` evaluates
    centred), then uses the Peizer-Pratt method-2 inversion of the
    cumulative binomial to converge faster than CRR for European
    options. Recommended default in the C++ test suite.

    The ``strike`` argument is mandatory — d2 depends on it.
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,
    ) -> None:
        # C++ parity: binomialtree.cpp:99-106 — round steps up to odd
        # when even (the PP inversion has its centre at d2=0; with even
        # steps the centre falls between nodes). C++ does the rounding in
        # the base-class initialiser, so ``dt`` uses the rounded count.
        odd_steps = steps if steps % 2 == 1 else steps + 1
        super().__init__(process, end, odd_steps)
        qassert.require(strike > 0.0, "strike must be positive")
        # Variance over the full period (not per step) — C++ uses
        # process->variance(0, x0, end).
        variance = process.variance_1d(0.0, self._x0, end)
        ermqdt = math.exp(self._drift_per_step + 0.5 * variance / odd_steps)
        d2 = (
            math.log(self._x0 / strike) + self._drift_per_step * odd_steps
        ) / math.sqrt(variance)
        self._pu: float = peizer_pratt_method2_inversion(d2, odd_steps)
        self._pd: float = 1.0 - self._pu
        pdash = peizer_pratt_method2_inversion(d2 + math.sqrt(variance), odd_steps)
        self._up: float = ermqdt * pdash / self._pu
        self._down: float = (ermqdt - self._pu * self._up) / (1.0 - self._pu)

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: binomialtree.hpp:178-181 — ``x0 * down^(i - index) * up^index``.
        return self._x0 * (self._down ** (i - index)) * (self._up**index)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: binomialtree.hpp:182-184.
        del i, index
        return self._pu if branch == 1 else self._pd


class Joshi4(BinomialTree):
    """Joshi fourth-order binomial tree.

    # C++ parity: ``class Joshi4`` (binomialtree.hpp:190-206 +
    # binomialtree.cpp:119-157).

    Same shape as :class:`LeisenReimer` — odd steps forced, up/down
    recovered from an up-probability and its shifted counterpart — but
    the up-probability comes from Joshi's fourth-order asymptotic
    expansion in ``1/sqrt(k)`` rather than from the Peizer-Pratt
    inversion of the cumulative binomial.
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,
    ) -> None:
        # C++ parity: binomialtree.cpp:140-157.
        odd_steps = steps if steps % 2 == 1 else steps + 1
        super().__init__(process, end, odd_steps)
        qassert.require(strike > 0.0, "strike must be positive")
        variance = process.variance_1d(0.0, self._x0, end)
        ermqdt = math.exp(self._drift_per_step + 0.5 * variance / odd_steps)
        d2 = (
            math.log(self._x0 / strike) + self._drift_per_step * odd_steps
        ) / math.sqrt(variance)
        k = (odd_steps - 1.0) / 2.0
        self._pu: float = self._compute_up_prob(k, d2)
        self._pd: float = 1.0 - self._pu
        pdash = self._compute_up_prob(k, d2 + math.sqrt(variance))
        self._up: float = ermqdt * pdash / self._pu
        self._down: float = (ermqdt - self._pu * self._up) / (1.0 - self._pu)

    @staticmethod
    def _compute_up_prob(k: float, dj: float) -> float:
        """Joshi's fourth-order up-probability expansion.

        # C++ parity: ``Joshi4::computeUpProb`` (binomialtree.cpp:119-138) —
        # a protected const member there. The four ``p +=`` terms are kept
        # in the C++ order (they are not associative in floating point).
        """
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
        # C++ note: "delete next line to get results for j three tree".
        p += delta / (k * k * k * rootk)
        return p

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: binomialtree.hpp:196-199 — ``x0 * down^(i - index) * up^index``.
        return self._x0 * (self._down ** (i - index)) * (self._up**index)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: binomialtree.hpp:200-202.
        del i, index
        return self._pu if branch == 1 else self._pd


__all__ = [
    "AdditiveEQPBinomialTree",
    "BinomialTree",
    "CoxRossRubinstein",
    "EqualJumpsBinomialTree",
    "EqualProbabilitiesBinomialTree",
    "JarrowRudd",
    "Joshi4",
    "LeisenReimer",
    "Tian",
    "Trigeorgis",
]
