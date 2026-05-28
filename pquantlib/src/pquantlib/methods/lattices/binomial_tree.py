"""BinomialTree concrete trees — CRR, JarrowRudd, Tian, LeisenReimer.

# C++ parity: ql/methods/lattices/binomialtree.{hpp,cpp} (v1.42.1).

The C++ class hierarchy is

    Tree<Impl> (CRTP) <- BinomialTree<Impl>
                          <- EqualProbabilitiesBinomialTree<Impl>
                             <- JarrowRudd
                             <- AdditiveEQPBinomialTree   (deferred — L5+ carry-over)
                          <- EqualJumpsBinomialTree<Impl>
                             <- CoxRossRubinstein
                             <- Trigeorgis                 (deferred — L5+ carry-over)
                          <- Tian
                          <- LeisenReimer
                          <- Joshi4                        (deferred — L5+ carry-over)

The Python port collapses the CRTP indirection (Python's dynamic
dispatch handles it natively). All concrete trees inherit from
:class:`~pquantlib.methods.lattices.tree.Tree` with ``T = float`` and
provide ``size`` / ``underlying`` / ``descendant`` / ``probability``.

We deliberately keep the four most-used concretes (CRR, JarrowRudd,
Tian, LeisenReimer). The L3-D ``BinomialVanillaEngine`` continues to
use its own inline coefficient-builder (it pre-computes only the
terminal slice + rolls back with a tight numpy loop, faster than
going through the Tree API for vanilla options).  These concrete
trees exist for the more general lattice path (L5-B ShortRateTree /
TreeLattice1D consumers, future basket/American multi-asset trees).

# C++ parity notes:
#
# * The C++ ``BinomialTree`` ctor takes ``steps`` and stores
#   ``columns = steps + 1``. Our ``columns()`` (inherited from
#   :class:`Tree`) matches.
# * Underlying-value formulas follow the C++ exactly — the
#   ``EqualProbabilitiesBinomialTree`` and ``EqualJumpsBinomialTree``
#   sub-bases use slightly different parameterisations (drift-centred
#   vs jump-centred); we inline those formulas in the concrete classes
#   for readability rather than introducing two intermediate Python
#   bases (the saving is one helper class, the cost is two extra
#   docstrings).
# * ``descendant(i, index, branch) = index + branch`` for every
#   binomial tree (branch 0 = down, branch 1 = up — same orientation
#   as the L3-D BinomialVanillaEngine).
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
    """Base binomial tree (abstract — concretes set ``_up`` and ``_down``).

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
        not checked here — callers (``Lattice.stepback``) are expected
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


class CoxRossRubinstein(BinomialTree):
    """CRR multiplicative equal-jumps binomial tree.

    # C++ parity: ``class CoxRossRubinstein`` (binomialtree.hpp:117-124 +
    # binomialtree.cpp:37-48).

    Equal-jump (constant log-multiplier per step) parameterisation:

        dx = sigma * sqrt(dt)
        up   = exp(+dx)
        down = exp(-dx)
        pu   = 0.5 + 0.5 * (drift / dx)
        pd   = 1 - pu

    The underlying-formula relies on the jump-centred layout
    (``x = x0 * exp(j * dx)`` where ``j = 2*index - i``).
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
        super().__init__(process, end, steps)
        # C++: ``dx_ = process.stdDeviation(0, x0, dt)``.
        self._dx: float = process.std_deviation_1d(0.0, self._x0, self._dt)
        self._pu: float = 0.5 + 0.5 * self._drift_per_step / self._dx
        self._pd: float = 1.0 - self._pu
        qassert.require(0.0 <= self._pu <= 1.0, "negative probability")

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: ``EqualJumpsBinomialTree<T>::underlying``
        # (binomialtree.hpp:91-95) — jump-centred formula.
        j = 2 * index - i
        return self._x0 * math.exp(j * self._dx)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: ``EqualJumpsBinomialTree<T>::probability``
        # (binomialtree.hpp:96-98).
        del i, index
        return self._pu if branch == 1 else self._pd


class JarrowRudd(BinomialTree):
    """Jarrow-Rudd equal-probabilities multiplicative binomial tree.

    # C++ parity: ``class JarrowRudd`` (binomialtree.hpp:106-112 +
    # binomialtree.cpp:28-34).

    Equal-probabilities (pu = pd = 0.5), drift-centred up factor:

        up   = sigma * sqrt(dt)        (as an additive log-shift)
        underlying = x0 * exp(i * drift + j * up)

    where ``j = 2*index - i`` is the standard up/down offset.
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,  # kept for C++ signature parity (unused in this builder)
    ) -> None:
        # C++ parity: binomialtree.cpp:28-34.
        super().__init__(process, end, steps)
        self._up: float = process.std_deviation_1d(0.0, self._x0, self._dt)

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: ``EqualProbabilitiesBinomialTree<T>::underlying``
        # (binomialtree.hpp:70-74).
        j = 2 * index - i
        return self._x0 * math.exp(i * self._drift_per_step + j * self._up)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: ``EqualProbabilitiesBinomialTree<T>::probability``
        # (binomialtree.hpp:75) — always 0.5.
        del i, index, branch
        return 0.5


class Tian(BinomialTree):
    """Tian third-moment-matching multiplicative binomial tree.

    # C++ parity: ``class Tian`` (binomialtree.hpp:151-168 +
    # binomialtree.cpp:77-96).

    Multiplicative up/down with explicit ``pu/pd``:

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
        super().__init__(process, end, steps)
        # C++ variance per step = process.variance(0, x0, dt).
        variance_per_step = (
            process.std_deviation_1d(0.0, self._x0, self._dt) ** 2
        )
        # C++ parity: q = exp(variance), r = exp(drift) * sqrt(q).
        q = math.exp(variance_per_step)
        rr = math.exp(self._drift_per_step) * math.sqrt(q)
        disc = math.sqrt(q * q + 2.0 * q - 3.0)
        self._up: float = 0.5 * rr * q * (q + 1.0 + disc)
        self._down: float = 0.5 * rr * q * (q + 1.0 - disc)
        self._pu: float = (rr - self._down) / (self._up - self._down)
        self._pd: float = 1.0 - self._pu
        qassert.require(0.0 <= self._pu <= 1.0, "negative probability")

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: binomialtree.hpp:159-162 — ``x0 * down^(i - index) * up^index``.
        return self._x0 * (self._down ** (i - index)) * (self._up**index)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: binomialtree.hpp:163-165.
        del i, index
        return self._pu if branch == 1 else self._pd


class LeisenReimer(BinomialTree):
    """Leisen-Reimer Peizer-Pratt method-2 inversion binomial tree.

    # C++ parity: ``class LeisenReimer`` (binomialtree.hpp:170-187 +
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
        qassert.require(strike > 0.0, "strike must be positive")
        # # C++ parity: binomialtree.cpp:99-103 — round steps up to odd
        # when even (the PP inversion has its centre at d2=0; with even
        # steps the centre falls between nodes).
        odd_steps = steps if steps % 2 == 1 else steps + 1
        super().__init__(process, end, odd_steps)
        # Variance over the full period (not per step) — C++ uses
        # process.variance(0, x0, end).
        variance = process.std_deviation_1d(0.0, self._x0, end) ** 2
        ermqdt = math.exp(
            self._drift_per_step + 0.5 * variance / odd_steps
        )
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


__all__ = [
    "BinomialTree",
    "CoxRossRubinstein",
    "JarrowRudd",
    "LeisenReimer",
    "Tian",
]
