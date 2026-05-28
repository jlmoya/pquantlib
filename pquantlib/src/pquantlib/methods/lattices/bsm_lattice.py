"""BlackScholesLattice — binomial lattice for vanilla equity options.

# C++ parity: ql/methods/lattices/bsmlattice.hpp (v1.42.1) —
#             ``template <class T> class BlackScholesLattice``.

Wraps a binomial :class:`BinomialTree` in a :class:`TreeLattice1D`
under a constant risk-free rate. The constant rate means every node
uses the same discount factor ``discount = exp(-r * dt)``, so the
``stepback`` reduces to a single scalar multiply per child pair.

C++ uses ``stepback`` directly inside the engine; we override the
default in the base ``TreeLattice1D`` to use the constant-rate
optimisation (one vectorised numpy operation per slice vs the
generic per-node loop).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib.math.array import Array
from pquantlib.methods.lattices.tree_lattice_1d import TreeLattice1D
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.methods.lattices.binomial_tree import BinomialTree


class BlackScholesLattice(TreeLattice1D):
    """Binomial lattice approximating a Black-Scholes-Merton model.

    # C++ parity: ``class BlackScholesLattice<T>`` (bsmlattice.hpp:35-66 +
    # bsmlattice.hpp:71-87 inline definitions).

    Builds an internal ``TimeGrid(end, steps)``. The held
    :class:`BinomialTree` provides ``underlying`` /
    ``descendant`` / ``probability``; this class supplies the
    constant per-step discount factor and the constant-rate stepback.
    """

    def __init__(
        self,
        tree: BinomialTree,
        risk_free_rate: float,
        end: float,
        steps: int,
    ) -> None:
        # # C++ parity: bsmlattice.hpp:72-80 — base ctor takes
        # ``TimeGrid(end, steps), 2`` (binomial = 2 branches).
        super().__init__(time_grid=TimeGrid.regular(end=end, steps=steps), n_branches=2)
        self._tree: BinomialTree = tree
        self._risk_free_rate: float = float(risk_free_rate)
        self._dt: float = end / steps
        self._discount: float = math.exp(-self._risk_free_rate * self._dt)
        # # C++ parity: bsmlattice.hpp:80 — pd / pu read from
        # tree.probability(0, 0, 0/1). The CRR/JR/Tian/LR trees all
        # carry constant probabilities so we cache them once.
        self._pd: float = tree.probability(0, 0, 0)
        self._pu: float = tree.probability(0, 0, 1)

    # --- inspectors -------------------------------------------------------

    def risk_free_rate(self) -> float:
        """Constant risk-free rate.

        # C++ parity: ``riskFreeRate()`` (bsmlattice.hpp:43).
        """
        return self._risk_free_rate

    def dt(self) -> float:
        """Per-step time increment.

        # C++ parity: ``dt()`` (bsmlattice.hpp:44).
        """
        return self._dt

    # --- Tree contract delegated to the held BinomialTree -----------------

    def size(self, i: int) -> int:
        # C++ parity: ``size(i)`` (bsmlattice.hpp:45).
        return self._tree.size(i)

    def discount(self, i: int, index: int) -> float:
        """Constant discount factor (independent of ``(i, index)``).

        # C++ parity: ``DiscountFactor discount(Size, Size)`` (bsmlattice.hpp:46-47).
        """
        del i, index
        return self._discount

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: bsmlattice.hpp:51-53.
        return self._tree.underlying(i, index)

    def descendant(self, i: int, index: int, branch: int) -> int:
        # C++ parity: bsmlattice.hpp:54-56.
        return self._tree.descendant(i, index, branch)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: bsmlattice.hpp:57-59.
        return self._tree.probability(i, index, branch)

    # --- constant-rate stepback override -----------------------------------

    def stepback(self, i: int, values: Array) -> Array:
        """Vectorised constant-rate stepback.

        # C++ parity: ``BlackScholesLattice<T>::stepback`` (bsmlattice.hpp:82-87).

        For each node ``j`` at slice ``i``:
            new[j] = (pd * values[j] + pu * values[j+1]) * discount
        """
        n_i = self.size(i)
        # values has length n_i + 1 (n_{i+1} = i+2 = (i+1)+1).
        # values[0..n_i] and values[1..n_i+1] are the down/up children.
        return (
            self._pd * values[0:n_i] + self._pu * values[1 : n_i + 1]
        ) * self._discount


__all__ = ["BlackScholesLattice"]
