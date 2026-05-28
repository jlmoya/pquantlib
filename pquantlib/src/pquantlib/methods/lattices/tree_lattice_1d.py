"""TreeLattice1D — concrete tree-based 1-D lattice base.

# C++ parity: ql/methods/lattices/lattice.hpp (``class TreeLattice``)
#             + ql/methods/lattices/lattice1d.hpp (``class TreeLattice1D``)
#             (v1.42.1).

C++ uses two layers — ``TreeLattice<Impl>`` (CRTP, holds the generic
``initialize`` / ``rollback`` / ``partialRollback`` / ``presentValue``
machinery and the Arrow-Debreu state-price cache) and
``TreeLattice1D<Impl>`` (also CRTP, supplies ``grid(t)`` from the
1-D underlying). The Python port collapses both into a single class
because Python's dynamic dispatch makes the CRTP indirection
unnecessary.

Concrete subclasses (``BlackScholesLattice``, ``ShortRateTree``) must
provide:

  * ``discount(i, index)`` — per-node one-step discount factor.
  * the four ``Tree`` methods (``size`` / ``underlying`` /
    ``descendant`` / ``probability``).

Optional override: ``stepback(i, values)`` — by default the base
provides a numpy-vectorised stepback (probability-weighted average
of next-slice values, scaled by ``discount(i, j)``). Subclasses
override when they have a tighter formulation (e.g.
``BlackScholesLattice`` uses a constant rate, so the per-step
discount is independent of ``j``).

The state-prices cache ``_state_prices`` is built lazily — when a
caller asks for ``present_value(asset)`` at slice ``i``, the lattice
walks forward from the last computed slice and accumulates
Arrow-Debreu prices. The result is then dotted against
``asset.values`` to produce the present value.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.methods.lattices.lattice import Lattice

if TYPE_CHECKING:
    from pquantlib.time.time_grid import TimeGrid


class TreeLattice1D(Lattice):
    """Tree-based 1-D lattice (binomial / trinomial / etc.).

    # C++ parity: collapsed ``TreeLattice<Impl>`` + ``TreeLattice1D<Impl>``
    # (lattice.hpp:56-92 + lattice1d.hpp:38-53).

    Subclasses set ``branches`` (2 for binomial, 3 for trinomial) and
    implement ``discount(i, index)`` + the four ``Tree`` methods.
    """

    def __init__(self, time_grid: TimeGrid, n_branches: int) -> None:
        # # C++ parity: lattice.hpp:60-67 — n_branches must be > 0.
        qassert.require(n_branches > 0, "there is no zeronomial lattice!")
        super().__init__(time_grid=time_grid)
        self._n_branches: int = n_branches
        # Arrow-Debreu state prices: state_prices[i][j] is the price
        # of an asset paying 1.0 at slice i in state j. Built lazily.
        self._state_prices: list[Array] = [np.ones(1, dtype=np.float64)]
        self._state_prices_limit: int = 0

    # --- contract: subclasses provide --------------------------------------

    @abstractmethod
    def discount(self, i: int, index: int) -> float:
        """Per-node one-step discount factor.

        # C++ parity: pure-virtual ``DiscountFactor TreeLattice<Impl>::discount(i, j)``
        # (lattice.hpp:42).
        """

    # --- inspectors -------------------------------------------------------

    @property
    def n_branches(self) -> int:
        """Number of children per node (2 / 3 / ...).

        # C++ parity: private ``n_`` (lattice.hpp:90).
        """
        return self._n_branches

    def state_prices(self, i: int) -> Array:
        """Arrow-Debreu state prices at slice ``i`` (computed lazily).

        # C++ parity: ``TreeLattice<Impl>::statePrices`` (lattice.hpp:114-118).
        """
        if i > self._state_prices_limit:
            self._compute_state_prices(i)
        return self._state_prices[i]

    def grid(self, t: float) -> Array:
        """Underlying-value grid at time ``t``.

        # C++ parity: ``TreeLattice1D<Impl>::grid`` (lattice1d.hpp:43-49).
        """
        i = self._time_grid.index(t)
        n = self.size(i)
        return np.array([self.underlying(i, j) for j in range(n)], dtype=np.float64)

    # --- Lattice interface -----------------------------------------------

    def initialize(self, asset: object, t: float) -> None:
        """Initialise ``asset`` at time ``t`` on this lattice.

        # C++ parity: ``TreeLattice<Impl>::initialize`` (lattice.hpp:127-131).
        """
        if not isinstance(asset, DiscretizedAsset):
            raise TypeError("asset must be a DiscretizedAsset")
        i = self._time_grid.index(t)
        asset.set_time(t)
        asset.set_method(self)
        asset.reset(self.size(i))

    def partial_rollback(self, asset: object, to_t: float) -> None:
        """Roll back ``asset`` without applying the final adjustment.

        # C++ parity: ``TreeLattice<Impl>::partialRollback`` (lattice.hpp:140-164).
        """
        if not isinstance(asset, DiscretizedAsset):
            raise TypeError("asset must be a DiscretizedAsset")

        from_t = asset.time
        if close_enough(from_t, to_t):
            return
        qassert.require(
            from_t > to_t,
            f"cannot roll the asset back to {to_t} (it is already at t = {from_t})",
        )

        i_from = self._time_grid.index(from_t)
        i_to = self._time_grid.index(to_t)

        for i in range(i_from - 1, i_to - 1, -1):
            new_values = self.stepback(i, asset.values)
            asset.set_time(self._time_grid[i])
            asset.set_values(new_values)
            # Skip the very last adjustment (the rollback wrapper applies it).
            if i != i_to:
                asset.adjust_values()

    def rollback(self, asset: object, to_t: float) -> None:
        """Roll back ``asset`` to ``to_t`` and apply the final adjustment.

        # C++ parity: ``TreeLattice<Impl>::rollback`` (lattice.hpp:134-137).
        """
        if not isinstance(asset, DiscretizedAsset):
            raise TypeError("asset must be a DiscretizedAsset")
        self.partial_rollback(asset, to_t)
        asset.adjust_values()

    def present_value(self, asset: object) -> float:
        """Present value via ``DotProduct(values, state_prices)``.

        # C++ parity: ``TreeLattice<Impl>::presentValue`` (lattice.hpp:121-124).
        """
        if not isinstance(asset, DiscretizedAsset):
            raise TypeError("asset must be a DiscretizedAsset")
        i = self._time_grid.index(asset.time)
        return float(np.dot(asset.values, self.state_prices(i)))

    # --- stepback (default — subclasses may override) ---------------------

    def stepback(self, i: int, values: Array) -> Array:
        """One-step backward induction with per-node discount.

        # C++ parity: ``TreeLattice<Impl>::stepback`` (lattice.hpp:166-179).

        For each node ``j`` at slice ``i``, the new value is the
        ``discount(i, j)`` times the probability-weighted sum of
        the children at slice ``i+1``.
        """
        n_i = self.size(i)
        new_values = np.zeros(n_i, dtype=np.float64)
        for j in range(n_i):
            v = 0.0
            for branch in range(self._n_branches):
                v += self.probability(i, j, branch) * values[
                    self.descendant(i, j, branch)
                ]
            new_values[j] = v * self.discount(i, j)
        return new_values

    # --- state-prices machinery ------------------------------------------

    def _compute_state_prices(self, until: int) -> None:
        """Walk forward and fill ``_state_prices`` up to slice ``until``.

        # C++ parity: ``TreeLattice<Impl>::computeStatePrices`` (lattice.hpp:97-111).
        """
        for i in range(self._state_prices_limit, until):
            # Allocate the next slice (filled below).
            next_size = self.size(i + 1)
            next_prices = np.zeros(next_size, dtype=np.float64)
            cur_prices = self._state_prices[i]
            for j in range(self.size(i)):
                disc = self.discount(i, j)
                sp = float(cur_prices[j])
                for branch in range(self._n_branches):
                    k = self.descendant(i, j, branch)
                    next_prices[k] += (
                        sp * disc * self.probability(i, j, branch)
                    )
            self._state_prices.append(next_prices)
        self._state_prices_limit = until


__all__ = ["TreeLattice1D"]
