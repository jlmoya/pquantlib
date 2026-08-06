"""TreeLattice — tree-based lattice-method base class.

# C++ parity: ql/methods/lattices/lattice.hpp (v1.43) —
#             ``template <class Impl> class TreeLattice``.

C++ makes this a CRTP template: ``TreeLattice<Impl>`` derives from
``Lattice`` and ``CuriouslyRecurringTemplate<Impl>``, and every call
to ``size`` / ``discount`` / ``descendant`` / ``probability`` /
``stepback`` is routed through ``this->impl()`` so the compiler can
inline the derived-class implementation. Python's method lookup is
already dynamic, so the port drops the CRTP indirection and keeps the
class hierarchy shape:

    Lattice
      └── TreeLattice          (this module — generic machinery)
            ├── TreeLattice1D  (adds ``grid(t)`` from ``underlying``)
            └── TreeLattice2D  (two trees + a correlation correction)

Derived classes must supply, per the C++ header's documented contract
(lattice.hpp:39-52):

  * ``discount(i, index)`` — per-node one-step discount factor;
  * ``size(i)`` / ``descendant(i, index, branch)`` /
    ``probability(i, index, branch)`` — the ``Tree`` contract;

and may override ``stepback(i, values)``.

The Arrow-Debreu state-price cache ``_state_prices`` is built lazily:
when a caller asks for ``present_value(asset)`` at slice ``i``, the
lattice walks forward from the last computed slice, accumulating state
prices. The result is then dotted against ``asset.values``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.closeness import close_enough
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.methods.lattices.lattice import Lattice

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.time.time_grid import TimeGrid


class TreeLattice(Lattice):
    """Tree-based lattice able to roll a discretized asset back with discount.

    # C++ parity: ``TreeLattice<Impl>`` (lattice.hpp:56-92 + the template
    # definitions at lattice.hpp:97-179).

    ``n_branches`` is the C++ ``n_`` constructor argument — the number of
    children each node has (2 for binomial, 3 for trinomial, 9 for the
    two-factor 2-D lattice).
    """

    def __init__(self, time_grid: TimeGrid, n_branches: int) -> None:
        # C++ parity: lattice.hpp:60-66 — n must be > 0, and the state-price
        # cache starts as a single slice holding Array(1, 1.0).
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

        # C++ parity: the ``DiscountFactor discount(Size i, Size index) const``
        # entry of the Impl contract documented at lattice.hpp:42.
        """

    # --- inspectors -------------------------------------------------------

    @property
    def n_branches(self) -> int:
        """Number of children per node (2 / 3 / 9 / ...).

        # C++ parity: private ``n_`` (lattice.hpp:90).
        """
        return self._n_branches

    def state_prices(self, i: int) -> Array:
        """Arrow-Debreu state prices at slice ``i`` (computed lazily).

        # C++ parity: ``TreeLattice<Impl>::statePrices`` (lattice.hpp:113-118).
        """
        if i > self._state_prices_limit:
            self._compute_state_prices(i)
        return self._state_prices[i]

    # --- Lattice interface -----------------------------------------------

    def initialize(self, asset: object, t: float) -> None:
        """Initialise ``asset`` at time ``t`` on this lattice.

        # C++ parity: ``TreeLattice<Impl>::initialize`` (lattice.hpp:126-131).

        C++ sets only ``asset.time()`` and calls ``asset.reset(size(i))``;
        the owning ``DiscretizedAsset::initialize`` is what assigns the
        method. The port additionally assigns it here so that calling the
        lattice's ``initialize`` directly leaves the asset usable.
        """
        if not isinstance(asset, DiscretizedAsset):
            raise TypeError("asset must be a DiscretizedAsset")
        i = self._time_grid.index(t)
        asset.set_time(t)
        asset.set_method(self)
        asset.reset(self.size(i))

    def partial_rollback(self, asset: object, to_t: float) -> None:
        """Roll back ``asset`` without applying the final adjustment.

        # C++ parity: ``TreeLattice<Impl>::partialRollback`` (lattice.hpp:139-164).
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

        # C++ parity: ``TreeLattice<Impl>::rollback`` (lattice.hpp:133-137).
        """
        if not isinstance(asset, DiscretizedAsset):
            raise TypeError("asset must be a DiscretizedAsset")
        self.partial_rollback(asset, to_t)
        asset.adjust_values()

    def present_value(self, asset: object) -> float:
        """Present value via ``DotProduct(values, state_prices)``.

        # C++ parity: ``TreeLattice<Impl>::presentValue`` (lattice.hpp:120-124).
        """
        if not isinstance(asset, DiscretizedAsset):
            raise TypeError("asset must be a DiscretizedAsset")
        i = self._time_grid.index(asset.time)
        return float(np.dot(asset.values, self.state_prices(i)))

    # --- stepback (default — subclasses may override) ---------------------

    def stepback(self, i: int, values: Array) -> Array:
        """One-step backward induction with per-node discount.

        # C++ parity: ``TreeLattice<Impl>::stepback`` (lattice.hpp:166-179).

        For each node ``j`` at slice ``i``, the new value is
        ``discount(i, j)`` times the probability-weighted sum of the
        children at slice ``i+1``.
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


__all__ = ["TreeLattice"]
