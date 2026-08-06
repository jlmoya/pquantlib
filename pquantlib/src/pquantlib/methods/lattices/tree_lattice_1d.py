"""TreeLattice1D — one-dimensional tree-based lattice.

# C++ parity: ql/methods/lattices/lattice1d.hpp (v1.43) —
#             ``template <class Impl> class TreeLattice1D``.

C++ layers the lattice machinery in two: ``TreeLattice<Impl>``
(:mod:`pquantlib.methods.lattices.tree_lattice`) carries the generic
``initialize`` / ``rollback`` / ``partialRollback`` / ``presentValue``
machinery plus the Arrow-Debreu state-price cache, and
``TreeLattice1D<Impl>`` adds exactly two members on top: ``grid(t)``,
which materialises the whole underlying row at time ``t``, and a
``underlying(i, index)`` forwarder to the implementation. This module
mirrors that split — the CRTP indirection is dropped (Python's method
lookup is already dynamic) but the class layering is the C++ one.

Concrete subclasses (``BlackScholesLattice``, ``ShortRateTree``) must
provide:

  * ``discount(i, index)`` — per-node one-step discount factor;
  * the four ``Tree`` methods (``size`` / ``underlying`` /
    ``descendant`` / ``probability``).

Optional override: ``stepback(i, values)`` — :class:`TreeLattice`
provides a generic probability-weighted stepback; subclasses override
when they have a tighter formulation (e.g. ``BlackScholesLattice``
uses a constant rate, so the per-step discount is independent of
``j``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib.methods.lattices.tree_lattice import TreeLattice

if TYPE_CHECKING:
    from pquantlib.math.array import Array


class TreeLattice1D(TreeLattice):
    """Tree-based 1-D lattice (binomial / trinomial / etc.).

    # C++ parity: ``TreeLattice1D<Impl>`` (lattice1d.hpp:38-53).

    Subclasses set ``branches`` (2 for binomial, 3 for trinomial) and
    implement ``discount(i, index)`` + the four ``Tree`` methods.
    """

    def grid(self, t: float) -> Array:
        """Underlying-value grid at time ``t``.

        # C++ parity: ``TreeLattice1D<Impl>::grid`` (lattice1d.hpp:43-49).
        """
        i = self._time_grid.index(t)
        n = self.size(i)
        return np.array([self.underlying(i, j) for j in range(n)], dtype=np.float64)


__all__ = ["TreeLattice1D"]
