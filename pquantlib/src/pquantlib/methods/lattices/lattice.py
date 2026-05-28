"""Lattice base class — abstract numerical-method interface.

# C++ parity: ql/numericalmethod.hpp (v1.42.1) — ``class Lattice``.
#             Header lives at ``ql/numericalmethod.hpp`` but the class
#             is the canonical base for tree-based lattices in
#             ``ql/methods/lattices/lattice.hpp`` (TreeLattice<Impl>).

C++ ``Lattice`` is a pure-virtual interface used by ``DiscretizedAsset``
to delegate ``initialize`` / ``rollback`` / ``partialRollback`` /
``presentValue`` calls back to the lattice. ``TreeLattice<Impl>``
(CRTP) is the concrete generic implementation; binomial and trinomial
pricers inherit from it.

The Python port mirrors the abstract base only — concrete tree
lattices (``BinomialTree``, ``TrinomialTree``, ``TFLattice``) are
deferred. The class is built around a held ``TimeGrid`` and exposes
the four pure-virtual methods plus ``grid(t) -> Array``.

L5-A stage scope: the abstract base; concretes follow in later stages.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib.methods.lattices.tree import Tree

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.time.time_grid import TimeGrid


class Lattice(Tree[float]):
    """Numerical-method base (tree, finite-differences) over a TimeGrid.

    # C++ parity: ``class Lattice`` (numericalmethod.hpp:37-96). The
    # Python port subclasses ``Tree[float]`` for the four-method
    # ``size`` / ``underlying`` / ``descendant`` / ``probability``
    # contract used by C++ ``TreeLattice<Impl>``; the four high-level
    # methods ``initialize`` / ``rollback`` / ``partialRollback`` /
    # ``presentValue`` plus ``grid(t)`` live on this class.

    The ``Tree`` ancestry diverges from the C++ class layout (C++
    ``Lattice`` does not derive from ``Tree`` — only ``TreeLattice<Impl>``
    composes them). The Python port collapses them because every
    pquantlib lattice we plan to port (binomial / trinomial / TFLattice)
    is tree-based; if a non-tree lattice (e.g. pure FD) appears later
    we can introduce a separate base.
    """

    def __init__(self, time_grid: TimeGrid) -> None:
        # ``Tree.__init__`` records ``columns`` — for a Lattice the
        # number of columns equals the time-grid size.
        # C++ parity: ``Lattice(TimeGrid timeGrid)`` (numericalmethod.hpp:39).
        super().__init__(columns=time_grid.size())
        self._time_grid: TimeGrid = time_grid

    def time_grid(self) -> TimeGrid:
        """Underlying TimeGrid.

        # C++ parity: ``Lattice::timeGrid()`` (numericalmethod.hpp:44).
        """
        return self._time_grid

    # --- abstract contract — high-level lattice interface ----------------

    @abstractmethod
    def initialize(self, asset: object, t: float) -> None:
        """Initialize ``asset`` at time ``t``.

        # C++ parity: ``Lattice::initialize`` (numericalmethod.hpp:58).

        ``asset`` is a ``DiscretizedAsset`` instance; we type it as
        ``object`` here to avoid the import cycle (DiscretizedAsset
        references Lattice and vice versa). Concrete trees should
        type-narrow.
        """

    @abstractmethod
    def rollback(self, asset: object, to_t: float) -> None:
        """Roll ``asset`` back to time ``to_t`` and perform the final
        pre+post adjustment.

        # C++ parity: ``Lattice::rollback`` (numericalmethod.hpp:64).
        """

    @abstractmethod
    def partial_rollback(self, asset: object, to_t: float) -> None:
        """Roll back without the final adjustment.

        # C++ parity: ``Lattice::partialRollback`` (numericalmethod.hpp:84).
        """

    @abstractmethod
    def present_value(self, asset: object) -> float:
        """Present value of ``asset`` (typically Arrow-Debreu weighted).

        # C++ parity: ``Lattice::presentValue`` (numericalmethod.hpp:88).
        """

    @abstractmethod
    def grid(self, t: float) -> Array:
        """Underlying-value grid at time ``t``.

        # C++ parity: ``Lattice::grid(Time)`` (numericalmethod.hpp:93) —
        # marked "this is a smell" in the C++ header. Kept for parity.
        """
