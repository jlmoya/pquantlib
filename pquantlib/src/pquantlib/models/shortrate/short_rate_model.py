"""ShortRateModel — abstract base for short-rate models.

# C++ parity: ``class ShortRateModel`` in ql/models/model.hpp:139-145
# (v1.42.1). The C++ class is a thin CalibratedModel subclass that adds
# a single pure-virtual ``tree(TimeGrid)`` hook.

The Python port keeps the same shape; ``tree`` is declared abstract so
that subclasses MUST provide it. L4-D's ``TwoFactorModel.tree`` is
currently deferred (no Lattice2D port yet) — the override raises
``LibraryException`` with a clear message.

Cross-branch coexistence note (L4-D vs L4-B):
The single-factor side of L4 (Vasicek / HullWhite / BlackKarasinski)
lives on the L4-B branch. That branch defines ``ShortRateModel``
independently. Both definitions are *structurally* identical (one
abstract ``tree(grid)`` method). When the two branches merge, keep
this file as the single source of truth and drop the L4-B copy.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib.exceptions import LibraryException
from pquantlib.models.model import CalibratedModel

if TYPE_CHECKING:
    from pquantlib.time.time_grid import TimeGrid


class ShortRateModel(CalibratedModel):
    """Abstract short-rate model — CalibratedModel + tree(grid).

    # C++ parity: ``class ShortRateModel : public CalibratedModel`` in
    # model.hpp:139-145 (v1.42.1).
    """

    def __init__(self, n_arguments: int) -> None:
        # C++ parity: model.hpp:143 — only ctor argument is ``Size``.
        super().__init__(n_arguments=n_arguments)

    @abstractmethod
    def tree(self, grid: TimeGrid) -> object:
        """Build a lattice tree on the supplied TimeGrid.

        # C++ parity: ``ShortRateModel::tree`` (pure virtual).

        Returns: a Lattice (1-D for OneFactorModel, 2-D for TwoFactorModel).
        Python returns ``object`` rather than a concrete Lattice type
        because the Lattice / TreeLattice1D / TreeLattice2D hierarchy is
        deferred per the L4 carve-outs; subclasses raise
        ``LibraryException`` until those tree types are ported.
        """


def _tree_not_implemented(model_name: str) -> LibraryException:
    """Helper for subclasses that defer ``tree`` pending Lattice2D port.

    # C++ parity: none — Python-only helper that gives a consistent
    # error message across all deferring subclasses.
    """
    return LibraryException(
        f"{model_name}.tree() requires the Lattice / TrinomialTree hierarchy, "
        "which is deferred per L4 carve-out"
    )


__all__ = ["ShortRateModel", "_tree_not_implemented"]
