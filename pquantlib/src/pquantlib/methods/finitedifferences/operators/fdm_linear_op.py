"""FdmLinearOp — abstract linear operator on a FD grid.

# C++ parity: ql/methods/finitedifferences/operators/fdmlinearop.hpp
# (v1.42.1).

A linear operator maps an ``Array`` of values on the grid to another
``Array`` (``apply``). Concrete implementations (``TripleBandLinearOp``
and its derived ops ``FirstDerivativeOp`` / ``SecondDerivativeOp`` /
``FdmBlackScholesOp``) provide an efficient banded representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scipy.sparse import csr_matrix  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib.math.array import Array


class FdmLinearOp(ABC):
    """Abstract linear operator on a FD grid.

    # C++ parity: ``class FdmLinearOp``.
    """

    @abstractmethod
    def apply(self, r: Array) -> Array:
        """Return ``L @ r`` where ``L`` is the operator's matrix."""

    @abstractmethod
    def to_matrix(self) -> csr_matrix:
        """Return the operator as a sparse matrix.

        # C++ parity: ``FdmLinearOp::toMatrix`` returns a
        # ``QuantLib::SparseMatrix``. The Python port uses
        # scipy.sparse CSR for compatibility with scipy's
        # iterative + direct solvers.
        """


__all__ = ["FdmLinearOp"]
