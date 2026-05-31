"""BasisIncompleteOrdered — incremental ordered orthonormal basis (Gram-Schmidt).

# C++ parity: ql/math/matrixutilities/basisincompleteordered.{hpp,cpp}
# (v1.42.1).

Builds an orthonormal basis one vector at a time: each candidate is projected
onto the orthogonal complement of the basis so far, then normalised (or
rejected if it is linearly dependent). Used by the max-homogeneity caplet
calibration to rotate into a coordinate frame aligned with the cylinder centre
and target. Only this class is ported; the companion ``OrthogonalProjections``
(used only by the matrices test-suite) is out of scope.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix


class BasisIncompleteOrdered:
    """Incrementally-built ordered orthonormal basis.

    # C++ parity: BasisIncompleteOrdered.
    """

    def __init__(self, euclidean_dimension: int) -> None:
        # C++ parity: BasisIncompleteOrdered(Size euclideanDimension).
        self._euclidean_dimension = euclidean_dimension
        self._current_basis: list[np.ndarray] = []

    def add_vector(self, new_vector1: np.ndarray) -> bool:
        """Add a vector; return ``True`` iff it was linearly independent.

        # C++ parity: BasisIncompleteOrdered::addVector.
        """
        nv = np.asarray(new_vector1, dtype=np.float64)
        qassert.require(
            nv.shape[0] == self._euclidean_dimension,
            "missized vector passed to BasisIncompleteOrdered::addVector",
        )
        new_vector = nv.copy()

        if len(self._current_basis) == self._euclidean_dimension:
            return False

        for current_basi in self._current_basis:
            inner_prod = float(np.dot(new_vector, current_basi))
            for k in range(self._euclidean_dimension):
                new_vector[k] -= inner_prod * current_basi[k]

        norm = math.sqrt(float(np.dot(new_vector, new_vector)))
        if norm < 1e-12:  # maybe this should be a tolerance
            return False

        for ll in range(self._euclidean_dimension):
            new_vector[ll] /= norm

        self._current_basis.append(new_vector)
        return True

    def basis_size(self) -> int:
        # C++ parity: BasisIncompleteOrdered::basisSize.
        return len(self._current_basis)

    def euclidean_dimension(self) -> int:
        # C++ parity: BasisIncompleteOrdered::euclideanDimension.
        return self._euclidean_dimension

    def get_basis_as_rows_in_matrix(self) -> Matrix:
        # C++ parity: BasisIncompleteOrdered::getBasisAsRowsInMatrix.
        rows = len(self._current_basis)
        basis = np.zeros((rows, self._euclidean_dimension), dtype=np.float64)
        for i in range(rows):
            for j in range(self._euclidean_dimension):
                basis[i][j] = self._current_basis[i][j]
        return basis
