"""BasisIncompleteOrdered — incremental ordered orthonormal basis (Gram-Schmidt).

# C++ parity: ql/math/matrixutilities/basisincompleteordered.{hpp,cpp}
# (v1.42.1).

Builds an orthonormal basis one vector at a time: each candidate is projected
onto the orthogonal complement of the basis so far, then normalised (or
rejected if it is linearly dependent). Used by the max-homogeneity caplet
calibration to rotate into a coordinate frame aligned with the cylinder centre
and target.

The companion ``OrthogonalProjections`` (originally noted as out of scope for
the matrices test-suite) is ported alongside it: given a collection of vectors
``w_i`` it finds vectors ``x_i`` such that ``x_i`` is orthogonal to ``w_j`` for
``i != j`` and ``<x_i, w_i> = <w_i, w_i>``. This is the orthogonalisation engine
behind ``OrthogonalizedBumpFinder`` (W11-D pathwise greeks).
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


class OrthogonalProjections:
    """Project each vector onto the complement of the span of the others.

    # C++ parity: OrthogonalProjections (basisincompleteordered.{hpp,cpp}).

    Given the rows ``w_i`` of ``original_vectors``, computes vectors ``x_i``
    such that ``x_i`` is orthogonal to ``w_j`` for ``i != j`` and
    ``<x_i, w_i> = <w_i, w_i>``. Vectors whose length-multiplier exceeds
    ``multiplier_cut_off`` (or whose projection norm falls below ``tolerance``)
    are flagged invalid and skipped.
    """

    def __init__(
        self,
        original_vectors: Matrix,
        multiplier_cut_off: float,
        tolerance: float,
    ) -> None:
        # C++ parity: OrthogonalProjections::OrthogonalProjections.
        self._original_vectors = np.asarray(original_vectors, dtype=np.float64).copy()
        self._multiplier_cutoff = multiplier_cut_off
        self._number_vectors = self._original_vectors.shape[0]
        self._dimension = self._original_vectors.shape[1]
        # validVectors_(true, rows) — opposite order from std::vector ctor
        self._valid_vectors = [True] * self._number_vectors
        self._projected_vectors: list[list[float]] = []

        ortho = np.zeros((self._number_vectors, self._dimension), dtype=np.float64)
        current_vector = [0.0] * self._dimension

        for j in range(self._number_vectors):
            if self._valid_vectors[j]:
                # build an orthonormal basis not containing j
                for k in range(self._number_vectors):
                    for m in range(self._dimension):
                        ortho[k][m] = self._original_vectors[k][m]

                    if k != j and self._valid_vectors[k]:
                        for ll in range(k):
                            if self._valid_vectors[ll] and ll != j:
                                dot = float(np.dot(ortho[k], ortho[ll]))
                                for nn in range(self._dimension):
                                    ortho[k][nn] -= dot * ortho[ll][nn]

                        norm_before = math.sqrt(float(np.dot(ortho[k], ortho[k])))
                        if norm_before < tolerance:
                            self._valid_vectors[k] = False
                        else:
                            recip = 1.0 / norm_before
                            for m in range(self._dimension):
                                ortho[k][m] *= recip

                # we now have an o.n. basis for everything except j
                prev_norm_squared = float(
                    np.dot(self._original_vectors[j], self._original_vectors[j])
                )

                for r in range(self._number_vectors):
                    if self._valid_vectors[r] and r != j:
                        dot = float(np.dot(ortho[j], ortho[r]))
                        for ss in range(self._dimension):
                            ortho[j][ss] -= dot * ortho[r][ss]

                projection_on_original = float(
                    np.dot(self._original_vectors[j], ortho[j])
                )
                # C++ parity: ``prevNormSquared / projectionOnOriginalDirection``
                # with NO guard — C++ relies on IEEE-754, so a zero projection
                # (a linearly-dependent input vector) yields +/-inf or nan and
                # ``fabs(.) < cutoff`` is false, discarding the vector. Python's
                # float ``/`` would raise ZeroDivisionError, so divide under
                # numpy float semantics to reproduce the C++ inf/nan behaviour.
                with np.errstate(divide="ignore", invalid="ignore"):
                    size_multiplier = float(
                        np.float64(prev_norm_squared) / np.float64(projection_on_original)
                    )

                if abs(size_multiplier) < self._multiplier_cutoff:
                    for t in range(self._dimension):
                        current_vector[t] = ortho[j][t] * size_multiplier
                else:
                    self._valid_vectors[j] = False

            self._projected_vectors.append(list(current_vector))

        self._number_valid_vectors = sum(1 for v in self._valid_vectors if v)

    def valid_vectors(self) -> list[bool]:
        # C++ parity: OrthogonalProjections::validVectors.
        return self._valid_vectors

    def get_vector(self, index: int) -> list[float]:
        # C++ parity: OrthogonalProjections::GetVector.
        return self._projected_vectors[index]

    def number_valid_vectors(self) -> int:
        # C++ parity: OrthogonalProjections::numberValidVectors.
        return self._number_valid_vectors
