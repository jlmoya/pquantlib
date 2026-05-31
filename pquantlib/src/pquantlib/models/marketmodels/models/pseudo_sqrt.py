"""rank_reduced_sqrt — spectral rank-reduced pseudo-square-root of a matrix.

# C++ parity: ql/math/matrixutilities/pseudosqrt.{hpp,cpp} (v1.42.1) —
#             ``rankReducedSqrt`` + ``normalizePseudoRoot``.

Given a symmetric (covariance) matrix ``M``, returns a rectangular ``B`` such
that ``B @ B.T`` approximates ``M`` using at most ``max_rank`` factors. The
C++ implementation does a principal-component (spectral) analysis:

1. ``SymmetricSchurDecomposition`` (Jacobi) — eigenvalues sorted **decreasing**.
2. Salvaging: ``None`` (require ``min eig >= -1e-16``), ``Spectral`` (clip
   negatives to 0), or ``Higham``.
3. Factor reduction: retain the leading factors whose cumulative eigenvalue
   reaches ``componentRetainedPercentage`` of the total, capped at ``max_rank``
   (at least 1).
4. ``B = Q @ diag(sqrt(eig))`` over the retained factors, then
   ``normalizePseudoRoot`` rescales each row so ``B[i] · B[i] == M[i][i]``
   exactly.

This W10-A port implements the ``None`` and ``Spectral`` salvaging arms (the
only ones the concrete market models use — ``FlatVol`` / ``AbcdVol`` call
``rankReducedSqrt(cov, numberOfFactors, 1.0, SalvagingAlgorithm::None)``). The
``Higham`` arm (a nearest-correlation-matrix iteration) is not needed by any
W10 consumer and is deferred.

Documented divergences from C++:

* **Eigensolver: spectral vs Jacobi.** C++ uses a bespoke
  ``SymmetricSchurDecomposition`` (cyclic Jacobi rotations); we use
  ``numpy.linalg.eigh`` (LAPACK ``?syevd``, divide-and-conquer). Both compute
  the exact spectral decomposition of a symmetric matrix; ``eigh`` returns
  eigenvalues in **ascending** order, so we reverse to match the C++
  decreasing convention. Eigenvectors of *degenerate* eigenvalues can still
  differ by an orthogonal rotation within the eigenspace, so the raw
  pseudo-root ``B`` is not guaranteed bit-identical to C++ in that case — but
  ``B @ B.T`` (the covariance it reconstructs) always is, and the diagonal is
  additionally pinned exactly by ``normalizePseudoRoot``.
* **Sign convention (matched to C++).** ``SymmetricSchurDecomposition`` pins
  each eigenvector's sign so its first component is non-negative; we apply the
  same rule to the ``eigh`` output. For the distinct-eigenvalue covariance
  matrices the concrete market models produce, this makes ``B`` itself match
  C++ — which the BGM evolvers (W10-B) rely on, because the diffusion term
  ``B @ Z`` is sign-sensitive even though ``B @ B.T`` is not.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix


class SalvagingAlgorithm(Enum):
    """Eigenvalue salvaging strategy.

    # C++ parity: pseudosqrt.hpp ``SalvagingAlgorithm::Type``.
    """

    NONE = "None"
    SPECTRAL = "Spectral"


def _normalize_pseudo_root(matrix: Matrix, pseudo: Matrix) -> None:
    """Rescale each row of ``pseudo`` so its norm matches ``matrix`` diagonal.

    # C++ parity: pseudosqrt.cpp ``normalizePseudoRoot`` (in-place).

    For each row ``i`` with positive norm, multiply by
    ``sqrt(matrix[i][i] / norm)`` so that ``pseudo[i] · pseudo[i]`` equals the
    target variance ``matrix[i][i]`` exactly.
    """
    size = matrix.shape[0]
    for i in range(size):
        norm = float(np.dot(pseudo[i], pseudo[i]))
        if norm > 0.0:
            norm_adj = float(np.sqrt(matrix[i, i] / norm))
            pseudo[i, :] *= norm_adj


def rank_reduced_sqrt(
    matrix: Matrix,
    max_rank: int,
    component_retained_percentage: float = 1.0,
    sa: SalvagingAlgorithm = SalvagingAlgorithm.NONE,
) -> Matrix:
    """Spectral rank-reduced pseudo-square-root of a symmetric ``matrix``.

    # C++ parity: pseudosqrt.cpp ``rankReducedSqrt``.

    Returns a ``(size, retainedFactors)`` matrix ``B`` with
    ``retainedFactors <= max_rank``, such that ``B @ B.T`` approximates
    ``matrix`` (with the diagonal matched exactly via ``normalizePseudoRoot``).
    """
    m = np.asarray(matrix, dtype=np.float64)
    size = m.shape[0]
    qassert.require(
        size == m.shape[1],
        f"non square matrix: {size} rows, {m.shape[1]} columns",
    )
    qassert.require(
        component_retained_percentage > 0.0,
        "no eigenvalues retained",
    )
    qassert.require(
        component_retained_percentage <= 1.0,
        "percentage to be retained > 100%",
    )
    qassert.require(max_rank >= 1, "max rank required < 1")

    # spectral (Principal Component) analysis. numpy.linalg.eigh returns
    # eigenvalues ascending; reverse to C++ decreasing order and reorder
    # the eigenvector columns to match.
    eig_vals_asc, eig_vecs_asc = np.linalg.eigh(m)
    eigen_values = eig_vals_asc[::-1].copy()
    eigen_vectors = eig_vecs_asc[:, ::-1].copy()

    # C++ parity: symmetricschurdecomposition.cpp pins each eigenvector's sign
    # so that its first component is non-negative
    # (``if (temp[col].second[0] < 0.0) sign = -1.0;``). numpy's ``eigh`` makes
    # an arbitrary per-column sign choice, which leaves ``B @ B.T`` invariant
    # but flips the diffusion term ``B @ Z`` used by the market-model evolvers.
    # Applying the same convention makes the pseudo-root itself match C++ (up to
    # the usual degenerate-eigenspace rotation, which does not arise for the
    # distinct-eigenvalue covariance matrices the market models produce).
    for k in range(eigen_vectors.shape[1]):
        if eigen_vectors[0, k] < 0.0:
            eigen_vectors[:, k] = -eigen_vectors[:, k]

    # salvaging algorithm
    if sa is SalvagingAlgorithm.NONE:
        # eigenvalues are sorted in decreasing order
        qassert.require(
            eigen_values[size - 1] >= -1e-16,
            f"negative eigenvalue(s) ({eigen_values[size - 1]:e})",
        )
    elif sa is SalvagingAlgorithm.SPECTRAL:
        # negative eigenvalues set to zero
        eigen_values = np.maximum(eigen_values, 0.0)

    # factor reduction
    enough = component_retained_percentage * float(np.sum(eigen_values))
    if component_retained_percentage == 1.0:
        # numerical glitches might cause some factors to be discarded
        enough *= 1.1
    # retain at least one factor
    components = float(eigen_values[0])
    retained_factors = 1
    i = 1
    while components < enough and i < size:
        components += float(eigen_values[i])
        retained_factors += 1
        i += 1
    # output is granted to have a rank <= max_rank
    retained_factors = min(retained_factors, max_rank)

    diagonal = np.zeros((size, retained_factors), dtype=np.float64)
    for k in range(retained_factors):
        diagonal[k, k] = float(np.sqrt(eigen_values[k]))
    result: Matrix = eigen_vectors @ diagonal

    _normalize_pseudo_root(m, result)
    return result
