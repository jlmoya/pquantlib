"""Cholesky decomposition.

# C++ parity: ql/math/matrixutilities/choleskydecomposition.hpp +
# ql/math/matrixutilities/choleskydecomposition.cpp (v1.42.1) —
# ``Matrix CholeskyDecomposition(const Matrix& m, bool flexible)``.

For a symmetric positive-definite (SPD) matrix ``M``, returns the
lower-triangular factor ``L`` such that ``M == L @ L.T``. C++ delegates
to its own hand-rolled forward-elimination loop (``choleskydecomposition.cpp``);
the Python port delegates to ``scipy.linalg.cholesky(m, lower=True)``
which calls LAPACK ``dpotrf``. Cross-validated against the C++ probe at
TIGHT tolerance.

The C++ ``flexible=true`` branch falls back to an eigen-decomposition-
based pseudo-Cholesky for non-SPD inputs (using
``SymmetricSchurDecomposition``). This branch is **deferred** —
``SymmetricSchurDecomposition`` is not yet ported (L1-E carve-out per
``phase1-l1-E-design.md``). Calling with ``flexible=True`` raises
``LibraryException`` until that gets added in a follow-up cluster.

The C++ helper ``CholeskySolveFor`` (forward+back substitution for
``L L.T x = b``) is also not ported here — it can be added trivially
on top of ``scipy.linalg.solve_triangular`` when first needed.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.linalg  # pyright: ignore[reportMissingTypeStubs]

from pquantlib import qassert
from pquantlib.math.matrix import Matrix


def cholesky_decomposition(m: Matrix, flexible: bool = False) -> Matrix:
    """Return ``L`` (lower-triangular) such that ``m == L @ L.T``.

    Args:
        m: A symmetric positive-definite ``(n, n)`` ``float64`` matrix.
        flexible: If True, fall back to eigenvalue-based pseudo-Cholesky
            when ``m`` is not strictly SPD. **Deferred** — raises
            ``LibraryException`` until ``SymmetricSchurDecomposition``
            lands.

    Raises:
        LibraryException: if ``m`` is not square, or if ``flexible`` is
            requested (deferred), or if the strict-Cholesky path fails
            (``scipy.linalg.LinAlgError`` is wrapped).
    """
    arr = np.ascontiguousarray(m, dtype=np.float64)
    qassert.require(arr.ndim == 2, f"cholesky_decomposition requires a 2-D matrix, got ndim={arr.ndim}")
    qassert.require(
        arr.shape[0] == arr.shape[1],
        f"cholesky_decomposition requires a square matrix, got shape {arr.shape}",
    )
    if flexible:
        # C++ falls back to eigenvalue-based pseudo-Cholesky here. Requires
        # SymmetricSchurDecomposition which is deferred from L1-E per
        # phase1-l1-E-design.md carve-outs.
        qassert.fail(
            "cholesky_decomposition(flexible=True) is not yet implemented "
            "(SymmetricSchurDecomposition deferred per L1-E carve-outs)"
        )
    # scipy lacks bundled type stubs, so the result is `Unknown`; cast to
    # ``Matrix`` (a numpy float64 NDArray alias) for pyright. ``lower=True``
    # selects the C++-parity convention ``m == L @ L.T``.
    try:
        result = scipy.linalg.cholesky(arr, lower=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except scipy.linalg.LinAlgError as e:  # pyright: ignore[reportUnknownMemberType]
        qassert.fail(f"cholesky_decomposition failed: {e}")
    return cast(Matrix, result)
