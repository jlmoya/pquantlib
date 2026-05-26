"""Matrix — rank-2 dense float64 matrix type alias.

# C++ parity: ql/math/matrix.hpp (v1.42.1) — ``QuantLib::Matrix``.

C++ wraps ``boost::numeric::ublas::matrix<Real>`` in a thin ``Matrix``
class. JQuantLib reimplemented this as a custom ``Matrix`` class
because the JVM lacked vectorized primitive 2-D arrays (and a
``double[][]`` jagged array is the wrong shape for BLAS-style ops).

PQuantLib diverges: ``Matrix`` is a **type alias** for
``numpy.typing.NDArray[numpy.float64]``. Same rationale as ``Array``:

- numpy + scipy give us BLAS/LAPACK-backed linear algebra (``solve``,
  ``inv``, ``cholesky``, ``svd``, ``eig``) without rewriting a wrapper.
- Pyright sees a rank-agnostic ``NDArray[float64]``; the rank-2
  invariant is documented and enforced at runtime via ``matrix.ndim == 2``.
- Indexing is C-order ``m[row, col]`` matching C++ ``m[i][j]`` with
  ``i`` the row index. Bilinear interpolation uses ``z[j, i]`` (j=y, i=x)
  as in the C++ source.

Construction helpers (when needed) should use ``numpy.asarray(..., dtype=float64)``
with explicit 2-D nesting at the call site, or ``numpy.zeros((rows, cols),
dtype=float64)`` / ``numpy.eye(n, dtype=float64)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# RANK INVARIANT: 2-D. Pyright cannot enforce this at the type level
# (NDArray has no rank parameter in current numpy stubs); callers MUST
# treat ``Matrix`` as a rank-2 ``float64`` matrix indexed as
# ``m[row, col]``. Use ``Array`` for rank-1.
type Matrix = npt.NDArray[np.float64]
