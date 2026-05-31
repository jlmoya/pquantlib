"""moore_penrose_inverse — Moore-Penrose pseudo-inverse of a real matrix.

# C++ parity: ql/experimental/math/moorepenroseinverse.hpp @ v1.42.1 (099987f0).

# C++ parity divergence (wrapped delegation): the C++ ``moorePenroseInverse``
# hand-rolls the pseudo-inverse from QuantLib's own ``SVD`` decomposition —
# ``V * diag(1/sigma_i) * U^T`` with a singular-value cut-off
# ``tol = max(m, n) * eps * |sigma_0|``. The Python port delegates to
# :func:`numpy.linalg.pinv`, which is the ecosystem-superior equivalent (it
# uses LAPACK's divide-and-conquer SVD and applies the identical relative
# singular-value cut-off ``rcond * sigma_max``). The result is the same
# minimal-norm pseudo-inverse to machine precision. See ``docs/carve-outs.md``
# Category 3 (ecosystem-tooling replacements).

The default cut-off matches C++: ``max(m, n) * eps * sigma_max`` (an *absolute*
threshold). To reproduce it exactly we pass numpy a *relative* ``rcond`` of
``max(m, n) * eps`` (numpy multiplies ``rcond`` by ``sigma_max`` internally).
An explicit ``tol`` argument, when given, is the absolute singular-value
threshold and is converted to the equivalent relative ``rcond``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib.math.constants import QL_EPSILON


def moore_penrose_inverse(
    matrix: npt.ArrayLike, tol: float | None = None
) -> npt.NDArray[np.float64]:
    """Moore-Penrose pseudo-inverse of a real matrix.

    :param matrix: the (m x n) real matrix.
    :param tol: optional absolute singular-value cut-off. When ``None`` the
        C++ default ``max(m, n) * eps * sigma_max`` is used.
    :returns: the (n x m) pseudo-inverse.
    """
    a = np.asarray(matrix, dtype=np.float64)
    m, n = a.shape
    if tol is None:
        # numpy multiplies rcond by sigma_max, giving the C++ absolute cut-off.
        rcond = max(m, n) * QL_EPSILON
    else:
        sigma_max = float(np.linalg.svd(a, compute_uv=False)[0])
        # convert the absolute tolerance into numpy's relative rcond.
        rcond = tol / sigma_max if sigma_max != 0.0 else 0.0
    return np.asarray(np.linalg.pinv(a, rcond=rcond), dtype=np.float64)
