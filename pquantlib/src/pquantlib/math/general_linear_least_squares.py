"""General linear least squares by singular value decomposition.

# C++ parity: ql/math/generallinearleastsquares.hpp (v1.43) — header-only
#             template.

Fit ``y ~ sum_j a_j v_j(x)`` for an arbitrary set of basis functions ``v``.

Not ``numpy.linalg.lstsq`` and not a normal-equation solve. C++ builds the
design matrix, takes *its own* :class:`~pquantlib.math.matrixutilities.svd.SVD`
and then drops every direction whose singular value fails ``w[i] > n * eps *
w[0]`` — so a rank-deficient basis does not blow up, it silently loses that
direction, and the coefficients you get back are the minimum-norm ones for the
surviving subspace. ``numpy.linalg.lstsq`` uses a different (``rcond``-based)
cut-off and a different driver.

Two quantities that look interchangeable are not:

* :meth:`error` is the modelling uncertainty of Numerical Recipes,
  ``sqrt(sum_i V[j][i]^2 / w[i]^2)``;
* :meth:`standard_errors` is that multiplied by ``sqrt(chiSq / (n - 2))`` —
  note ``n - 2``, not ``n - m``, regardless of how many basis functions there
  are. That is what C++ does, and callers are calibrated against it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.matrixutilities.svd import SVD

# The C++ template parameter ``ArgumentType`` is either a Real or a container
# of Reals; Python models that as a union at the call site.
type RegressionArgument = float | Sequence[float]


class GeneralLinearLeastSquares:
    """Least-squares fit of ``y`` onto the span of ``v`` evaluated at ``x``.

    # C++ parity: ``class GeneralLinearLeastSquares`` —
    # generallinearleastsquares.hpp:37-77, ``calculate`` at :103-146.
    """

    __slots__ = ("_a", "_err", "_residuals", "_standard_errors")

    def __init__(
        self,
        x: Sequence[RegressionArgument],
        y: Sequence[float],
        v: Sequence[Callable[..., float]],
    ) -> None:
        # C++ parity: generallinearleastsquares.hpp:103-146.
        n = len(y)
        m = len(v)

        qassert.require(n == len(y), "sample set need to be of the same size")
        qassert.require(n >= m, "sample set is too small")

        design = np.empty((n, m), dtype=np.float64)
        for i in range(m):
            for k, xk in enumerate(x):
                design[k, i] = v[i](xk)

        svd = SVD(design)
        v_matrix = svd.v()
        u_matrix = svd.u()
        w = svd.singular_values()
        threshold = n * QL_EPSILON * float(w[0])

        a = np.zeros(m, dtype=np.float64)
        err = np.zeros(m, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        for i in range(m):
            if float(w[i]) > threshold:
                u = float(np.dot(u_matrix[:, i], y_arr)) / float(w[i])
                for j in range(m):
                    a[j] += u * float(v_matrix[j, i])
                    err[j] += float(v_matrix[j, i]) ** 2 / (float(w[i]) ** 2)

        self._a: Array = a
        self._err: Array = np.sqrt(err)
        self._residuals: Array = design @ a - y_arr

        chi_sq = float(np.dot(self._residuals, self._residuals))
        multiplier = math.sqrt(chi_sq / (n - 2))
        self._standard_errors: Array = self._err * multiplier

    def coefficients(self) -> Array:
        # C++ parity: generallinearleastsquares.hpp:59.
        return self._a

    def residuals(self) -> Array:
        # C++ parity: generallinearleastsquares.hpp:60.
        return self._residuals

    def standard_errors(self) -> Array:
        """Standard parameter errors as given by Excel, R etc.

        # C++ parity: generallinearleastsquares.hpp:63.
        """
        return self._standard_errors

    def error(self) -> Array:
        """Modelling uncertainty as defined in Numerical Recipes.

        # C++ parity: generallinearleastsquares.hpp:65.
        """
        return self._err

    def size(self) -> int:
        # C++ parity: generallinearleastsquares.hpp:67.
        return int(self._residuals.shape[0])

    def dim(self) -> int:
        # C++ parity: generallinearleastsquares.hpp:69.
        return int(self._a.shape[0])


__all__ = ["GeneralLinearLeastSquares", "RegressionArgument"]
