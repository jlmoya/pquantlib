"""LmLinearExponentialCorrelationModel — rho + (1 - rho) exp(-beta |i - j|).

# C++ parity: ql/legacy/libormarketmodels/lmlinexpcorrmodel.{hpp,cpp} (v1.43).

Two free parameters: ``rho`` on ``BoundaryConstraint(-1, 1)`` and ``beta`` on
``PositiveConstraint``. The distinctive part is the factor reduction: the
pseudo-root is a RANK-REDUCED square root with at most ``factors`` columns, and
``_generate_arguments`` then OVERWRITES the correlation matrix with
``B B^T``. The stored correlation is therefore the rank-reduced
reconstruction, not the analytic formula, whenever ``factors < size``.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib.legacy.libormarketmodels.lm_corr_model import LmCorrelationModel
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    PositiveConstraint,
)
from pquantlib.models.marketmodels.models.pseudo_sqrt import (
    SalvagingAlgorithm,
    rank_reduced_sqrt,
)
from pquantlib.models.parameter import ConstantParameter


class LmLinearExponentialCorrelationModel(LmCorrelationModel):
    """Linear-exponential correlation model with optional factor reduction.

    # C++ parity: ``class LmLinearExponentialCorrelationModel``
    # (lmlinexpcorrmodel.hpp:45-63).
    """

    def __init__(
        self, size: int, rho: float, beta: float, factors: int | None = None
    ) -> None:
        """Build the model.

        # C++ parity: lmlinexpcorrmodel.cpp:25-33. C++ signals "use every
        # factor" with ``factors = Null<Size>()``; the Python port uses
        # ``None`` for the same sentinel.
        """
        super().__init__(size, 2)
        self._corr_matrix: Matrix = np.zeros((size, size), dtype=np.float64)
        self._pseudo_sqrt: Matrix = np.zeros((size, size), dtype=np.float64)
        self._factors: int = factors if factors is not None else size
        self._arguments[0] = ConstantParameter(rho, BoundaryConstraint(-1.0, 1.0))
        self._arguments[1] = ConstantParameter(beta, PositiveConstraint())
        self._generate_arguments()

    def correlation(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lmlinexpcorrmodel.cpp:35-38."""
        return self._corr_matrix.copy()

    def correlation_scalar(
        self, i: int, j: int, t: float, x: Array | None = None
    ) -> float:
        """# C++ parity: lmlinexpcorrmodel.cpp:40-43."""
        return float(self._corr_matrix[i, j])

    def is_time_independent(self) -> bool:
        """# C++ parity: lmlinexpcorrmodel.cpp:45-47 — always ``true``."""
        return True

    def factors(self) -> int:
        """# C++ parity: lmlinexpcorrmodel.cpp:49-51."""
        return self._factors

    def pseudo_sqrt(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lmlinexpcorrmodel.cpp:54-57."""
        return self._pseudo_sqrt.copy()

    def _generate_arguments(self) -> None:
        """# C++ parity: lmlinexpcorrmodel.cpp:59-74.

        Note the last two statements: the pseudo-root is rank-reduced with
        ``SalvagingAlgorithm::None`` and the correlation matrix is then
        REPLACED by ``pseudoSqrt * transpose(pseudoSqrt)``. Dropping that
        replacement would leave ``correlation()`` inconsistent with
        ``pseudoSqrt()`` for any ``factors < size``.
        """
        rho = self._arguments[0](0.0)
        beta = self._arguments[1](0.0)

        for i in range(self._size):
            for j in range(i, self._size):
                value = rho + (1 - rho) * math.exp(-beta * abs(float(i) - float(j)))
                self._corr_matrix[i, j] = value
                self._corr_matrix[j, i] = value

        self._pseudo_sqrt = rank_reduced_sqrt(
            self._corr_matrix, self._factors, 1.0, SalvagingAlgorithm.NONE
        )
        self._corr_matrix = self._pseudo_sqrt @ self._pseudo_sqrt.T


__all__ = ["LmLinearExponentialCorrelationModel"]
