"""LmExponentialCorrelationModel — rho_ij = exp(-beta |i - j|).

# C++ parity: ql/legacy/libormarketmodels/lmexpcorrmodel.{hpp,cpp} (v1.43).

One free parameter (called ``rho`` in the C++ constructor, though it plays the
role of the decay rate ``beta`` in the formula) under a ``PositiveConstraint``.
The correlation matrix and its spectral pseudo-square-root are computed once in
``_generate_arguments`` and cached, so both accessors are time-independent.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib.legacy.libormarketmodels.lm_corr_model import (
    LmCorrelationModel,
    spectral_pseudo_sqrt,
)
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.models.parameter import ConstantParameter


class LmExponentialCorrelationModel(LmCorrelationModel):
    """Exponential correlation model.

    # C++ parity: ``class LmExponentialCorrelationModel``
    # (lmexpcorrmodel.hpp:45-59).
    """

    def __init__(self, size: int, rho: float) -> None:
        # C++ parity: lmexpcorrmodel.cpp:25-32.
        super().__init__(size, 1)
        self._corr_matrix: Matrix = np.zeros((size, size), dtype=np.float64)
        self._pseudo_sqrt: Matrix = np.zeros((size, size), dtype=np.float64)
        self._arguments[0] = ConstantParameter(rho, PositiveConstraint())
        self._generate_arguments()

    def correlation(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lmexpcorrmodel.cpp:34-38 — returns a copy of the cache."""
        return self._corr_matrix.copy()

    def correlation_scalar(
        self, i: int, j: int, t: float, x: Array | None = None
    ) -> float:
        """# C++ parity: lmexpcorrmodel.cpp:40-43."""
        return float(self._corr_matrix[i, j])

    def is_time_independent(self) -> bool:
        """# C++ parity: lmexpcorrmodel.cpp:45-47 — always ``true``."""
        return True

    def pseudo_sqrt(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lmexpcorrmodel.cpp:49-52 — returns a copy of the cache."""
        return self._pseudo_sqrt.copy()

    def _generate_arguments(self) -> None:
        """# C++ parity: lmexpcorrmodel.cpp:54-66."""
        rho = self._arguments[0](0.0)
        for i in range(self._size):
            for j in range(i, self._size):
                value = math.exp(-rho * abs(float(i) - float(j)))
                self._corr_matrix[i, j] = value
                self._corr_matrix[j, i] = value
        self._pseudo_sqrt = spectral_pseudo_sqrt(self._corr_matrix)


__all__ = ["LmExponentialCorrelationModel"]
