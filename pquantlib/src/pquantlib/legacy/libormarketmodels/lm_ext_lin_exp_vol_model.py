"""LmExtLinearExponentialVolModel — per-forward-scaled linear-exponential vols.

# C++ parity: ql/legacy/libormarketmodels/lmextlinexpvolmodel.{hpp,cpp} (v1.43).

    sigma_i(t) = k_i ((a (T_i - t) + d) exp(-b (T_i - t)) + c)

i.e. the :class:`LmLinearExponentialVolatilityModel` scaled by one extra free
parameter per forward rate. The parameter vector grows from 4 to ``4 + size``;
the k_i occupy slots ``4 .. 4 + size - 1`` and all start at 1.0, so a freshly
constructed instance is numerically identical to its base class.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.legacy.libormarketmodels.lm_lin_exp_vol_model import (
    LmLinearExponentialVolatilityModel,
)
from pquantlib.math.array import Array
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.models.parameter import ConstantParameter


class LmExtLinearExponentialVolModel(LmLinearExponentialVolatilityModel):
    """Extended linear-exponential volatility model.

    # C++ parity: ``class LmExtLinearExponentialVolModel``
    # (lmextlinexpvolmodel.hpp:46-56).
    """

    def __init__(
        self, fixing_times: Sequence[float], a: float, b: float, c: float, d: float
    ) -> None:
        # C++ parity: lmextlinexpvolmodel.cpp:24-33 — the base ctor allocates
        # four arguments, then ``arguments_.resize(4 + size_)`` appends one
        # unit scaling factor per forward rate.
        super().__init__(fixing_times, a, b, c, d)
        self._arguments.extend(
            ConstantParameter(1.0, PositiveConstraint()) for _ in range(self._size)
        )

    def volatility(self, t: float, x: Array | None = None) -> Array:
        """# C++ parity: lmextlinexpvolmodel.cpp:36-44."""
        tmp = super().volatility(t, x)
        for i in range(self._size):
            tmp[i] *= self._arguments[i + 4](0.0)
        return tmp

    def volatility_scalar(self, i: int, t: float, x: Array | None = None) -> float:
        """# C++ parity: lmextlinexpvolmodel.cpp:46-50."""
        return self._arguments[i + 4](0.0) * super().volatility_scalar(i, t, x)

    def integrated_variance(
        self, i: int, j: int, u: float, x: Array | None = None
    ) -> float:
        """# C++ parity: lmextlinexpvolmodel.cpp:52-56 — scaled by k_i k_j."""
        return (
            self._arguments[i + 4](0.0)
            * self._arguments[j + 4](0.0)
            * super().integrated_variance(i, j, u, x)
        )


__all__ = ["LmExtLinearExponentialVolModel"]
