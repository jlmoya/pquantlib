"""LmLinearExponentialVolatilityModel — linear-exponential caplet volatilities.

# C++ parity: ql/legacy/libormarketmodels/lmlinexpvolmodel.{hpp,cpp} (v1.43).

    sigma_i(t) = (a (T_i - t) + d) exp(-b (T_i - t)) + c   for T_i > t
               = 0                                          otherwise

References: Damiano Brigo, Fabio Mercurio, Massimo Morini, 2003, *Different
Covariance Parameterizations of Libor Market Model and Joint Caps/Swaptions
Calibration*.

The four parameters live in ``params()`` as ``ConstantParameter``s under a
``PositiveConstraint``, in the order (a, b, c, d).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib.legacy.libormarketmodels.lm_vol_model import LmVolatilityModel
from pquantlib.math.array import Array
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.models.parameter import ConstantParameter


class LmLinearExponentialVolatilityModel(LmVolatilityModel):
    """Linear-exponential volatility model.

    # C++ parity: ``class LmLinearExponentialVolatilityModel``
    # (lmlinexpvolmodel.hpp:46-59).
    """

    def __init__(
        self, fixing_times: Sequence[float], a: float, b: float, c: float, d: float
    ) -> None:
        # C++ parity: lmlinexpvolmodel.cpp:24-33.
        super().__init__(len(fixing_times), 4)
        self._fixing_times: list[float] = list(fixing_times)
        self._arguments[0] = ConstantParameter(a, PositiveConstraint())
        self._arguments[1] = ConstantParameter(b, PositiveConstraint())
        self._arguments[2] = ConstantParameter(c, PositiveConstraint())
        self._arguments[3] = ConstantParameter(d, PositiveConstraint())

    # --- inspectors -------------------------------------------------------

    def fixing_times(self) -> list[float]:
        """The T_i grid the model was built on.

        Python addition: C++ keeps ``fixingTimes_`` private with no accessor.
        Exposed here because the Python port has no ``friend`` mechanism and
        the value is needed to reason about a constructed model.
        """
        return list(self._fixing_times)

    def _abcd(self) -> tuple[float, float, float, float]:
        """Read (a, b, c, d) off the current arguments, as C++ does at t = 0."""
        return (
            self._arguments[0](0.0),
            self._arguments[1](0.0),
            self._arguments[2](0.0),
            self._arguments[3](0.0),
        )

    # --- volatility -------------------------------------------------------

    def volatility(self, t: float, x: Array | None = None) -> Array:
        """# C++ parity: lmlinexpvolmodel.cpp:36-53."""
        a, b, c, d = self._abcd()
        tmp = np.zeros(self._size, dtype=np.float64)
        for i in range(self._size):
            big_t = self._fixing_times[i]
            if big_t > t:
                tmp[i] = (a * (big_t - t) + d) * math.exp(-b * (big_t - t)) + c
        return tmp

    def volatility_scalar(self, i: int, t: float, x: Array | None = None) -> float:
        """# C++ parity: lmlinexpvolmodel.cpp:55-65."""
        a, b, c, d = self._abcd()
        big_t = self._fixing_times[i]
        if big_t > t:
            return (a * (big_t - t) + d) * math.exp(-b * (big_t - t)) + c
        return 0.0

    def integrated_variance(
        self, i: int, j: int, u: float, x: Array | None = None
    ) -> float:
        """Closed-form integral of sigma_i sigma_j over [0, u].

        # C++ parity: lmlinexpvolmodel.cpp:67-91 — transcribed term by term,
        # including the ``4 b^3 k2 k3`` denominator.
        """
        a, b, c, d = self._abcd()

        big_t = self._fixing_times[i]
        big_s = self._fixing_times[j]

        k1 = math.exp(b * u)
        k2 = math.exp(b * big_s)
        k3 = math.exp(b * big_t)

        return (
            a
            * a
            * (
                -1
                - 2 * b * b * big_s * big_t
                - b * (big_s + big_t)
                + k1
                * k1
                * (
                    1
                    + b * (big_s + big_t - 2 * u)
                    + 2 * b * b * (big_s - u) * (big_t - u)
                )
            )
            + 2
            * b
            * b
            * (
                2 * c * d * (k2 + k3) * (k1 - 1)
                + d * d * (k1 * k1 - 1)
                + 2 * b * c * c * k2 * k3 * u
            )
            + 2
            * a
            * b
            * (
                d * (-1 - b * (big_s + big_t) + k1 * k1 * (1 + b * (big_s + big_t - 2 * u)))
                - 2
                * c
                * (
                    k3 * (1 + b * big_s)
                    + k2 * (1 + b * big_t)
                    - k1 * k3 * (1 + b * (big_s - u))
                    - k1 * k2 * (1 + b * (big_t - u))
                )
            )
        ) / (4 * b * b * b * k2 * k3)

    # --- protected --------------------------------------------------------

    def _generate_arguments(self) -> None:
        """No cached state to rebuild.

        # C++ parity: lmlinexpvolmodel.cpp:93 — empty body. The (a, b, c, d)
        # are read out of ``arguments_`` on every call, so ``setParams`` takes
        # effect without any recomputation.
        """


__all__ = ["LmLinearExponentialVolatilityModel"]
