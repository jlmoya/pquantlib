"""LfmCovarianceProxy — covariance parameterization from a vol + a corr model.

# C++ parity: ql/legacy/libormarketmodels/lfmcovarproxy.{hpp,cpp} (v1.43).

Combines an :class:`LmVolatilityModel` and an :class:`LmCorrelationModel`:

    diffusion(t)[i][q]  = volatility(t)[i] * corr.pseudoSqrt(t)[i][q]
    covariance(t)[i][j] = volatility(t)[i] * corr(t)[i][j] * volatility(t)[j]

``integrated_covariance_scalar`` has two branches, both load-bearing:

1. If the correlation model is time independent AND the volatility model
   implements ``integrated_variance``, the answer is analytic.
2. Otherwise (including when the analytic call RAISES — C++ catches
   ``Error&`` and carries on) it falls back to 64 adaptive Gauss-Kronrod
   segments over ``vol_i * rho_ij * vol_j``.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.legacy.libormarketmodels.lfm_covar_param import (
    LfmCovarianceParameterization,
)
from pquantlib.legacy.libormarketmodels.lm_corr_model import LmCorrelationModel
from pquantlib.legacy.libormarketmodels.lm_vol_model import LmVolatilityModel
from pquantlib.math.array import Array
from pquantlib.math.integrals.kronrod import GaussKronrodAdaptive
from pquantlib.math.matrix import Matrix


class LfmCovarianceProxy(LfmCovarianceParameterization):
    """Covariance parameterization proxying a volatility and correlation model.

    # C++ parity: ``class LfmCovarianceProxy`` (lfmcovarproxy.hpp:35-55).
    """

    def __init__(
        self, vola_model: LmVolatilityModel, corr_model: LmCorrelationModel
    ) -> None:
        # C++ parity: lfmcovarproxy.cpp:25-35 — size and factors both come
        # from the CORRELATION model, not the volatility model.
        super().__init__(corr_model.size(), corr_model.factors())
        self._vola_model: LmVolatilityModel = vola_model
        self._corr_model: LmCorrelationModel = corr_model
        qassert.require(
            vola_model.size() == corr_model.size(),
            f"different size for the volatility ({vola_model.size()}) and "
            f"correlation ({corr_model.size()}) models",
        )

    # --- inspectors -------------------------------------------------------

    def volatility_model(self) -> LmVolatilityModel:
        """# C++ parity: ``LfmCovarianceProxy::volatilityModel`` (.cpp:37-40)."""
        return self._vola_model

    def correlation_model(self) -> LmCorrelationModel:
        """# C++ parity: ``LfmCovarianceProxy::correlationModel`` (.cpp:42-45)."""
        return self._corr_model

    # --- parameterization -------------------------------------------------

    def diffusion(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lfmcovarproxy.cpp:47-58 — scale row i of the
        correlation pseudo-root by ``vol[i]``.
        """
        pca = self._corr_model.pseudo_sqrt(t, x)
        vol = self._vola_model.volatility(t, x)
        for i in range(self._size):
            pca[i, :] *= vol[i]
        return pca

    def covariance(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lfmcovarproxy.cpp:60-73 — the analytic form, NOT
        ``diffusion @ diffusion.T``.
        """
        volatility = self._vola_model.volatility(t, x)
        correlation = self._corr_model.correlation(t, x)

        tmp = np.zeros((self._size, self._size), dtype=np.float64)
        for i in range(self._size):
            for j in range(self._size):
                tmp[i, j] = volatility[i] * correlation[i, j] * volatility[j]
        return tmp

    def integrated_covariance_scalar(
        self, i: int, j: int, t: float, x: Array | None = None
    ) -> float:
        """Integrated covariance of forwards ``i`` and ``j`` over [0, t].

        # C++ parity: ``LfmCovarianceProxy::integratedCovariance``
        # (lfmcovarproxy.cpp:107-134).

        Note the argument order in the analytic branch: C++ asks the volatility
        model for ``integratedVariance(j, i, t, x)`` — j FIRST — while the
        correlation factor is ``correlation(i, j, 0.0, x)``. For every shipped
        volatility model ``integratedVariance`` is symmetric in (i, j), but the
        order is reproduced rather than "fixed".
        """
        if self._corr_model.is_time_independent():
            try:
                # if all objects support these methods that's by far the
                # fastest way to get the integrated covariance
                return self._corr_model.correlation_scalar(
                    i, j, 0.0, x
                ) * self._vola_model.integrated_variance(j, i, t, x)
            except LibraryException:
                # okay proceed with the slow numerical integration routine
                pass

        qassert.require(x is None or np.asarray(x).size == 0, "can not handle given x here")

        def helper(u: float) -> float:
            # C++ parity: LfmCovarianceProxy::Var_Helper::operator()
            # (lfmcovarproxy.cpp:94-105).
            if i == j:
                v1 = v2 = self._vola_model.volatility_scalar(i, u)
            else:
                v1 = self._vola_model.volatility_scalar(i, u)
                v2 = self._vola_model.volatility_scalar(j, u)
            return v1 * self._corr_model.correlation_scalar(i, j, u) * v2

        integrator = GaussKronrodAdaptive(1e-10, 10000)
        total = 0.0
        for k in range(64):
            total += integrator(helper, k * t / 64.0, (k + 1) * t / 64.0)
        return total


__all__ = ["LfmCovarianceProxy"]
