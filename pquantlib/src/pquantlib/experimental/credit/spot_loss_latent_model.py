"""SpotLossLatentModel — random spot recovery rates correlated with defaults.

# C++ parity: ql/experimental/credit/spotlosslatentmodel.hpp @ v1.42.1.

Bennani-Maetz (2009) / Li (2009) spot-recovery model: each issuer's
recovery rate is itself a Gaussian latent variable correlated with the
default trigger. The C++ template is parameterised on a copula policy
and a multi-factor weight matrix; the Python port simplifies this to a
single common-factor copula and per-name (default_loading, rr_loading)
pair plus a model A parameter that scales the recovery noise.

Conditional on a market factor ``m`` and an unconditional default
probability ``p``, the conditional default probability is the standard
one-factor copula expression; the expected conditional recovery uses
equation 44 of Li (2009) — see ``exp_conditional_recovery_inv_p_inv_rr``
below.

# C++ parity divergence: the C++ template walks per-name factor weight
# rows for both default and RR variables; the Python port uses a single
# common-factor loading per name (the standard one-factor setup). The
# ``cross_idiosync_factors`` array reduces to ``rho_default[i] *
# rho_rr[i]`` (the product of the per-name loadings).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)


class SpotLossLatentModel:
    """Spot-recovery latent model (single-factor Python adaptation).

    Pricing inputs per-name:
      - ``rr_mean[i]`` — the unconditional mean recovery rate.
      - ``rr_loading[i]`` — the per-name factor loading on the *recovery*
        latent variable (square root of correlation).

    Per-default latent variable:
      - ``copula`` — the OneFactorCopula instance carrying ``F_Y`` and ``F_Z``
        (Gaussian or Student).

    The cross-idiosyncratic factor is ``rho_d_i * rho_l_i`` (product of
    the per-name default and RR loadings). The model A parameter is the
    multiplier on the RR latent variable noise.
    """

    __slots__ = ("_copula", "_cum_norm", "_inv_norm", "_model_a", "_n", "_rr_loading", "_rr_mean")

    def __init__(
        self,
        copula: OneFactorCopula,
        rr_mean: Sequence[float],
        rr_loading: Sequence[float],
        model_a: float,
    ) -> None:
        qassert.require(
            len(rr_mean) == len(rr_loading),
            f"rr_mean ({len(rr_mean)}) and rr_loading ({len(rr_loading)}) sizes differ",
        )
        qassert.require(
            len(rr_mean) > 0,
            "rr_mean must be non-empty",
        )
        qassert.require(model_a > 0.0, f"model_a must be > 0, got {model_a}")
        for i, r in enumerate(rr_mean):
            qassert.require(
                0.0 <= r <= 1.0, f"rr_mean[{i}] = {r} not in [0, 1]"
            )
        for i, rl in enumerate(rr_loading):
            qassert.require(
                0.0 <= rl <= 1.0, f"rr_loading[{i}] = {rl} not in [0, 1]"
            )
        self._copula = copula
        self._rr_mean = list(rr_mean)
        self._rr_loading = list(rr_loading)
        self._model_a = model_a
        self._n = len(rr_mean)
        self._inv_norm = InverseCumulativeNormal()
        self._cum_norm = CumulativeNormalDistribution()

    def copula(self) -> OneFactorCopula:
        return self._copula

    def pool_size(self) -> int:
        return self._n

    def rr_mean(self) -> list[float]:
        return list(self._rr_mean)

    def rr_loading(self) -> list[float]:
        return list(self._rr_loading)

    def model_a(self) -> float:
        return self._model_a

    def _cross_idiosync(self, i: int) -> float:
        # # C++ parity: spotlosslatentmodel.hpp constructor — uniform case
        # reduces to rho_d * rho_l = (loading_default)^2 * (loading_rr)^2.
        # In the one-factor Gaussian copula the default loading is sqrt(rho)
        # so the cross factor is rho * rr_loading[i]^2.
        return self._copula.correlation() * self._rr_loading[i] ** 2

    def conditional_default_probability(
        self, prob: float, m: float
    ) -> float:
        """Per-name conditional default probability at factor ``m``.

        # C++ parity: spotlosslatentmodel.hpp:159-177 (the same call as
        # DefaultLatentModel for the single-factor case).
        """
        return self._copula.conditional_probability(prob, m)

    def exp_conditional_recovery(
        self,
        i_name: int,
        prob: float,
        m: float,
    ) -> float:
        """Expected recovery conditional on default and factor m.

        # C++ parity: spotlosslatentmodel.hpp:218 expCondRecoveryP +
        # 229 expCondRecoveryInvPinvRR — Eq. 44 of Li (2009) under the
        # one-factor reduction.
        """
        qassert.require(0 <= i_name < self._n, f"i_name {i_name} out of range")
        if prob < 1e-10:
            return self._rr_mean[i_name]
        rho = self._copula.correlation()
        # Inverse Phi of the unconditional default prob and mean RR
        inv_p = self._inv_norm(prob)
        inv_rr = self._inv_norm(self._rr_mean[i_name])
        cross = self._cross_idiosync(i_name)
        sum_betas_loss = self._rr_loading[i_name] ** 2
        # Eq. 44 reduction with sum_ms = sqrt(rho) m
        sum_ms = np.sqrt(rho) * m
        num = (
            sum_ms
            + np.sqrt(1.0 - cross) * np.sqrt(1.0 + self._model_a**2) * inv_rr
            - np.sqrt(cross) * inv_p
        )
        denom = np.sqrt(
            1.0
            - sum_betas_loss
            + self._model_a**2 * (1.0 - cross)
        )
        return float(self._cum_norm(float(num / denom)))

    def expected_loss(self, i_name: int, prob: float) -> float:
        """Single-name expected loss = pd * (1 - exp RR).

        # C++ parity: spotlosslatentmodel.hpp:318. Integrates the
        # conditional product over M.
        """
        qassert.require(0 <= i_name < self._n, f"i_name {i_name} out of range")
        if prob < 1e-10:
            return 0.0
        self._copula.calculate()
        acc = 0.0
        for k in range(self._copula.steps()):
            mk = self._copula.m(k)
            pd_m = self._copula.conditional_probability(prob, mk)
            rr_m = self.exp_conditional_recovery(i_name, prob, mk)
            acc += pd_m * (1.0 - rr_m) * self._copula.density_dm(k)
        return acc
