"""BinomialLossModel — O'Kane (2007) adjusted-binomial loss approximation.

# C++ parity: ql/experimental/credit/binomiallossmodel.hpp @ v1.42.1.

Models the portfolio loss distribution by an adjusted binomial that
matches the first two moments of the true conditional loss distribution.
Cheaper than the recursive convolution and adequate for pricing
(though not for VaR-style tail metrics — for that use Recursive or LHP).

The C++ template is parameterised on the LLM (loss-latent-model) type;
the Python port consumes a ``ConstantLossLatentModel`` directly.

# C++ parity divergence: the C++ surface uses Basket-backed
# remainingNotionals + remainingProbabilities; the Python port takes
# them as explicit arguments. Heterogeneous notionals + probabilities
# are supported; the LGD vector comes from the latent model's recoveries.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentModel,
)
from pquantlib.experimental.credit.default_loss_model import DefaultLossModel


class BinomialLossModel(DefaultLossModel):
    """Adjusted-binomial loss-distribution model.

    Constructor takes a ``ConstantLossLatentModel`` and the per-pool
    LGD-bucket parameter (currently unused — kept for signature parity).
    """

    __slots__ = ("_latent_model", "_lgd_buckets")

    def __init__(
        self,
        latent_model: ConstantLossLatentModel,
        lgd_buckets: int = 10,
    ) -> None:
        super().__init__()
        qassert.require(
            lgd_buckets > 0, f"lgd_buckets must be > 0, got {lgd_buckets}"
        )
        self._latent_model = latent_model
        self._lgd_buckets = lgd_buckets

    def latent_model(self) -> ConstantLossLatentModel:
        return self._latent_model

    def lgd_buckets(self) -> int:
        return self._lgd_buckets

    def _loss_probability_density(
        self,
        probs: Sequence[float],
        notionals: Sequence[float],
        m: float,
    ) -> list[float]:
        """Conditional loss probability density at factor M = m.

        Implements the O'Kane adjusted-binomial recursion.

        # C++ parity: binomiallossmodel.hpp:146-242 lossProbability.
        """
        n = len(probs)
        rrs = self._latent_model.recoveries()
        # # C++ parity: binomiallossmodel.hpp:165-169
        # fractionalEL[i] = 1 - conditional_recovery (= 1 - rr_i for constant model).
        fractional_el = [1.0 - rrs[i] for i in range(n)]
        lgds_left = [fractional_el[i] * notionals[i] for i in range(n)]
        avg_lgd = sum(lgds_left) / n

        # Per-name conditional default probabilities.
        cond_def_probs = self._latent_model.conditional_default_probability_vec(
            probs, m
        )

        # Average per-name conditional probability weighted by LGD.
        eps = 1e-14
        if avg_lgd <= eps:
            avg_prob = 0.0
        else:
            avg_prob = sum(
                cond_def_probs[i] * lgds_left[i] for i in range(n)
            ) / (avg_lgd * n)

        # # C++ parity: binomiallossmodel.hpp:185-211 — adjusted-binomial parameters.
        m_param = avg_prob * n
        floor_avg_prob = min(float(n - 1), float(int(m_param)))
        ceil_avg_prob = floor_avg_prob + 1.0

        variance_binom = avg_prob * (1.0 - avg_prob) / n
        # Compute variance of true LGD-weighted Bernoulli.
        var_num = sum(
            cond_def_probs[j] * (1.0 - cond_def_probs[j]) * lgds_left[j] ** 2
            for j in range(n)
        )
        variance = 0.0 if avg_lgd <= eps else var_num / (n**2 * avg_lgd**2)

        sum_aves = -((ceil_avg_prob - m_param) ** 2) - (
            (floor_avg_prob - m_param) ** 2 - ceil_avg_prob**2
        ) * (ceil_avg_prob - m_param)
        if abs(variance_binom * n + sum_aves) < eps:
            alpha = 1.0
        else:
            alpha = (variance * n + sum_aves) / (variance_binom * n + sum_aves)

        # Build the binomial density.
        density = [0.0] * (n + 1)
        if avg_prob >= 1.0 - eps:
            density[n] = 1.0
        elif avg_prob <= eps:
            density[0] = 1.0
        else:
            probs_ratio = avg_prob / (1.0 - avg_prob)
            density[0] = (1.0 - avg_prob) ** n
            for i in range(1, n + 1):
                density[i] = density[i - 1] * probs_ratio * (n - i + 1) / i
            # Adjust with alpha and redistribute the residual.
            for i in range(n + 1):
                density[i] *= alpha
            epsilon = (1.0 - alpha) * (ceil_avg_prob - m_param)
            epsilon_plus = 1.0 - alpha - epsilon
            density[int(floor_avg_prob)] += epsilon
            density[int(ceil_avg_prob)] += epsilon_plus
        return density

    def _loss_points(
        self,
        probs: Sequence[float],
        notionals: Sequence[float],
    ) -> list[float]:
        """Attainable per-bucket loss values.

        # C++ parity: binomiallossmodel.hpp:272-289 lossPoints +
        # binomiallossmodel.hpp:247-269 averageLoss. ``averageLoss``
        # returns the (averaged-per-name) LGD scaled by total notional
        # — it does NOT multiply by the conditional default probability,
        # because the binomial-density integration over M already
        # captures the conditional-default mass.

        Returns ``i * avg_loss_frct * total_notional`` for i = 0..N where
        ``avg_loss_frct = E[ sum_i (1 - RR_i) N_i / (N * total_notional) ]``
        (factoring out the per-name LGD-weighted notional, integrated
        over the M factor — but for constant LGD this is deterministic).
        """
        rrs = self._latent_model.recoveries()
        total_notional = sum(notionals)
        n = len(probs)
        if total_notional == 0:
            return [0.0] * (n + 1)
        # # C++ parity: averageLoss returns sum_i lgd_i * notional_i / (N * total_not).
        # Constant LGD model: no factor dependence; integration over M is
        # just the constant (the C++ ``integratedExpectedValue`` of a
        # constant equals that constant since the density integrates to 1).
        lgds_per_name = [
            (1.0 - rrs[i]) * notionals[i] for i in range(n)
        ]
        avg_loss_frct = sum(lgds_per_name) / (n * total_notional)
        return [i * avg_loss_frct * total_notional for i in range(n + 1)]

    def loss_distribution(
        self,
        probs: Sequence[float],
        notionals: Sequence[float],
    ) -> dict[float, float]:
        """Return the discrete loss distribution as a dict {loss: prob_density}.

        # C++ parity: binomiallossmodel.hpp:329-343 lossDistribution.
        """
        loss_pts = self._loss_points(probs, notionals)
        cop = self._latent_model.copula()
        cop.calculate()
        # Integrate the conditional density over M.
        density = [0.0] * (self._latent_model.pool_size() + 1)
        for k in range(cop.steps()):
            mk = cop.m(k)
            cond = self._loss_probability_density(probs, notionals, mk)
            w_k = cop.density_dm(k)
            for i in range(len(density)):
                density[i] += cond[i] * w_k
        return dict(zip(loss_pts, density, strict=True))

    def expected_tranche_loss(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
    ) -> float:
        """Expected tranche loss in the homogeneous case.

        For heterogeneous baskets call
        ``expected_tranche_loss_heterogeneous`` directly.

        # C++ parity: binomiallossmodel.hpp:311-325 expectedTrancheLoss.
        """
        _ = average_rr
        n = self._latent_model.pool_size()
        return self.expected_tranche_loss_heterogeneous(
            remaining_notional=remaining_notional,
            probs=[prob] * n,
            notionals=[remaining_notional / n] * n,
            attach=attach,
            detach=detach,
        )

    def expected_tranche_loss_heterogeneous(
        self,
        remaining_notional: float,
        probs: Sequence[float],
        notionals: Sequence[float],
        attach: float,
        detach: float,
    ) -> float:
        """ETL given per-name probabilities and notionals."""
        _ = remaining_notional
        loss_pts = self._loss_points(probs, notionals)
        cop = self._latent_model.copula()
        cop.calculate()
        # Integrate conditional tranche loss over M.
        etl = 0.0
        for k in range(cop.steps()):
            mk = cop.m(k)
            cond = self._loss_probability_density(probs, notionals, mk)
            w_k = cop.density_dm(k)
            cond_tranche = 0.0
            for i in range(len(loss_pts)):
                clipped = max(0.0, min(detach, loss_pts[i]) - attach)
                cond_tranche += cond[i] * clipped
            etl += cond_tranche * w_k
        return etl

    def percentile(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
        perctl: float,
    ) -> float:
        """Percentile loss — walks the discrete CDF.

        # C++ parity: binomiallossmodel.hpp:346-370 percentile.
        """
        _ = average_rr
        qassert.require(
            0.0 <= perctl <= 1.0, f"perctl {perctl} out of range [0, 1]"
        )
        n = self._latent_model.pool_size()
        return self.percentile_heterogeneous(
            remaining_notional=remaining_notional,
            probs=[prob] * n,
            notionals=[remaining_notional / n] * n,
            attach=attach,
            detach=detach,
            perctl=perctl,
        )

    def percentile_heterogeneous(
        self,
        remaining_notional: float,
        probs: Sequence[float],
        notionals: Sequence[float],
        attach: float,
        detach: float,
        perctl: float,
    ) -> float:
        """Percentile loss given heterogeneous basket."""
        _ = remaining_notional
        dist = self.loss_distribution(probs, notionals)
        sorted_items = sorted(dist.items())
        if not sorted_items:
            return 0.0
        # Build CDF and interpolate
        cum = 0.0
        cdf: list[tuple[float, float]] = []
        for loss, mass in sorted_items:
            cum += mass
            cdf.append((loss, cum))
        if cdf[0][1] >= perctl:
            return cdf[0][0]
        if perctl == 1.0:
            return cdf[-1][0]
        for i, (loss, cprob) in enumerate(cdf):
            if cprob > perctl:
                if i == 0:
                    return loss
                x_min, v_min = cdf[i - 1]
                x_plus, v_plus = loss, cprob
                portf = x_plus - (x_plus - x_min) * (v_plus - perctl) / (
                    v_plus - v_min
                )
                return min(max(portf - attach, 0.0), detach - attach)
        return cdf[-1][0]


# Re-export numpy types for downstream pyright (no public API).
__all__ = ["BinomialLossModel"]
_ = np  # silence unused import warning while keeping numpy available for future extensions
