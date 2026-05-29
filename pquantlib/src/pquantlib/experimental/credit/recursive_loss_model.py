"""RecursiveLossModel — Andersen-Sidenius-Basu recursive convolution.

# C++ parity: ql/experimental/credit/recursivelossmodel.hpp @ v1.42.1.

The C++ template walks the basket name-by-name building up the
conditional loss distribution via the recursion

    P(L = l | M=m) = P(L_{<i} = l - w_i | M=m) p_i(m)
                     + P(L_{<i} = l | M=m) (1 - p_i(m))

where w_i is the rounded LGD bucket for name i. The unconditional
distribution then integrates this over the M factor.

The Python port preserves the same algorithm using the
``ConstantLossLatentModel`` for the per-name conditional default
probability + the LGD vector, and the Euler-grid integration over M
from the underlying ``OneFactorCopula``.

# C++ parity divergence: the C++ algorithm uses a std::map keyed on
# (rounded integer) loss buckets; Python uses a dict. The bucket
# rounding policy matches the C++ ``std::floor(lgd/lossUnit + 0.5)``
# convention.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentModel,
)
from pquantlib.experimental.credit.default_loss_model import DefaultLossModel


class RecursiveLossModel(DefaultLossModel):
    """Andersen-Sidenius-Basu (2003) recursive loss-distribution model.

    Constructor takes a ``ConstantLossLatentModel`` (which carries the
    copula and the recoveries vector) and ``n_buckets`` controlling the
    bucket size (loss_unit = min_lgd / n_buckets).
    """

    __slots__ = ("_latent_model", "_n_buckets")

    def __init__(
        self,
        latent_model: ConstantLossLatentModel,
        n_buckets: int = 1,
    ) -> None:
        super().__init__()
        qassert.require(n_buckets > 0, f"n_buckets must be > 0, got {n_buckets}")
        self._latent_model = latent_model
        self._n_buckets = n_buckets

    def latent_model(self) -> ConstantLossLatentModel:
        return self._latent_model

    def n_buckets(self) -> int:
        return self._n_buckets

    def _compute_loss_unit_and_wk(
        self, notionals: Sequence[float]
    ) -> tuple[float, list[int]]:
        """Compute the loss-bucket unit + per-name bucket counts.

        # C++ parity: recursivelossmodel.hpp:191-212 resetModel.
        """
        recoveries = self._latent_model.recoveries()
        lgds = [notionals[i] * (1.0 - recoveries[i]) for i in range(len(notionals))]
        nonzero_lgds = [lgd for lgd in lgds if lgd > 0.0]
        qassert.require(
            len(nonzero_lgds) > 0, "all LGDs are zero — cannot bucket"
        )
        loss_unit = min(nonzero_lgds) / self._n_buckets
        # Per-name bucket count: floor(lgd / loss_unit + 0.5).
        wk = [int(lgd / loss_unit + 0.5) for lgd in lgds]
        return loss_unit, wk

    def _conditional_loss_distribution(
        self,
        probs: Sequence[float],
        notionals: Sequence[float],
        m: float,
    ) -> dict[int, float]:
        """Conditional loss distribution given factor M=m.

        Returns a dict keyed by bucket count -> probability mass.

        # C++ parity: recursivelossmodel.hpp:302-346
        # conditionalLossDistrib.
        """
        _, wk = self._compute_loss_unit_and_wk(notionals)
        # Per-name conditional default probabilities at this factor.
        cond_defs = self._latent_model.conditional_default_probability_vec(
            probs, m
        )
        dist: dict[int, float] = {0: 1.0}
        for i_name in range(len(probs)):
            p_def = cond_defs[i_name]
            new_dist: dict[int, float] = {}
            for bucket, mass in dist.items():
                # name does NOT default
                new_dist[bucket] = new_dist.get(bucket, 0.0) + mass * (1.0 - p_def)
                # name DOES default
                target = bucket + wk[i_name]
                new_dist[target] = new_dist.get(target, 0.0) + mass * p_def
            dist = new_dist
        return dist

    def unconditional_loss_distribution(
        self,
        probs: Sequence[float],
        notionals: Sequence[float],
    ) -> dict[float, float]:
        """Unconditional loss distribution — integrates over the M factor.

        Returns a dict keyed by loss amount -> probability mass.

        # C++ parity: recursivelossmodel.hpp:178-186 lossProbability.
        """
        loss_unit, _ = self._compute_loss_unit_and_wk(notionals)
        cop = self._latent_model.copula()
        cop.calculate()
        # Accumulate bucket -> prob over the M grid.
        bucket_probs: dict[int, float] = {}
        for k in range(cop.steps()):
            mk = cop.m(k)
            cond_dist = self._conditional_loss_distribution(probs, notionals, mk)
            w_k = cop.density_dm(k)
            for bucket, mass in cond_dist.items():
                bucket_probs[bucket] = bucket_probs.get(bucket, 0.0) + mass * w_k
        # Re-key by loss amount.
        return {bucket * loss_unit: mass for bucket, mass in bucket_probs.items()}

    def expected_tranche_loss(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
    ) -> float:
        """Recursive expected tranche loss.

        # C++ parity: recursivelossmodel.hpp:130-175 expectedTrancheLoss.

        The DefaultLossModel-uniform signature takes a scalar ``prob`` and
        ``average_rr``; this loss model handles only homogeneous baskets
        in that signature. For heterogeneous baskets call
        ``expected_tranche_loss_heterogeneous`` directly.
        """
        return self.expected_tranche_loss_heterogeneous(
            remaining_notional=remaining_notional,
            probs=[prob] * self._latent_model.pool_size(),
            notionals=[remaining_notional / self._latent_model.pool_size()]
            * self._latent_model.pool_size(),
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
        """Expected tranche loss given per-name unconditional probs + notionals.

        # C++ parity: recursivelossmodel.hpp:130-175 expectedTrancheLoss.
        """
        _ = remaining_notional
        dist = self.unconditional_loss_distribution(probs, notionals)
        # ETL = sum_l max(0, min(detach, l) - attach) * P(l)
        etl = 0.0
        for loss_amount, mass in dist.items():
            clipped = max(0.0, min(detach, loss_amount) - attach)
            etl += clipped * mass
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
        """Recursive percentile loss.

        # C++ parity: recursivelossmodel.hpp:232-259 percentile.
        """
        _ = average_rr
        return self.percentile_heterogeneous(
            remaining_notional=remaining_notional,
            probs=[prob] * self._latent_model.pool_size(),
            notionals=[remaining_notional / self._latent_model.pool_size()]
            * self._latent_model.pool_size(),
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
        """Percentile loss given per-name unconditional probs + notionals."""
        _ = remaining_notional
        qassert.require(
            0.0 <= perctl <= 1.0, f"perctl {perctl} out of range [0, 1]"
        )
        dist = self.unconditional_loss_distribution(probs, notionals)
        # Sort by loss amount and walk the CDF.
        sorted_items = sorted(dist.items())
        if len(sorted_items) == 1:
            return sorted_items[0][0]
        # Build cumulative distribution.
        cum_prob = 0.0
        cum: list[tuple[float, float]] = []
        for loss, mass in sorted_items:
            cum_prob += mass
            cum.append((loss, cum_prob))
        if cum[0][1] >= 1.0:
            return cum[0][0]
        if perctl == 1.0:
            return cum[-1][0]
        if perctl == 0.0:
            return cum[0][0]
        # Linear interpolation between bracket points.
        for i, (loss, cprob) in enumerate(cum):
            if cprob > perctl:
                if i == 0:
                    return loss
                x_min, val_min = cum[i - 1]
                x_plus, val_plus = loss, cprob
                portf_loss = x_plus - (x_plus - x_min) * (val_plus - perctl) / (
                    val_plus - val_min
                )
                return min(max(portf_loss - attach, 0.0), detach - attach)
        return cum[-1][0]
