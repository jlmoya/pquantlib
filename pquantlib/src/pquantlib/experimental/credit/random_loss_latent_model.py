"""RandomLossLatentModel — Monte-Carlo default + loss-amount sampling.

# C++ parity: ql/experimental/credit/randomlosslatentmodel.hpp @ v1.42.1.

Extends RandomDefaultLatentModel by sampling per-default loss amounts.
The default-only model determines *which* names default; this model
additionally samples the conditional LGD by drawing from the RR latent
variable distribution (constant or spot-recovery, supplied by the
``recovery_model`` argument).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula
from pquantlib.math.randomnumbers.mersenne_twister import (
    MersenneTwisterUniformRng,
)


class _ConditionalRecoveryDraw(Protocol):
    """Callable returning (random_recovery_value) given (i_name, m, u_rr) ∈ [0,1]."""

    def __call__(self, i_name: int, m: float, u_rr: float) -> float: ...


class RandomLossLatentModel:
    """Monte-Carlo loss-event sampler — extends RandomDefault with LGD draws.

    The recovery_draw callable should map (i_name, m, u_rr ∈ [0,1]) to a
    recovery value in [0, 1]. The model passes it the uniform draw used
    for that path's RR variable so the caller can pick its own
    distribution (constant ignores u_rr; spot uses
    ``CumulativeNormalDistribution`` with the model A noise; users may
    provide arbitrary functionals).

    For convenience the module ships ``constant_recovery_draw`` which
    closes over a per-name recovery vector and ignores u_rr.
    """

    __slots__ = ("_copula", "_n", "_notionals", "_recovery_draw", "_seed")

    def __init__(
        self,
        copula: OneFactorCopula,
        notionals: Sequence[float],
        recovery_draw: _ConditionalRecoveryDraw,
        seed: int = 42,
    ) -> None:
        qassert.require(
            len(notionals) > 0,
            "notionals must be non-empty",
        )
        self._copula = copula
        self._notionals = list(notionals)
        self._recovery_draw = recovery_draw
        self._n = len(notionals)
        self._seed = seed

    def copula(self) -> OneFactorCopula:
        return self._copula

    def pool_size(self) -> int:
        return self._n

    def notionals(self) -> list[float]:
        return list(self._notionals)

    def simulate_loss_distribution(
        self,
        probs: Sequence[float],
        n_paths: int,
    ) -> list[float]:
        """Return ``n_paths`` total-loss values, one per path."""
        qassert.require(
            len(probs) == self._n,
            f"probs size {len(probs)} != pool_size {self._n}",
        )
        self._copula.calculate()
        rng = MersenneTwisterUniformRng(self._seed)
        thresholds = [self._copula.inverse_cumulative_y(p) for p in probs]
        rho = self._copula.correlation()
        sqrt_rho = float(np.sqrt(rho))
        sqrt_1mr = float(np.sqrt(1.0 - rho))
        losses: list[float] = [0.0] * n_paths
        for path in range(n_paths):
            u_m = rng.next().value
            m = self._copula.inverse_cumulative_y(u_m)
            total = 0.0
            for i in range(self._n):
                u_z = rng.next().value
                z_i = self._copula.inverse_cumulative_y(u_z)
                y_i = sqrt_rho * m + sqrt_1mr * z_i
                if y_i < thresholds[i]:
                    u_rr = rng.next().value
                    rr_draw = self._recovery_draw(i, m, u_rr)
                    total += self._notionals[i] * (1.0 - rr_draw)
                else:
                    # advance RR draw to keep RNG stream deterministic
                    _ = rng.next()
            losses[path] = total
        return losses

    def expected_loss_mc(
        self,
        probs: Sequence[float],
        n_paths: int,
    ) -> float:
        """MC estimate of expected total loss."""
        losses = self.simulate_loss_distribution(probs, n_paths)
        return sum(losses) / float(n_paths)


def constant_recovery_draw(
    recoveries: Sequence[float],
) -> _ConditionalRecoveryDraw:
    """Build a constant-recovery draw closure for ``RandomLossLatentModel``.

    The closure ignores ``m`` and ``u_rr`` and returns the stored per-name
    recovery — i.e. the deterministic ``ConstantLossLatentModel`` behavior.
    """
    rs = list(recoveries)

    def _draw(i_name: int, m: float, u_rr: float) -> float:
        _ = m  # ignored — constant model
        _ = u_rr
        return rs[i_name]

    return _draw
