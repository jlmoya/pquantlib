"""RandomDefaultLatentModel — Monte-Carlo default-time sampling.

# C++ parity: ql/experimental/credit/randomdefaultlatentmodel.hpp @ v1.42.1.

Generates Monte-Carlo samples of (default-or-not) for each name in the
pool by drawing a common factor + per-name idiosyncratic factors from
the underlying one-factor copula, then comparing the resulting Y_i
against the inverse-CDF of each name's unconditional default probability.

The C++ template is parameterised on a copula policy and uses the
LatentModel sampling infrastructure; the Python port uses the
mersenne_twister RNG to draw uniform samples that get inverse-transformed
into copula draws (via ``inverseCumulativeY``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula
from pquantlib.math.randomnumbers.mersenne_twister import (
    MersenneTwisterUniformRng,
)


class RandomDefaultLatentModel:
    """Monte-Carlo default-event sampler.

    Each draw selects:
      1. A common factor draw ``m`` from ``F_Y`` (via the copula's
         ``inverse_cumulative_y`` of a uniform).
      2. Per-name idiosyncratic draws ``z_i`` from ``F_Z``.
      3. Builds ``y_i = sqrt(rho) m + sqrt(1 - rho) z_i``.
      4. Marks name ``i`` defaulted iff ``y_i < F_Y^-1(p_i)``.

    Returns the simulated default-count distribution per draw.

    # C++ parity divergence: the C++ template uses LatentModel sampling
    # primitives that pre-compute inverses for the copula; the Python
    # port inlines the simpler one-factor draw using mersenne_twister.
    """

    __slots__ = ("_copula", "_n", "_seed")

    def __init__(
        self,
        copula: OneFactorCopula,
        pool_size: int,
        seed: int = 42,
    ) -> None:
        qassert.require(pool_size > 0, f"pool_size must be > 0, got {pool_size}")
        self._copula = copula
        self._n = pool_size
        self._seed = seed

    def copula(self) -> OneFactorCopula:
        return self._copula

    def pool_size(self) -> int:
        return self._n

    def simulate_default_counts(
        self,
        probs: Sequence[float],
        n_paths: int,
    ) -> list[int]:
        """Return a list of ``n_paths`` default counts (0..pool_size).

        For each path: draw common factor, draw n idiosyncratic factors,
        count the number of names where Y_i < F_Y^-1(p_i).
        """
        qassert.require(
            len(probs) == self._n,
            f"probs size {len(probs)} != pool_size {self._n}",
        )
        self._copula.calculate()
        rng = MersenneTwisterUniformRng(self._seed)
        # Pre-compute the per-name default thresholds.
        thresholds = [self._copula.inverse_cumulative_y(p) for p in probs]
        rho = self._copula.correlation()
        sqrt_rho = float(np.sqrt(rho))
        sqrt_1mr = float(np.sqrt(1.0 - rho))
        counts: list[int] = [0] * n_paths
        for path in range(n_paths):
            # Draw uniform for M -> inverse_cumulative_y -> m.
            u_m = rng.next().value
            m = self._copula.inverse_cumulative_y(u_m)
            count = 0
            for i in range(self._n):
                u_z = rng.next().value
                z_i = self._copula.inverse_cumulative_y(u_z)
                y_i = sqrt_rho * m + sqrt_1mr * z_i
                if y_i < thresholds[i]:
                    count += 1
            counts[path] = count
        return counts

    def prob_at_least_n_events_mc(
        self,
        n: int,
        probs: Sequence[float],
        n_paths: int,
    ) -> float:
        """MC estimate of P(>= n defaults).

        # C++ parity: defaultprobabilitylatentmodel.hpp:probAtLeastNEvents
        # via MC sampling.
        """
        counts = self.simulate_default_counts(probs, n_paths)
        return sum(1 for c in counts if c >= n) / float(n_paths)
