"""AbcdVol — Abcd-parametric volatility MarketModel.

# C++ parity: ql/models/marketmodels/models/abcdvol.{hpp,cpp} (v1.42.1).

Like ``FlatVol`` but the per-rate instantaneous volatility follows the Rebonato
abcd functional ``f(T-t) = [a + b(T-t)] e^{-c(T-t)} + d`` (shared across all
rates), scaled per rate by ``ks[i]``. The per-step covariance integrates the
abcd covariance over each correlation sub-interval, then takes the spectral
rank-reduced pseudo-square-root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.market_model import MarketModel
from pquantlib.models.marketmodels.models.abcd_function import AbcdFunction
from pquantlib.models.marketmodels.models.pseudo_sqrt import (
    SalvagingAlgorithm,
    rank_reduced_sqrt,
)

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
    from pquantlib.models.marketmodels.piecewise_constant_correlation import (
        PiecewiseConstantCorrelation,
    )


class AbcdVol(MarketModel):
    """Abcd-parametric volatility market model.

    # C++ parity: abcdvol.hpp/.cpp AbcdVol.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        ks: list[float],
        correlations: PiecewiseConstantCorrelation,
        evolution: EvolutionDescription,
        number_of_factors: int,
        initial_rates: list[float],
        displacements: list[float],
    ) -> None:
        super().__init__()
        self._number_of_factors = number_of_factors
        self._number_of_rates = len(initial_rates)
        self._number_of_steps = len(evolution.evolution_times())
        self._initial_rates = list(initial_rates)
        self._displacements = list(displacements)
        self._evolution = evolution

        n = self._number_of_rates
        rate_times = evolution.rate_times()
        qassert.require(
            n == len(rate_times) - 1,
            f"mismatch between number of rates ({n}) and rate times",
        )
        qassert.require(
            n == len(displacements),
            f"mismatch between number of rates ({n}) and displacements "
            f"({len(displacements)})",
        )
        qassert.require(
            n == len(ks),
            f"mismatch between number of rates ({n}) and ks ({len(ks)})",
        )
        # C++ parity: the numberOfRates <= numberOfFactors*numberOfSteps check
        # is commented out in abcdvol.cpp; we keep it commented out too.
        qassert.require(
            number_of_factors <= n,
            f"number of factors ({number_of_factors}) cannot be greater than "
            f"numberOfRates ({n})",
        )
        qassert.require(
            number_of_factors > 0,
            f"number of factors ({number_of_factors}) must be greater than zero",
        )

        abcd = AbcdFunction(a, b, c, d)
        self._pseudo_roots: list[Matrix] = []
        eff_stop_time = 0.0
        corr_times = correlations.times()
        evol_times = evolution.evolution_times()
        # C++ parity: single running ``kk`` index across all steps.
        kk = 0
        for k in range(self._number_of_steps):
            covariance = np.zeros((n, n), dtype=np.float64)
            while corr_times[kk] < evol_times[k]:
                eff_start_time = eff_stop_time
                eff_stop_time = corr_times[kk]
                corr_matrix = correlations.correlation(kk)
                self._accumulate(
                    covariance, abcd, eff_start_time, eff_stop_time, rate_times,
                    ks, corr_matrix, n,
                )
                kk += 1
            # last part in the evolution step
            eff_start_time = eff_stop_time
            eff_stop_time = evol_times[k]
            corr_matrix = correlations.correlation(kk)
            self._accumulate(
                covariance, abcd, eff_start_time, eff_stop_time, rate_times,
                ks, corr_matrix, n,
            )
            # no more use for the kk-th correlation matrix
            while kk < len(corr_times) and corr_times[kk] <= evol_times[k]:
                kk += 1

            # make it symmetric (we only filled the upper triangle)
            for i in range(n):
                for j in range(i + 1, n):
                    covariance[j, i] = covariance[i, j]

            pseudo_root = rank_reduced_sqrt(
                covariance, number_of_factors, 1.0, SalvagingAlgorithm.NONE
            )
            qassert.require(
                pseudo_root.shape[0] == n,
                f"step {k} abcd vol wrong number of rows: {pseudo_root.shape[0]} "
                f"instead of {n}",
            )
            qassert.require(
                pseudo_root.shape[1] == number_of_factors,
                f"step {k} abcd vol wrong number of columns: "
                f"{pseudo_root.shape[1]} instead of {number_of_factors}",
            )
            self._pseudo_roots.append(pseudo_root)

    @staticmethod
    def _accumulate(
        covariance: Matrix,
        abcd: AbcdFunction,
        eff_start_time: float,
        eff_stop_time: float,
        rate_times: list[float],
        ks: list[float],
        corr_matrix: Matrix,
        n: int,
    ) -> None:
        """Add the abcd covariance over ``[start, stop]`` (upper triangle).

        # C++ parity: the inner i/j double-loop body in abcdvol.cpp
        (``cov = ks[i]*ks[j]*abcd.covariance(...)``).
        """
        for i in range(n):
            for j in range(i, n):
                cov = ks[i] * ks[j] * abcd.covariance(
                    eff_start_time, eff_stop_time, rate_times[i], rate_times[j]
                )
                covariance[i, j] += cov * corr_matrix[i, j]

    def initial_rates(self) -> list[float]:
        """The initial forward rates."""
        return self._initial_rates

    def displacements(self) -> list[float]:
        """The displacement (shift) of each rate."""
        return self._displacements

    def evolution(self) -> EvolutionDescription:
        """The evolution description."""
        return self._evolution

    def number_of_rates(self) -> int:
        """Number of forward rates."""
        return self._number_of_rates

    def number_of_factors(self) -> int:
        """Number of driving factors."""
        return self._number_of_factors

    def number_of_steps(self) -> int:
        """Number of evolution steps."""
        return self._number_of_steps

    def pseudo_root(self, i: int) -> Matrix:
        """Pseudo-square-root of the covariance matrix for step ``i``.

        # C++ parity: AbcdVol::pseudoRoot.
        """
        qassert.require(
            i < self._number_of_steps,
            f"the index {i} is invalid: it must be less than number of steps "
            f"({self._number_of_steps})",
        )
        return self._pseudo_roots[i]
