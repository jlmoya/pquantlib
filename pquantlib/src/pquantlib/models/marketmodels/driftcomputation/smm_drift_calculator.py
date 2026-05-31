"""SMMDriftCalculator — coterminal-swap market model drift.

# C++ parity:
# ql/models/marketmodels/driftcomputation/smmdriftcalculator.{hpp,cpp}
# (v1.42.1).

Computes the per-step drift ``mu * dt`` for the coterminal-swap market model
(SMM). See Mark Joshi, Lorenzo Liesch, *Effective Implementation Of Generic
Market Models*. The terminal bond is the reference measure; the drift is
assembled from the cross-variations ``<W(k) | A(j)/P(n)>`` (``wkaj``) and
``<W(k) | P(j)/P(n)>`` (``wkpj``), then rebased to a general numeraire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curvestates.coterminal_swap_curve_state import (
        CoterminalSwapCurveState,
    )


class SMMDriftCalculator:
    """Drift computation for coterminal-swap market models.

    # C++ parity: smmdriftcalculator.hpp/.cpp SMMDriftCalculator.
    """

    def __init__(
        self,
        pseudo: Matrix,
        displacements: list[float],
        taus: list[float],
        numeraire: int,
        alive: int,
    ) -> None:
        self._pseudo: Matrix = np.asarray(pseudo, dtype=np.float64)
        self._number_of_rates = len(taus)
        self._number_of_factors = self._pseudo.shape[1]
        self._numeraire = numeraire
        self._alive = alive
        self._displacements = list(displacements)

        # Check requirements
        qassert.require(self._number_of_rates > 0, "Dim out of range")
        qassert.require(
            len(displacements) == self._number_of_rates,
            "Displacements out of range",
        )
        qassert.require(
            self._pseudo.shape[0] == self._number_of_rates,
            "pseudo.rows() not consistent with dim",
        )
        qassert.require(
            0 < self._number_of_factors <= self._number_of_rates,
            "pseudo.rows() not consistent with pseudo.columns()",
        )
        qassert.require(alive < self._number_of_rates, "Alive out of bounds")
        qassert.require(numeraire <= self._number_of_rates, "Numeraire larger than dim")
        qassert.require(numeraire >= alive, "Numeraire smaller than alive")

        # Precompute 1/taus
        self._one_over_taus = [1.0 / t for t in taus]

        # Compute covariance matrix from pseudoroot
        self._c: Matrix = self._pseudo @ self._pseudo.T

        # Cross-variation buffers (zero initialization required for the last
        # element). Shapes mirror C++: (factors, rates) and (factors, rates+1).
        self._wkaj: Matrix = np.zeros(
            (self._number_of_factors, self._number_of_rates), dtype=np.float64
        )
        self._wkpj: Matrix = np.zeros(
            (self._number_of_factors, self._number_of_rates + 1), dtype=np.float64
        )
        self._wkajshifted: Matrix = np.zeros(
            (self._number_of_factors, self._number_of_rates), dtype=np.float64
        )

    def compute(self, cs: CoterminalSwapCurveState, drifts: list[float]) -> None:
        """Compute the SMM drifts at a coterminal-swap curve state.

        # C++ parity: SMMDriftCalculator::compute.
        """
        n = self._number_of_rates
        sr = cs.coterminal_swap_rates()
        taus = cs.rate_taus()
        annuities = [cs.coterminal_swap_annuity(n, j) for j in range(n)]

        # calculates and stores wkaj_, wkpj_ assuming terminal bond measure
        # (eq 5.4-5.7); last column already zero from construction.
        for k in range(self._number_of_factors):
            for j in range(n - 2, self._alive - 2, -1):
                # < W(k) | P(j+1)/P(n) >
                annuity = annuities[j + 1]
                self._wkpj[k, j + 1] = (
                    sr[j + 1] * (self._pseudo[j + 1, k] * annuity + self._wkaj[k, j + 1])
                    + self._pseudo[j + 1, k] * self._displacements[j + 1] * annuity
                )
                if j >= self._alive:
                    self._wkaj[k, j] = self._wkpj[k, j + 1] * taus[j] + self._wkaj[k, j + 1]

        numeraire_ratio = cs.discount_ratio(n, self._numeraire)

        # change to work for general numeraire
        for k in range(self._number_of_factors):
            for j in range(self._alive, n):
                self._wkajshifted[k, j] = (
                    -self._wkaj[k, j] / annuities[j]
                    + self._wkpj[k, self._numeraire] * numeraire_ratio
                )

        # eq 5.3 (in log coordinates)
        for j in range(self._alive, n):
            total = 0.0
            for k in range(self._number_of_factors):
                total += self._wkajshifted[k, j] * self._pseudo[j, k]
            drifts[j] = total
