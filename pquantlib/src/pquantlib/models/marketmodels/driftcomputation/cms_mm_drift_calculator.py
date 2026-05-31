"""CMSMMDriftCalculator — constant-maturity-swap market model drift.

# C++ parity:
# ql/models/marketmodels/driftcomputation/cmsmmdriftcalculator.{hpp,cpp}
# (v1.42.1).

Computes the per-step drift ``mu * dt`` for the constant-maturity-swap market
model (CMS-MM). See Mark Joshi, *Rapid Computation of Drifts in a Reduced
Factor Libor Market Model*, Wilmott Magazine, May 2003. The final bond is the
numeraire; the drift is assembled from the cross-variations ``<Wk, P_j/P_n>``
(``PjPnWk``) and ``<Wk, A_j/P_n>`` (``wkaj``) over the constant-maturity-swap
geometry (each swap spans ``spanning_forwards`` forward rates), then rebased
to a general numeraire (``wkajN``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curvestates.cm_swap_curve_state import (
        CMSwapCurveState,
    )


class CMSMMDriftCalculator:
    """Drift computation for CMS market models.

    # C++ parity: cmsmmdriftcalculator.hpp/.cpp CMSMMDriftCalculator.
    """

    def __init__(
        self,
        pseudo: Matrix,
        displacements: list[float],
        taus: list[float],
        numeraire: int,
        alive: int,
        spanning_fwds: int,
    ) -> None:
        self._pseudo: Matrix = np.asarray(pseudo, dtype=np.float64)
        self._number_of_rates = len(taus)
        self._number_of_factors = self._pseudo.shape[1]
        self._numeraire = numeraire
        self._alive = alive
        self._displacements = list(displacements)
        self._spanning_fwds = spanning_fwds

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

        # Cross-variation buffers. Shapes mirror C++:
        # PjPnWk_ (factors, rates+1); wkaj_, wkajN_ (factors, rates).
        self._pjpnwk: Matrix = np.zeros(
            (self._number_of_factors, self._number_of_rates + 1), dtype=np.float64
        )
        self._wkaj: Matrix = np.zeros(
            (self._number_of_factors, self._number_of_rates), dtype=np.float64
        )
        self._wkajn: Matrix = np.zeros(
            (self._number_of_factors, self._number_of_rates), dtype=np.float64
        )

    def compute(self, cs: CMSwapCurveState, drifts: list[float]) -> None:
        """Compute the CMS-MM drifts at a CMS curve state.

        # C++ parity: CMSMMDriftCalculator::compute.
        """
        n = self._number_of_rates
        taus = cs.rate_taus()
        span = self._spanning_fwds

        # Compute cross variations (final bond is numeraire).
        for k in range(self._pjpnwk.shape[0]):
            self._pjpnwk[k, n] = 0.0
            self._wkaj[k, n - 1] = 0.0
            for j in range(n - 2, self._alive - 2, -1):
                sr = cs.cm_swap_rate(j + 1, span)
                end_index = min(j + span + 1, n)
                first = sr * self._wkaj[k, j + 1]
                second = (
                    cs.cm_swap_annuity(n, j + 1, span)
                    * (sr + self._displacements[j + 1])
                    * self._pseudo[j + 1, k]
                )
                third = self._pjpnwk[k, end_index]
                self._pjpnwk[k, j + 1] = first + second + third

                if j >= self._alive:
                    self._wkaj[k, j] = self._wkaj[k, j + 1] + self._pjpnwk[k, j + 1] * taus[j]
                    if j + span + 1 <= n:
                        self._wkaj[k, j] -= self._pjpnwk[k, end_index] * taus[end_index - 1]

        pn_over_pnum = cs.discount_ratio(n, self._numeraire)

        for j in range(self._alive, n):
            for k in range(self._number_of_factors):
                self._wkajn[k, j] = (
                    self._wkaj[k, j] * pn_over_pnum
                    - self._pjpnwk[k, self._numeraire]
                    * pn_over_pnum
                    * cs.cm_swap_annuity(self._numeraire, j, span)
                )

        for j in range(self._alive, n):
            total = 0.0
            for k in range(self._number_of_factors):
                total += self._pseudo[j, k] * self._wkajn[k, j]
            drifts[j] = total / -cs.cm_swap_annuity(self._numeraire, j, span)
