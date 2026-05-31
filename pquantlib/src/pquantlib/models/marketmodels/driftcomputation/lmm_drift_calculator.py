"""LMMDriftCalculator — log-normal LIBOR market model drift.

# C++ parity:
# ql/models/marketmodels/driftcomputation/lmmdriftcalculator.{hpp,cpp}
# (v1.42.1).

Computes the per-step drift ``mu * dt`` for the log-normal LIBOR market
model. See Mark Joshi, *Rapid Computation of Drifts in a Reduced Factor
Libor Market Model*, Wilmott Magazine, May 2003.

Two algorithms:

- ``compute_plain`` — no factor reduction; uses the covariance matrix ``C``
  directly (eqs. 2, 4 of ref. [1]).
- ``compute_reduced`` — factor reduction via the pseudo-square-root of the
  covariance matrix (eq. 7 of ref. [1]).

``compute`` dispatches to ``compute_plain`` when the pseudo-root is
full-factor, else ``compute_reduced``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState


class LMMDriftCalculator:
    """Drift computation for log-normal LIBOR market models.

    # C++ parity: lmmdriftcalculator.hpp/.cpp LMMDriftCalculator.
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
        self._is_full_factor = self._number_of_factors == self._number_of_rates
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

        # Temporary buffers (mutable, reused per call) — mirror C++ mutables.
        self._tmp = [0.0] * self._number_of_rates
        # e_ is (factors, rates) in C++ (columns x rows).
        self._e: Matrix = np.zeros(
            (self._number_of_factors, self._number_of_rates), dtype=np.float64
        )

        # Lower / upper extrema for (non-reduced) drift calculation.
        self._downs = [0] * self._number_of_rates
        self._ups = [0] * self._number_of_rates
        for i in range(alive, self._number_of_rates):
            self._downs[i] = min(i + 1, numeraire)
            self._ups[i] = max(i + 1, numeraire)

    def compute(self, arg: LMMCurveState | list[float], drifts: list[float]) -> None:
        """Compute drifts, dispatching plain (full-factor) vs reduced.

        # C++ parity: LMMDriftCalculator::compute.

        ``arg`` is an ``LMMCurveState`` (forward rates pulled via
        ``forward_rates()``) or the forward-rate list directly.
        """
        fwds = self._forwards(arg)
        if self._is_full_factor:
            self.compute_plain(fwds, drifts)
        else:
            self.compute_reduced(fwds, drifts)

    def compute_plain(self, arg: LMMCurveState | list[float], drifts: list[float]) -> None:
        """Drifts without factor reduction (covariance matrix directly).

        # C++ parity: LMMDriftCalculator::computePlain.
        """
        forwards = self._forwards(arg)
        # Precompute forwards factor.
        for i in range(self._alive, self._number_of_rates):
            self._tmp[i] = (forwards[i] + self._displacements[i]) / (
                self._one_over_taus[i] + forwards[i]
            )
        # Compute drifts.
        for i in range(self._alive, self._number_of_rates):
            down = self._downs[i]
            up = self._ups[i]
            total = 0.0
            for k in range(down, up):
                total += self._tmp[k] * self._c[i, k]
            drifts[i] = total
            if self._numeraire > i + 1:
                drifts[i] = -drifts[i]

    def compute_reduced(self, arg: LMMCurveState | list[float], drifts: list[float]) -> None:
        """Drifts with factor reduction (pseudo-square-root).

        # C++ parity: LMMDriftCalculator::computeReduced.
        """
        forwards = self._forwards(arg)
        # Precompute forwards factor.
        for i in range(self._alive, self._number_of_rates):
            self._tmp[i] = (forwards[i] + self._displacements[i]) / (
                self._one_over_taus[i] + forwards[i]
            )

        # Enforce initialization.
        init_col = max(0, self._numeraire - 1)
        for r in range(self._number_of_factors):
            self._e[r, init_col] = 0.0

        # 1st step: drift at the numeraire is zero.
        if self._numeraire > 0:
            drifts[self._numeraire - 1] = 0.0

        # 2nd step: move backward from N-2 down to alive.
        for i in range(self._numeraire - 2, self._alive - 1, -1):
            drifts[i] = 0.0
            for r in range(self._number_of_factors):
                self._e[r, i] = self._e[r, i + 1] + self._tmp[i + 1] * self._pseudo[i + 1, r]
                drifts[i] -= self._e[r, i] * self._pseudo[i, r]

        # 3rd step: move forward from N up to n (excluded).
        for i in range(self._numeraire, self._number_of_rates):
            drifts[i] = 0.0
            for r in range(self._number_of_factors):
                if i == 0:
                    self._e[r, i] = self._tmp[i] * self._pseudo[i, r]
                else:
                    self._e[r, i] = self._e[r, i - 1] + self._tmp[i] * self._pseudo[i, r]
                drifts[i] += self._e[r, i] * self._pseudo[i, r]

    @staticmethod
    def _forwards(arg: LMMCurveState | list[float]) -> list[float]:
        """Resolve forward rates from a curve state or a plain list.

        # C++ parity: the two ``compute*`` overloads (LMMCurveState vs
        # vector<Rate>) collapse to one Python signature; a curve state is
        # detected by its ``forward_rates()`` accessor.
        """
        if isinstance(arg, list):
            return arg
        return arg.forward_rates()
