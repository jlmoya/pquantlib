"""Extended CreditRisk+ portfolio-loss model (Panjer recursion).

# C++ parity: ql/experimental/risk/creditriskplus.{hpp,cpp}
#             (recovered from 6f379f4e9~1 — see package note).

Extended CreditRisk+ model as described in [1] *Integrating Correlations*,
Risk, July 1999.  Computes the portfolio loss distribution by the
Panjer/CreditRisk+ recursion over exposure bands, given per-obligor
exposures + default probabilities + sector assignments, per-sector
relative default variances, and an inter-sector correlation matrix.

Outputs: expected loss, unexpected loss (portfolio std-dev), per-sector
exposures / EL / UL, per-obligor marginal (risk-contribution) loss, the
discrete loss distribution, and the loss quantile.

# C++ parity divergence — the original C++ class is marked
# ``[[deprecated]]`` ("out of scope; copy into your codebase if needed");
# PQuantLib ports it intact (no deprecation marker) since the W8-B brief
# requests it as in-scope functionality.  ``Matrix correlation`` maps onto
# a ``numpy.ndarray`` of shape (n_sectors, n_sectors); the warning that the
# correlation matrix is not checked for positive-definiteness carries over.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pquantlib import qassert


class CreditRiskPlus:
    """Extended CreditRisk+ portfolio-loss model.

    # C++ parity: ``class CreditRiskPlus`` (creditriskplus.{hpp,cpp},
    # recovered pre-1.36 source).
    """

    def __init__(
        self,
        exposure: Sequence[float],
        default_probability: Sequence[float],
        sector: Sequence[int],
        relative_default_variance: Sequence[float],
        correlation: npt.NDArray[np.float64],
        unit: float,
    ) -> None:
        self._exposure: list[float] = list(exposure)
        self._pd: list[float] = list(default_probability)
        self._sector: list[int] = list(sector)
        self._relative_default_variance: list[float] = list(relative_default_variance)
        self._correlation: npt.NDArray[np.float64] = np.asarray(correlation, dtype=np.float64)
        self._unit: float = unit

        self._m: int = len(self._exposure)  # number of exposures
        qassert.require(self._m > 0, "no exposures given")
        qassert.require(
            self._m == len(self._pd),
            f"number of exposures ({self._m}) must be equal to number of pds ({len(self._pd)})",
        )
        qassert.require(
            self._m == len(self._sector),
            f"number of exposures ({self._m}) must be equal to number of "
            f"exposure sectors ({len(self._sector)})",
        )

        self._n: int = self._correlation.shape[0]  # number of sectors
        qassert.require(
            self._correlation.shape[1] == self._n,
            f"correlation matrix ({self._n},{self._correlation.shape[1]}) must be a square matrix",
        )
        qassert.require(
            len(self._relative_default_variance) == self._n,
            f"number of relative default variances ({len(self._relative_default_variance)}) "
            f"must be equal to number of sectors ({self._n})",
        )

        self._exposure_sum: float = 0.0
        self._el: float = 0.0
        self._el2: float = 0.0
        for i in range(self._m):
            qassert.require(
                self._exposure[i] >= 0.0, f"exposure #{i} is negative ({self._exposure[i]})"
            )
            qassert.require(self._pd[i] > 0.0, f"pd #{i} is negative ({self._pd[i]})")
            qassert.require(
                self._sector[i] < self._n,
                f"sector #{i} ({self._sector[i]}) is out of range 0...{self._n - 1}",
            )
            self._exposure_sum += self._exposure[i]
            self._el += self._pd[i] * self._exposure[i]
            self._el2 += self._pd[i] * self._exposure[i] * self._exposure[i]

        qassert.require(self._unit > 0.0, f"loss unit ({self._unit}) must be positive")

        # Populated by _compute.
        self._sector_exposure: list[float] = []
        self._sector_el: list[float] = []
        self._sector_ul: list[float] = []
        self._marginal_loss: list[float] = []
        self._loss: list[float] = []
        self._ul: float = 0.0
        self._upper_index: int = 0

        self._compute()

    # ------------------------------------------------------------------
    # Inspectors
    # ------------------------------------------------------------------

    def loss(self) -> list[float]:
        return self._loss

    def marginal_loss(self) -> list[float]:
        return self._marginal_loss

    def exposure(self) -> float:
        return self._exposure_sum

    def expected_loss(self) -> float:
        return self._el

    def unexpected_loss(self) -> float:
        return self._ul

    def relative_default_variance(self) -> float:
        # C++ parity: creditriskplus.hpp inline.
        return (self.unexpected_loss() ** 2 - self._el2) / (self.expected_loss() ** 2)

    def sector_exposures(self) -> list[float]:
        return self._sector_exposure

    def sector_expected_loss(self) -> list[float]:
        return self._sector_el

    def sector_unexpected_loss(self) -> list[float]:
        return self._sector_ul

    def loss_quantile(self, p: float) -> float:
        # C++ parity: creditriskplus.cpp ``lossQuantile``.
        i = 0
        total = self._loss[0]
        while i < self._upper_index - 1 and total < p:
            i += 1
            total += self._loss[i]
        if self._loss[0] >= p:
            return 0.0
        p1 = total - self._loss[i]
        p2 = total if total >= p else 1.0
        l1 = (i - 1) * self._unit
        l2 = i * self._unit
        return l1 + (p - p1) / (p2 - p1) * (l2 - l1)

    # ------------------------------------------------------------------
    # Core computation (Panjer recursion)
    # ------------------------------------------------------------------

    def _compute(self) -> None:  # noqa: PLR0915 - faithful 1:1 port of one C++ method
        # C++ parity: creditriskplus.cpp ``compute``.
        n = self._n
        m = self._m

        sector_pd_sum = [0.0] * n
        self._sector_exposure = [0.0] * n
        self._sector_el = [0.0] * n
        sector_spec_terms = [0.0] * n
        self._sector_ul = [0.0] * n
        self._marginal_loss = [0.0] * m

        pd_adj = [0.0] * m

        # --- exposure bands ---
        max_nu = 0
        self._upper_index = 0
        # map of exposure-band (nuC) -> expected loss
        eps_nu_c: dict[int, float] = {}

        for k in range(m):
            ex_unit = math.floor(0.5 + self._exposure[k] / self._unit)  # round
            if self._exposure[k] > 0 and ex_unit == 0:
                ex_unit = 1  # avoid zero exposure
            max_nu = max(max_nu, ex_unit)
            pd_adj[k] = (
                self._exposure[k] * self._pd[k] / (ex_unit * self._unit)
                if self._exposure[k] > 0.0
                else 0.0
            )
            el = ex_unit * pd_adj[k]
            if ex_unit > 0:
                eps_nu_c[ex_unit] = eps_nu_c.get(ex_unit, 0.0) + el
                self._upper_index += ex_unit

        # --- per-sector figures ---
        pd_sum = 0.0
        for k in range(m):
            pd_sum += pd_adj[k]
            sector_pd_sum[self._sector[k]] += self._pd[k]
            self._sector_exposure[self._sector[k]] += self._exposure[k]
            self._sector_el[self._sector[k]] += self._exposure[k] * self._pd[k]

        # sector-specific terms (formula 15 in [1])
        for i in range(n):
            sector_spec_terms[i] += self._relative_default_variance[i] * self._sector_el[i]
            for j in range(n):
                if j != i:
                    sector_spec_terms[i] += (
                        self._correlation[i][j]
                        * math.sqrt(
                            self._relative_default_variance[i]
                            * self._relative_default_variance[j]
                        )
                        * self._sector_el[j]
                    )

        # synthetic standard deviation (formula 12 in [1])
        self._ul = 0.0
        for i in range(n):
            self._sector_ul[i] = (
                self._relative_default_variance[i] * self._sector_el[i] * self._sector_el[i]
            )
            self._ul += self._sector_ul[i]
            for j in range(n):
                if j != i:
                    self._ul += (
                        self._correlation[i][j]
                        * math.sqrt(
                            self._relative_default_variance[i]
                            * self._relative_default_variance[j]
                        )
                        * self._sector_el[i]
                        * self._sector_el[j]
                    )

        match_ul = self._ul  # formula 13 in [1], rhs
        for k in range(m):
            tmp = self._pd[k] * self._exposure[k] * self._exposure[k]
            self._sector_ul[self._sector[k]] += tmp
            self._ul += tmp
        self._ul = math.sqrt(self._ul)
        for i in range(n):
            self._sector_ul[i] = math.sqrt(self._sector_ul[i])

        # risk contributions (formula 15 in [1])
        for k in range(m):
            self._marginal_loss[k] = (
                self._pd[k]
                * self._exposure[k]
                / self._ul
                * (sector_spec_terms[self._sector[k]] + self._exposure[k])
            )

        # sigmaC and deduced figures
        sigma_c = pd_sum * math.sqrt(match_ul / (self._el * self._el))
        alpha_c = pd_sum * pd_sum / (sigma_c * sigma_c)
        beta_c = sigma_c * sigma_c / pd_sum
        p_c = beta_c / (1.0 + beta_c)

        # loss distribution
        self._loss = []
        self._loss.append((1.0 - p_c) ** alpha_c)  # A(0)

        for nn in range(self._upper_index - 1):  # compute A(n+1) recursively
            res = 0.0
            for j in range(min(max_nu - 1, nn) + 1):
                eps = eps_nu_c.get(j + 1)
                if eps is not None:
                    res += eps * self._loss[nn - j] * alpha_c
                    if j <= nn - 1:
                        res += eps / (j + 1) * (nn - j) * self._loss[nn - j]
            self._loss.append(res * p_c / (pd_sum * (nn + 1)))


__all__ = ["CreditRiskPlus"]
