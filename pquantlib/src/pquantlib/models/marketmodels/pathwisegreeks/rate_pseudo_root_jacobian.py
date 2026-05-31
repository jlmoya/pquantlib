"""RatePseudoRootJacobian — d(evolved rates) / d(pseudo-root element).

# C++ parity: ql/models/marketmodels/pathwisegreeks/ratepseudorootjacobian.
# {hpp,cpp} (v1.42.1).

Computes the derivative of the map taking forward rates one step to the next
with respect to a change in the pseudo-root. Evolution is log-Euler. Three
classes are provided:

- ``RatePseudoRootJacobian`` — the analytic Jacobian contracted against a set
  of pseudo-root bumps (the GG-paper page-95 "B" matrix: one row per bump,
  one column per rate).
- ``RatePseudoRootJacobianAllElements`` — the full element-wise Jacobian (one
  ``Matrix`` per rate; entry ``[k][f]`` is the derivative of that rate with
  respect to pseudo-root element ``[k][f]``).
- ``RatePseudoRootJacobianNumerical`` — the finite-difference cross-check
  (bumps the pseudo-root and re-evolves), used by the C++ test-suite to
  validate the analytic versions.

The ``Matrix`` type is numpy 2-D float64 throughout, matching the C++
``Matrix`` in/out parameters (the ``B`` argument is mutated in place).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.models.marketmodels.driftcomputation.lmm_drift_calculator import (
    LMMDriftCalculator,
)

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix


class RatePseudoRootJacobianNumerical:
    """Finite-difference Jacobian of the one-step log-Euler evolution.

    # C++ parity: RatePseudoRootJacobianNumerical.
    """

    def __init__(
        self,
        pseudo_root: Matrix,
        alive_index: int,
        numeraire: int,
        taus: list[float],
        pseudo_bumps: list[Matrix],
        displacements: list[float],
    ) -> None:
        # C++ parity: RatePseudoRootJacobianNumerical ctor.
        self._pseudo_root = np.asarray(pseudo_root, dtype=np.float64)
        self._alive_index = alive_index
        self._taus = list(taus)
        self._displacements = list(displacements)
        self._number_bumps = len(pseudo_bumps)
        self._factors = self._pseudo_root.shape[1]
        number_rates = len(taus)
        self._drifts = [0.0] * number_rates
        self._bumped_rates = [0.0] * number_rates

        qassert.require(
            self._pseudo_root.shape[0] == number_rates,
            "pseudoRoot_.rows()<> taus.size()",
        )
        qassert.require(
            len(self._displacements) == number_rates,
            "displacements_.size()<> taus.size()",
        )

        self._pseudo_bumped: list[Matrix] = []
        self._drifts_computers: list[LMMDriftCalculator] = []
        for i, bump in enumerate(pseudo_bumps):
            b = np.asarray(bump, dtype=np.float64)
            qassert.require(
                b.shape[0] == number_rates,
                f"pseudoBumps[i].rows()<> taus.size() with i ={i}",
            )
            qassert.require(
                b.shape[1] == self._factors,
                f"pseudoBumps[i].columns()<> factors with i = {i}",
            )
            pseudo = self._pseudo_root + b
            self._pseudo_bumped.append(pseudo)
            self._drifts_computers.append(
                LMMDriftCalculator(
                    pseudo, displacements, taus, numeraire, alive_index
                )
            )

    def get_bumps(
        self,
        old_rates: list[float],
        one_step_dfs: list[float],
        new_rates: list[float],
        gaussians: list[float],
        b_out: Matrix,  # C++ parity: Matrix& B (mutated in place)
    ) -> None:
        # C++ parity: RatePseudoRootJacobianNumerical::getBumps.
        number_rates = len(self._taus)
        qassert.require(b_out.shape[0] == self._number_bumps, "B.rows()<> numberBumps_")
        qassert.require(
            b_out.shape[1] == number_rates, "B.columns()<> number of rates"
        )

        for i in range(self._number_bumps):
            pseudo = self._pseudo_bumped[i]
            self._drifts_computers[i].compute(list(old_rates), self._drifts)

            for j in range(self._alive_index):
                b_out[i][j] = 0.0

            for j in range(self._alive_index, number_rates):
                acc = math.log(old_rates[j] + self._displacements[j])
                for k in range(self._factors):
                    acc += -0.5 * pseudo[j][k] * pseudo[j][k]
                acc += self._drifts[j]
                for k in range(self._factors):
                    acc += pseudo[j][k] * gaussians[k]
                val = math.exp(acc) - self._displacements[j]
                self._bumped_rates[j] = val
                b_out[i][j] = val - new_rates[j]


class RatePseudoRootJacobian:
    """Analytic Jacobian contracted against pseudo-root bumps.

    # C++ parity: RatePseudoRootJacobian.
    """

    def __init__(
        self,
        pseudo_root: Matrix,
        alive_index: int,
        numeraire: int,
        taus: list[float],
        pseudo_bumps: list[Matrix],
        displacements: list[float],
    ) -> None:
        # C++ parity: RatePseudoRootJacobian ctor.
        self._pseudo_root = np.asarray(pseudo_root, dtype=np.float64)
        self._alive_index = alive_index
        self._taus = list(taus)
        self._pseudo_bumps = [np.asarray(b, dtype=np.float64) for b in pseudo_bumps]
        self._displacements = list(displacements)
        self._number_bumps = len(pseudo_bumps)
        self._factors = self._pseudo_root.shape[1]
        number_rates = len(taus)

        qassert.require(
            alive_index == numeraire,
            "we can do only do discretely compounding MM acount so aliveIndex "
            "must equal numeraire",
        )
        qassert.require(
            self._pseudo_root.shape[0] == number_rates,
            "pseudoRoot_.rows()<> taus.size()",
        )
        qassert.require(
            len(self._displacements) == number_rates,
            "displacements_.size()<> taus.size()",
        )
        for i, b in enumerate(self._pseudo_bumps):
            qassert.require(
                b.shape[0] == number_rates,
                f"pseudoBumps[i].rows()<> taus.size() with i ={i}",
            )
            qassert.require(
                b.shape[1] == self._factors,
                f"pseudoBumps[i].columns()<> factors with i = {i}",
            )

        # workspace
        # C++ parity: std::vector<Matrix> allDerivatives_ (one (rates,factors)
        # Matrix per rate); Matrix e_; std::vector<Real> ratios_.
        self._all_derivatives: list[Matrix] = [
            np.zeros((number_rates, self._factors), dtype=np.float64)
            for _ in range(number_rates)
        ]
        self._e: Matrix = np.zeros(
            (self._pseudo_root.shape[0], self._pseudo_root.shape[1]), dtype=np.float64
        )
        self._ratios = [0.0] * number_rates

    def get_bumps(
        self,
        old_rates: list[float],
        discount_ratios: list[float],
        new_rates: list[float],
        gaussians: list[float],
        b_out: Matrix,  # C++ parity: Matrix& B (mutated in place)
    ) -> None:
        # C++ parity: RatePseudoRootJacobian::getBumps.
        number_rates = len(self._taus)
        alive = self._alive_index
        factors = self._factors
        pr = self._pseudo_root
        e = self._e
        ratios = self._ratios
        all_d = self._all_derivatives

        qassert.require(
            b_out.shape[0] == self._number_bumps,
            f"we need B.rows() which is {b_out.shape[0]} to equal numberBumps_ "
            f"which is {self._number_bumps}",
        )
        qassert.require(
            b_out.shape[1] == number_rates,
            f"we need B.columns() which is {b_out.shape[1]} to equal numberRates "
            f"which is {number_rates}",
        )

        for j in range(alive, number_rates):
            ratios[j] = (old_rates[j] + self._displacements[j]) * discount_ratios[j + 1]

        for f in range(factors):
            e[alive][f] = 0.0
            for j in range(alive + 1, number_rates):
                e[j][f] = e[j - 1][f] + ratios[j - 1] * pr[j - 1][f]

        for f in range(factors):
            for j in range(alive, number_rates):
                for k in range(alive, j):
                    all_d[j][k][f] = (
                        new_rates[j] * ratios[k] * self._taus[k] * pr[j][f]
                    )
                # GG don't seem to have the 2, this term is miniscule in any case
                tmp = 2.0 * ratios[j] * self._taus[j] * pr[j][f]
                tmp -= pr[j][f]
                tmp += e[j][f] * self._taus[j]
                tmp += gaussians[f]
                tmp *= new_rates[j] + self._displacements[j]
                all_d[j][j][f] = tmp
                for k in range(j + 1, number_rates):
                    all_d[j][k][f] = 0.0

        for i in range(self._number_bumps):
            bump_i = self._pseudo_bumps[i]
            for j in range(alive):
                b_out[i][j] = 0.0
            for j in range(alive, number_rates):
                total = 0.0
                for k in range(alive, number_rates):
                    for f in range(factors):
                        total += bump_i[k][f] * all_d[j][k][f]
                b_out[i][j] = total


class RatePseudoRootJacobianAllElements:
    """Full element-wise Jacobian (one ``Matrix`` per rate).

    # C++ parity: RatePseudoRootJacobianAllElements.
    """

    def __init__(
        self,
        pseudo_root: Matrix,
        alive_index: int,
        numeraire: int,
        taus: list[float],
        displacements: list[float],
    ) -> None:
        # C++ parity: RatePseudoRootJacobianAllElements ctor.
        self._pseudo_root = np.asarray(pseudo_root, dtype=np.float64)
        self._alive_index = alive_index
        self._taus = list(taus)
        self._displacements = list(displacements)
        self._factors = self._pseudo_root.shape[1]
        number_rates = len(taus)

        qassert.require(
            alive_index == numeraire,
            "we can do only do discretely compounding MM acount so aliveIndex "
            "must equal numeraire",
        )
        qassert.require(
            self._pseudo_root.shape[0] == number_rates,
            "pseudoRoot_.rows()<> taus.size()",
        )
        qassert.require(
            len(self._displacements) == number_rates,
            "displacements_.size()<> taus.size()",
        )

        # workspace
        self._e: Matrix = np.zeros(
            (self._pseudo_root.shape[0], self._pseudo_root.shape[1]), dtype=np.float64
        )
        self._ratios = [0.0] * number_rates

    def get_bumps(
        self,
        old_rates: list[float],
        discount_ratios: list[float],
        new_rates: list[float],
        gaussians: list[float],
        b_out: list[Matrix],  # C++ parity: std::vector<Matrix>& B (one per rate)
    ) -> None:
        # C++ parity: RatePseudoRootJacobianAllElements::getBumps.
        number_rates = len(self._taus)
        alive = self._alive_index
        factors = self._factors
        pr = self._pseudo_root
        e = self._e
        ratios = self._ratios

        qassert.require(
            len(b_out) == number_rates,
            f"we need B.size() which is {len(b_out)} to equal numberRates which "
            f"is {number_rates}",
        )
        for j in b_out:
            qassert.require(
                j.shape[1] == factors and j.shape[0] == number_rates,
                "B[j] must be (numberRates, factors)",
            )

        for j in range(alive, number_rates):
            ratios[j] = (old_rates[j] + self._displacements[j]) * discount_ratios[j + 1]

        for f in range(factors):
            e[alive][f] = 0.0
            for j in range(alive + 1, number_rates):
                e[j][f] = e[j - 1][f] + ratios[j - 1] * pr[j - 1][f]

        # nullify B for rates that have already reset
        for j in range(alive):
            for k in range(number_rates):
                for f in range(factors):
                    b_out[j][k][f] = 0.0

        for f in range(factors):
            for j in range(alive, number_rates):
                for k in range(alive, j):
                    b_out[j][k][f] = (
                        new_rates[j] * ratios[k] * self._taus[k] * pr[j][f]
                    )
                tmp = 2.0 * ratios[j] * self._taus[j] * pr[j][f]
                tmp -= pr[j][f]
                tmp += e[j][f] * self._taus[j]
                tmp += gaussians[f]
                tmp *= new_rates[j] + self._displacements[j]
                b_out[j][j][f] = tmp
                for k in range(alive):
                    b_out[j][k][f] = 0.0
                for k in range(j + 1, number_rates):
                    b_out[j][k][f] = 0.0
