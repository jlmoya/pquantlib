"""CTSMMCapletOriginalCalibration — Joshi-original joint caplet+swaption calib.

# C++ parity: ql/models/marketmodels/models/
# capletcoterminalswaptioncalibration.{hpp,cpp} (v1.42.1).

The "original" coterminal-swap-market-model caplet calibration (Joshi 2007).
For each forward rate it modifies the vol of the next coterminal swap rate by
two multipliers ``a`` (up to the previous reset) and ``b`` (afterward), solving
a quadratic in ``a`` so that the implied caplet variance matches the target;
the residual variance goes into ``b`` to preserve the total swaption variance.
An optional ``alpha`` per-rate inhomogeneity-form pre-warps the time-bucketed
variances (then rescales so the total is unchanged).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.models.marketmodels.models.ctsmm_caplet_calibration import (
    CTSMMCapletCalibration,
)
from pquantlib.models.marketmodels.models.pseudo_sqrt import (
    SalvagingAlgorithm,
    rank_reduced_sqrt,
)
from pquantlib.models.marketmodels.swap_forward_mappings import SwapForwardMappings

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
    from pquantlib.models.marketmodels.models.piecewise_constant_variance import (
        PiecewiseConstantVariance,
    )
    from pquantlib.models.marketmodels.piecewise_constant_correlation import (
        PiecewiseConstantCorrelation,
    )


class CTSMMCapletOriginalCalibration(CTSMMCapletCalibration):
    """Joshi-original caplet-coterminal calibration.

    # C++ parity: CTSMMCapletOriginalCalibration.
    """

    def __init__(
        self,
        evolution: EvolutionDescription,
        corr: PiecewiseConstantCorrelation,
        displaced_swap_variances: list[PiecewiseConstantVariance],
        mkt_caplet_vols: list[float],
        cs: CurveState,
        displacement: float,
        alpha: list[float],
        lowest_root: bool,
        use_full_approx: bool,
    ) -> None:
        # C++ parity: CTSMMCapletOriginalCalibration ctor.
        super().__init__(
            evolution, corr, displaced_swap_variances, mkt_caplet_vols, cs,
            displacement,
        )
        qassert.require(
            self.number_of_rates == len(alpha),
            f"mismatch between number of rates ({self.number_of_rates}) and "
            f"alpha ({len(alpha)})",
        )
        self._alpha = list(alpha)
        self._lowest_root = lowest_root
        self._use_full_approx = use_full_approx

    @staticmethod
    def calibration_function(  # noqa: PLR0915
        evolution: EvolutionDescription,
        corr: PiecewiseConstantCorrelation,
        displaced_swap_variances: list[PiecewiseConstantVariance],
        caplet_vols: list[float],
        cs: CurveState,
        displacement: float,
        alpha: list[float],
        lowest_root: bool,
        use_full_approx: bool,
        number_of_factors: int,
    ) -> tuple[int, list[Matrix]]:
        """The static calibration function.

        Returns ``(failures, swap_covariance_pseudo_roots)``.

        # C++ parity: CTSMMCapletOriginalCalibration::calibrationFunction.
        """
        CTSMMCapletCalibration.perform_checks(
            evolution, corr, displaced_swap_variances, caplet_vols, cs
        )
        number_of_steps = evolution.number_of_steps()
        number_of_rates = evolution.number_of_rates()
        rate_times = evolution.rate_times()

        qassert.require(
            number_of_factors <= number_of_rates,
            f"number of factors ({number_of_factors}) cannot be greater than "
            f"numberOfRates ({number_of_rates})",
        )
        qassert.require(
            number_of_factors > 0,
            f"number of factors ({number_of_factors}) must be greater than zero",
        )

        failures = 0
        extra_multiplier = 1.0 if use_full_approx else 0.0

        # factor reduction
        corr_pseudo = [
            rank_reduced_sqrt(
                corr.correlation(i), number_of_factors, 1.0, SalvagingAlgorithm.NONE
            )
            for i in range(len(corr.times()))
        ]

        zed_matrix = SwapForwardMappings.coterminal_swap_zed_matrix(cs, displacement)
        inverted_zed_matrix = np.linalg.inv(
            np.asarray(zed_matrix, dtype=np.float64)
        )

        # --- alpha part: warp time-bucketed variances, rescale to keep total --
        swap_time_inhomogeneous_variances = np.zeros(
            (number_of_steps, number_of_rates), dtype=np.float64
        )
        original_variances = [0.0] * number_of_rates
        modified_variances = [0.0] * number_of_rates
        evolution_times = evolution.evolution_times()
        for i in range(number_of_steps):
            s = 0.0 if i == 0 else evolution_times[i - 1]
            for j in range(i, number_of_rates):
                var = displaced_swap_variances[j].variances()
                original_variances[j] += var[i]
                swap_time_inhomogeneous_variances[i][j] = var[i] / (
                    (1.0 + alpha[j] * s) * (1.0 + alpha[j] * s)
                )
                modified_variances[j] += swap_time_inhomogeneous_variances[i][j]

        for i in range(number_of_steps):
            for j in range(i, number_of_rates):
                swap_time_inhomogeneous_variances[i][j] *= (
                    original_variances[j] / modified_variances[j]
                )

        # --- swap covariances for the caplet approximation (without A, B) -----
        covariance_swap_pseudos: list[Matrix] = []
        covariance_swap_covs: list[Matrix] = []  # total cov
        covariance_swap_marginal_covs: list[Matrix] = []  # per-step cov
        for i in range(number_of_steps):
            csp = np.array(corr_pseudo[i], dtype=np.float64, copy=True)
            for j in range(number_of_rates):
                for k in range(csp.shape[1]):
                    csp[j][k] *= math.sqrt(swap_time_inhomogeneous_variances[i][j])
            covariance_swap_pseudos.append(csp)
            marginal = csp @ csp.T
            covariance_swap_marginal_covs.append(marginal)
            cov = marginal.copy()
            if i > 0:
                cov = cov + covariance_swap_covs[i - 1]
            covariance_swap_covs.append(cov)

        # --- partial variances / covariances that take A, B coefficients ------
        tot_variance = [0.0] * number_of_rates
        almost_tot_variance = [0.0] * number_of_rates
        almost_tot_covariance = [0.0] * number_of_rates
        left_covariance = [0.0] * number_of_rates
        for i in range(number_of_rates):
            for jj in range(i + 1):
                tot_variance[i] += displaced_swap_variances[i].variances()[jj]
            for j in range(i):  # j: 0..i-1
                almost_tot_variance[i] += swap_time_inhomogeneous_variances[j][i]
            for j in range(i - 1):  # j: 0..i-2
                this_pseudo = corr_pseudo[j]
                correlation = 0.0
                for k in range(number_of_factors):
                    correlation += this_pseudo[i - 1][k] * this_pseudo[i][k]
                almost_tot_covariance[i] += correlation * math.sqrt(
                    swap_time_inhomogeneous_variances[j][i]
                    * swap_time_inhomogeneous_variances[j][i - 1]
                )
            if i > 0:
                j = i - 1
                this_pseudo = corr_pseudo[j]
                correlation = 0.0
                for k in range(number_of_factors):
                    correlation += this_pseudo[i - 1][k] * this_pseudo[i][k]
                left_covariance[i] = correlation * math.sqrt(
                    swap_time_inhomogeneous_variances[j][i]
                    * swap_time_inhomogeneous_variances[j][i - 1]
                )

        # multiplier up to previous reset (a[0] unused) and afterward (b)
        a = [1.0] * number_of_steps
        b = [0.0] * number_of_steps
        b[0] = (
            displaced_swap_variances[0].variances()[0]
            / swap_time_inhomogeneous_variances[0][0]
        )

        # main loop
        for i in range(1, number_of_steps):
            # update variances for last a, b computed
            j = 0
            while j <= i - 2:
                swap_time_inhomogeneous_variances[j][i - 1] *= a[i - 1] * a[i - 1]
                j += 1
            swap_time_inhomogeneous_variances[j][i - 1] *= b[i - 1] * b[i - 1]

            w0 = inverted_zed_matrix[i - 1][i - 1]
            w1 = -inverted_zed_matrix[i - 1][i]
            v1t1 = caplet_vols[i - 1] * caplet_vols[i - 1] * rate_times[i - 1]

            # contribution from lower-right corner
            extra_constant_part = 0.0
            for k in range(i + 1, number_of_steps):
                for ll in range(i + 1, number_of_steps):
                    extra_constant_part += (
                        inverted_zed_matrix[i - 1][k]
                        * covariance_swap_covs[i - 1][k][ll]
                        * inverted_zed_matrix[i - 1][ll]
                    )

            # top row (excluding first two columns) and its transpose
            for k in range(i + 1, number_of_steps):
                if i > 1:
                    extra_constant_part += (
                        inverted_zed_matrix[i - 1][i - 1]
                        * covariance_swap_covs[i - 2][i - 1][k]
                        * inverted_zed_matrix[i - 1][k]
                        * a[i - 1]
                    )
                    extra_constant_part += (
                        inverted_zed_matrix[i - 1][k]
                        * covariance_swap_covs[i - 2][k][i - 1]
                        * inverted_zed_matrix[i - 1][i - 1]
                        * a[i - 1]
                    )
                extra_constant_part += (
                    inverted_zed_matrix[i - 1][i - 1]
                    * covariance_swap_marginal_covs[i - 1][i - 1][k]
                    * inverted_zed_matrix[i - 1][k]
                    * b[i - 1]
                )
                extra_constant_part += (
                    inverted_zed_matrix[i - 1][k]
                    * covariance_swap_covs[i - 1][k][i - 1]
                    * inverted_zed_matrix[i - 1][i - 1]
                    * b[i - 1]
                )

            # extra linear part: row i, columns j > i+1, and transpose
            extra_linear_part = 0.0
            for k in range(i + 1, number_of_steps):
                extra_linear_part += (
                    inverted_zed_matrix[i - 1][k]
                    * covariance_swap_covs[i - 1][k][i]
                    * inverted_zed_matrix[i - 1][i]
                )
                extra_linear_part += (
                    inverted_zed_matrix[i - 1][i]
                    * covariance_swap_covs[i - 1][i][k]
                    * inverted_zed_matrix[i - 1][k]
                )

            constant_part = (
                w0 * w0 * tot_variance[i - 1]
                + extra_constant_part * extra_multiplier
                - v1t1
            )
            linear_part = (
                -2 * w0 * w1
                * (a[i - 1] * almost_tot_covariance[i] + b[i - 1] * left_covariance[i])
                + extra_linear_part * extra_multiplier
            )
            quadratic_part = w1 * w1 * almost_tot_variance[i]
            disc = linear_part * linear_part - 4.0 * constant_part * quadratic_part

            minimum = -linear_part / (2.0 * quadratic_part)
            right_used = False
            if disc < 0.0:
                failures += 1
                root = minimum
            elif lowest_root or minimum > 1.0:
                root = (-linear_part - math.sqrt(disc)) / (2.0 * quadratic_part)
            else:
                right_used = True
                root = (-linear_part + math.sqrt(disc)) / (2.0 * quadratic_part)

            variance_found = root * root * almost_tot_variance[i]
            variance_to_find = tot_variance[i] - variance_found
            mult = variance_to_find / swap_time_inhomogeneous_variances[i][i]
            if mult <= 0.0 and right_used:
                root = (-linear_part - math.sqrt(disc)) / (2.0 * quadratic_part)
                variance_found = root * root * almost_tot_variance[i]
                variance_to_find = tot_variance[i] - variance_found
                mult = variance_to_find / swap_time_inhomogeneous_variances[i][i]

            if mult < 0.0:  # no solution
                failures += 1
                a[i] = root
                b[i] = 0.0
            else:
                a[i] = root
                b[i] = math.sqrt(mult)

            qassert.require(
                root >= 0.0, "negative root -- it should have not happened"
            )

        # final variance update for the last step
        i = number_of_steps
        j = 0
        while j <= i - 2:
            swap_time_inhomogeneous_variances[j][i - 1] *= a[i - 1] * a[i - 1]
            j += 1
        swap_time_inhomogeneous_variances[j][i - 1] *= b[i - 1] * b[i - 1]

        # results
        swap_covariance_pseudo_roots: list[Matrix] = []
        for k in range(number_of_steps):
            scpr = np.array(corr_pseudo[k], dtype=np.float64, copy=True)
            for j in range(number_of_rates):
                coeff = math.sqrt(swap_time_inhomogeneous_variances[k][j])
                for ii in range(number_of_factors):
                    scpr[j][ii] *= coeff
            qassert.require(
                scpr.shape[0] == number_of_rates,
                f"step {k} abcd vol wrong number of rows: {scpr.shape[0]} "
                f"instead of {number_of_rates}",
            )
            qassert.require(
                scpr.shape[1] == number_of_factors,
                f"step {k} abcd vol wrong number of columns: {scpr.shape[1]} "
                f"instead of {number_of_factors}",
            )
            swap_covariance_pseudo_roots.append(scpr)

        return failures, swap_covariance_pseudo_roots

    def calibration_impl(
        self,
        number_of_factors: int,
        inner_max_iterations: int,  # C++ parity: unused in original
        inner_tolerance: float,  # C++ parity: unused in original
    ) -> int:
        # C++ parity: CTSMMCapletOriginalCalibration::calibrationImpl_.
        failures, roots = self.calibration_function(
            self.evolution,
            self._corr,
            self._displaced_swap_variances,
            self._used_caplet_vols,
            self._cs,
            self._displacement,
            self._alpha,
            self._lowest_root,
            self._use_full_approx,
            number_of_factors,
        )
        self._swap_covariance_pseudo_roots = roots
        return failures
