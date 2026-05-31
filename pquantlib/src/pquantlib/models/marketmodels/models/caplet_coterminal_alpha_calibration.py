"""CTSMMCapletAlphaFormCalibration — alpha-form caplet calibration.

# C++ parity: ql/models/marketmodels/models/
# capletcoterminalalphacalibration.{hpp,cpp} (v1.42.1).

A coterminal-swap-market-model caplet calibration that, for each forward rate,
runs an :class:`AlphaFinder` to choose the alpha-form parameter (and the
consequent ``a, b`` multipliers) so the modified swap-rate vols reprice the
target caplet variance, optionally maximising time-homogeneity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.models.marketmodels.models.alpha_finder import AlphaFinder
from pquantlib.models.marketmodels.models.alpha_form_concrete import (
    AlphaFormLinearHyperbolic,
)
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
    from pquantlib.models.marketmodels.models.alpha_form import AlphaForm
    from pquantlib.models.marketmodels.models.piecewise_constant_variance import (
        PiecewiseConstantVariance,
    )
    from pquantlib.models.marketmodels.piecewise_constant_correlation import (
        PiecewiseConstantCorrelation,
    )


class CTSMMCapletAlphaFormCalibration(CTSMMCapletCalibration):
    """Alpha-form caplet-coterminal calibration.

    # C++ parity: CTSMMCapletAlphaFormCalibration.
    """

    def __init__(
        self,
        evolution: EvolutionDescription,
        corr: PiecewiseConstantCorrelation,
        displaced_swap_variances: list[PiecewiseConstantVariance],
        caplet_vols: list[float],
        cs: CurveState,
        displacement: float,
        alpha_initial: list[float],
        alpha_max: list[float],
        alpha_min: list[float],
        maximize_homogeneity: bool,
        parametric_form: AlphaForm | None = None,
    ) -> None:
        # C++ parity: CTSMMCapletAlphaFormCalibration ctor.
        super().__init__(
            evolution, corr, displaced_swap_variances, caplet_vols, cs, displacement
        )
        if parametric_form is None:
            parametric_form = AlphaFormLinearHyperbolic(evolution.rate_times())
        self._parametric_form = parametric_form

        n = self.number_of_rates
        qassert.require(
            n == len(alpha_initial),
            f"mismatch between number of rates ({n}) and alphaInitial "
            f"({len(alpha_initial)})",
        )
        qassert.require(
            n == len(alpha_max),
            f"mismatch between number of rates ({n}) and alphaMax "
            f"({len(alpha_max)})",
        )
        qassert.require(
            n == len(alpha_min),
            f"mismatch between number of rates ({n}) and alphaMin "
            f"({len(alpha_min)})",
        )
        self._alpha_initial = list(alpha_initial)
        self._alpha_max = list(alpha_max)
        self._alpha_min = list(alpha_min)
        self._maximize_homogeneity = maximize_homogeneity
        self._alpha = [0.0] * n
        self._a = [0.0] * n
        self._b = [0.0] * n

    def alpha(self) -> list[float]:
        # C++ parity: CTSMMCapletAlphaFormCalibration::alpha.
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._alpha

    @staticmethod
    def caplet_alpha_form_calibration(  # noqa: PLR0915
        evolution: EvolutionDescription,
        corr: PiecewiseConstantCorrelation,
        displaced_swap_variances: list[PiecewiseConstantVariance],
        caplet_vols: list[float],
        cs: CurveState,
        displacement: float,
        alpha_initial: list[float],
        alpha_max: list[float],
        alpha_min: list[float],
        maximize_homogeneity: bool,
        parametric_form: AlphaForm,
        number_of_factors: int,
        steps: int,
        tolerance: float,
    ) -> tuple[int, list[float], list[float], list[float], list[Matrix]]:
        """The static calibration function.

        Returns ``(failures, alpha, a, b, swap_covariance_pseudo_roots)``.

        # C++ parity:
        CTSMMCapletAlphaFormCalibration::capletAlphaFormCalibration.
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
        alpha = [0.0] * number_of_rates
        a = [0.0] * number_of_rates
        b = [0.0] * number_of_rates

        corr_pseudo = [
            rank_reduced_sqrt(
                corr.correlation(i), number_of_factors, 1.0, SalvagingAlgorithm.NONE
            )
            for i in range(len(corr.times()))
        ]

        zed_matrix = SwapForwardMappings.coterminal_swap_zed_matrix(cs, displacement)
        inverted_zed_matrix = np.linalg.inv(np.asarray(zed_matrix, dtype=np.float64))

        new_vols: list[list[float]] = []
        these_new_vols = [0.0] * number_of_rates
        first_rate_vols = [0.0] * number_of_rates
        first_rate_vols[0] = np.sqrt(displaced_swap_variances[0].variances()[0])
        second_rate_vols = [0.0] * number_of_rates
        correlations = [0.0] * number_of_rates
        new_vols.append(list(first_rate_vols))

        alpha[0] = alpha_initial[0]  # has no effect on anything in any case
        a[0] = b[0] = 1.0  # no modifications to swap vol for first rate

        solver = AlphaFinder(parametric_form)

        # final caplet and swaption are the same, so we skip that case
        for i in range(number_of_rates - 1):
            var = displaced_swap_variances[i + 1].variances()
            for j in range(i + 2):
                second_rate_vols[j] = np.sqrt(var[j])

            for k in range(i + 1):
                correlation = 0.0
                for ll in range(number_of_factors):
                    term1 = corr_pseudo[k][i][ll]
                    term2 = corr_pseudo[k][i + 1][ll]
                    correlation += term1 * term2
                correlations[k] = correlation

            w0 = inverted_zed_matrix[i][i]
            w1 = inverted_zed_matrix[i][i + 1]
            for k in range(i + 2, inverted_zed_matrix.shape[1]):
                w0 += inverted_zed_matrix[i][k]

            target_variance = caplet_vols[i] * caplet_vols[i] * rate_times[i]

            if maximize_homogeneity:
                success, this_alpha, this_a, this_b = solver.solve_with_max_homogeneity(
                    alpha_initial[i + 1],
                    i,
                    first_rate_vols,
                    second_rate_vols,
                    correlations,
                    w0,
                    w1,
                    target_variance,
                    tolerance,
                    alpha_max[i + 1],
                    alpha_min[i + 1],
                    steps,
                    these_new_vols,
                )
            else:
                success, this_alpha, this_a, this_b = solver.solve(
                    alpha_initial[i + 1],
                    i,
                    first_rate_vols,
                    second_rate_vols,
                    correlations,
                    w0,
                    w1,
                    target_variance,
                    tolerance,
                    alpha_max[i + 1],
                    alpha_min[i + 1],
                    steps,
                    these_new_vols,
                )
            alpha[i + 1] = this_alpha
            a[i + 1] = this_a
            b[i + 1] = this_b

            if not success:
                qassert.fail("alpha form failure")

            new_vols.append(list(these_new_vols))
            first_rate_vols = list(these_new_vols)

        swap_covariance_pseudo_roots: list[Matrix] = []
        for k in range(number_of_steps):
            scpr = np.array(corr_pseudo[k], dtype=np.float64, copy=True)
            for j in range(number_of_rates):
                coeff = new_vols[j][k]
                for i in range(number_of_factors):
                    scpr[j][i] *= coeff
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

        return failures, alpha, a, b, swap_covariance_pseudo_roots

    def calibration_impl(
        self,
        number_of_factors: int,
        inner_max_iterations: int,
        inner_tolerance: float,
    ) -> int:
        # C++ parity: CTSMMCapletAlphaFormCalibration::calibrationImpl_.
        failures, alpha, a, b, roots = self.caplet_alpha_form_calibration(
            self.evolution,
            self._corr,
            self._displaced_swap_variances,
            self._used_caplet_vols,  # not mktCapletVols_ but...
            self._cs,
            self._displacement,
            self._alpha_initial,
            self._alpha_max,
            self._alpha_min,
            self._maximize_homogeneity,
            self._parametric_form,
            number_of_factors,
            inner_max_iterations,
            inner_tolerance,
        )
        self._alpha = alpha
        self._a = a
        self._b = b
        self._swap_covariance_pseudo_roots = roots
        return failures
