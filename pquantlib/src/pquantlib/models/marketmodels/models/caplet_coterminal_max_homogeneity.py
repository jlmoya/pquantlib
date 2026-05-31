"""CTSMMCapletMaxHomogeneityCalibration — max-homogeneity caplet calibration.

# C++ parity: ql/models/marketmodels/models/
# capletcoterminalmaxhomogeneity.{hpp,cpp} (v1.42.1).

The canonical coterminal-swap-market-model caplet calibration (the one used in
the QuantLib marketmodel.cpp test). For each forward rate it modifies the vol
of the next coterminal swap rate so the caplet variance matches the target
while keeping the time-dependent vol profile as homogeneous as possible. The
per-rate sub-problem is a sphere-cylinder closest-point search (sphere = total
swap variance, cylinder = caplet-variance constraint) rotated into an
orthonormal frame aligned with the cylinder centre + target via
``BasisIncompleteOrdered``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrixutilities.basis_incomplete_ordered import (
    BasisIncompleteOrdered,
)
from pquantlib.math.optimization.sphere_cylinder import SphereCylinderOptimizer
from pquantlib.math.quadratic import Quadratic
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


def _single_rate_closest_point_finder(  # noqa: PLR0915
    caplet_number: int,
    homogeneous_solution: list[float],
    previous_rate_solution: list[float],
    caplet_variance: float,
    correlations: list[float],
    w0: float,
    w1: float,
    caplet_swaption_priority: float,
    max_iterations: int,
    tolerance: float,
    solution: list[float],
    final_weight: float,
) -> tuple[bool, float, float]:
    """Solve the single-rate sub-problem; fill ``solution`` in place.

    Returns ``(success, swaption_error, caplet_error)``.

    # C++ parity: anonymous-namespace singleRateClosestPointFinder.
    """
    if caplet_number == 0:
        # only one point — going through everything would be silly
        previous_swap_variance = previous_rate_solution[0] * previous_rate_solution[0]
        this_swap_variance = (
            homogeneous_solution[0] * homogeneous_solution[0]
            + homogeneous_solution[1] * homogeneous_solution[1]
        )
        cross_term = 2 * w0 * w1 * correlations[0] * previous_rate_solution[0]
        constant_term = w0 * w0 * previous_swap_variance - caplet_variance
        theta = w1 * w1

        q = Quadratic(theta, cross_term, constant_term)
        volminus, _volplus, cap_success = q.roots()
        residual = this_swap_variance - volminus * volminus
        swap_success = residual >= 0
        success = cap_success and swap_success

        if success:
            solution[0] = volminus
            solution[1] = math.sqrt(residual)
            return success, 0.0, 0.0

        prioritize_caplet = caplet_swaption_priority < 0.5

        if cap_success and prioritize_caplet:
            solution[0] = volminus
            solution[1] = 0  # residual negative or we'd have totally succeeded
            swaption_error = math.sqrt(this_swap_variance) - volminus
            return success, swaption_error, 0.0

        if cap_success and not prioritize_caplet:
            solution[0] = math.sqrt(this_swap_variance)
            solution[1] = 0.0
            caplet_error = math.sqrt(q(solution[0]) + caplet_variance) - math.sqrt(
                caplet_variance
            )
            return success, 0.0, caplet_error

        # caplets have failed
        if swap_success:
            solution[0] = volminus
            solution[1] = math.sqrt(residual)
            caplet_error = math.sqrt(q(solution[0]) + caplet_variance) - math.sqrt(
                caplet_variance
            )
            return success, 0.0, caplet_error

        # caplets failed and swaps fail with optimal caplet solution
        if prioritize_caplet:
            solution[0] = volminus
            solution[1] = 0
            swaption_error = math.sqrt(this_swap_variance) - volminus
            return success, swaption_error, 0.0
        solution[0] = math.sqrt(this_swap_variance)
        solution[1] = 0.0
        caplet_error = math.sqrt(q(solution[0]) + caplet_variance) - math.sqrt(
            caplet_variance
        )
        return success, 0.0, caplet_error

    # --- general case (caplet_number >= 1) ---------------------------------
    previous_swap_variance = 0.0
    this_swap_variance = 0.0
    i = 0
    while i < caplet_number + 1:
        previous_swap_variance += previous_rate_solution[i] * previous_rate_solution[i]
        this_swap_variance += homogeneous_solution[i] * homogeneous_solution[i]
        i += 1
    this_swap_variance += homogeneous_solution[i] * homogeneous_solution[i]

    theta = w1 * w1
    b = [0.0] * (caplet_number + 1)
    cylinder_centre = np.zeros(caplet_number + 1, dtype=np.float64)
    target_array = np.zeros(caplet_number + 2, dtype=np.float64)
    target_array_restricted = np.zeros(caplet_number + 1, dtype=np.float64)

    bsq = 0.0
    for i in range(caplet_number + 1):
        b[i] = 2 * w0 * w1 * correlations[i] * previous_rate_solution[i] / theta
        cylinder_centre[i] = -0.5 * b[i]
        target_array[i] = homogeneous_solution[i]
        target_array_restricted[i] = target_array[i]
        bsq += b[i] * b[i]
    target_array[caplet_number + 1] = homogeneous_solution[caplet_number + 1]

    a_const = previous_swap_variance * w0 * w0 / theta
    const_quadratic_term = a_const - 0.25 * bsq
    s2 = caplet_variance / theta - const_quadratic_term

    # if s2 < 0 there are no solutions so take the best we can
    s = math.sqrt(s2) if s2 > 0 else 0.0
    r = math.sqrt(this_swap_variance)

    basis = BasisIncompleteOrdered(caplet_number + 1)
    basis.add_vector(cylinder_centre)
    basis.add_vector(target_array_restricted)
    for i in range(caplet_number + 1):
        ei = np.zeros(caplet_number + 1, dtype=np.float64)
        ei[i] = 1.0
        basis.add_vector(ei)

    orth_transformation_restricted = basis.get_basis_as_rows_in_matrix()
    orth_transformation = np.zeros(
        (caplet_number + 2, caplet_number + 2), dtype=np.float64
    )
    orth_transformation[caplet_number + 1][caplet_number + 1] = 1.0
    for k in range(caplet_number + 1):
        for ll in range(caplet_number + 1):
            orth_transformation[k][ll] = orth_transformation_restricted[k][ll]

    moved_centre = orth_transformation_restricted @ cylinder_centre
    alpha = moved_centre[0]
    moved_target = orth_transformation @ target_array

    z1 = 0.0
    z2 = 0.0
    z3 = 0.0

    optimizer = SphereCylinderOptimizer(
        r,
        s,
        alpha,
        moved_target[0],
        moved_target[1],
        moved_target[len(moved_target) - 1],
        final_weight,
    )

    success = False
    swaption_error = 0.0
    caplet_error = 0.0

    if not optimizer.is_intersection_non_empty():
        z1 = r * caplet_swaption_priority + (1 - caplet_swaption_priority) * (alpha - s)
        z2 = 0.0
        z3 = 0.0
        swaption_error = z1 - r
        caplet_error = (alpha - s) - z1
    else:
        success = True
        caplet_error = 0.0
        swaption_error = 0.0
        if max_iterations > 0.0:
            z1, z2, z3 = optimizer.find_closest(max_iterations, tolerance)
        else:
            z1, z2, z3 = optimizer.find_by_projection()

    rotated_solution = np.zeros(caplet_number + 2, dtype=np.float64)
    rotated_solution[0] = z1
    rotated_solution[1] = z2
    rotated_solution[caplet_number + 1] = z3

    array_solution = orth_transformation.T @ rotated_solution
    i = 0
    while i < array_solution.shape[0]:
        solution[i] = array_solution[i]
        i += 1
    while i < len(solution):
        solution[i] = 0.0
        i += 1

    return success, swaption_error, caplet_error


class CTSMMCapletMaxHomogeneityCalibration(CTSMMCapletCalibration):
    """Max-homogeneity caplet-coterminal calibration.

    # C++ parity: CTSMMCapletMaxHomogeneityCalibration.
    """

    def __init__(
        self,
        evolution: EvolutionDescription,
        corr: PiecewiseConstantCorrelation,
        displaced_swap_variances: list[PiecewiseConstantVariance],
        caplet_vols: list[float],
        cs: CurveState,
        displacement: float,
        caplet0_swaption1_priority: float = 1.0,
    ) -> None:
        # C++ parity: CTSMMCapletMaxHomogeneityCalibration ctor.
        super().__init__(
            evolution, corr, displaced_swap_variances, caplet_vols, cs, displacement
        )
        qassert.require(
            0.0 <= caplet0_swaption1_priority <= 1.0,
            f"caplet0Swaption1Priority ({caplet0_swaption1_priority}) must be in "
            f"[0.0, 1.0]",
        )
        self._caplet0_swaption1_priority = caplet0_swaption1_priority
        self._total_swaption_error = 0.0

    @staticmethod
    def caplet_max_homogeneity_calibration(
        evolution: EvolutionDescription,
        corr: PiecewiseConstantCorrelation,
        displaced_swap_variances: list[PiecewiseConstantVariance],
        caplet_vols: list[float],
        cs: CurveState,
        displacement: float,
        caplet0_swaption1_priority: float,
        number_of_factors: int,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[int, float, float, list[Matrix]]:
        """The static calibration function.

        Returns ``(failures, deformation_size, total_swaption_error,
        swap_covariance_pseudo_roots)``.

        # C++ parity:
        CTSMMCapletMaxHomogeneityCalibration::capletMaxHomogeneityCalibration.
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
        total_swaption_error = 0.0
        deformation_size = 0.0

        # factor reduction
        corr_pseudo = [
            rank_reduced_sqrt(
                corr.correlation(i), number_of_factors, 1.0, SalvagingAlgorithm.NONE
            )
            for i in range(len(corr.times()))
        ]

        # Zinverse; wj later
        zed_matrix = SwapForwardMappings.coterminal_swap_zed_matrix(cs, displacement)
        inverted_zed_matrix = np.linalg.inv(np.asarray(zed_matrix, dtype=np.float64))

        new_vols: list[list[float]] = []
        these_new_vols = [0.0] * number_of_rates
        first_rate_vols = [0.0] * number_of_rates
        first_rate_vols[0] = math.sqrt(displaced_swap_variances[0].variances()[0])
        second_rate_vols = [0.0] * number_of_rates
        correlations = [0.0] * number_of_rates
        new_vols.append(list(first_rate_vols))

        # final caplet and swaption are the same, so we skip that case
        for i in range(number_of_rates - 1):
            # final weight does nothing when i < 2
            this_final_weight = (i - 1) / 2.0 if i > 1 else 1.0
            # calibrate caplet on forward rate i by modifying swap rate i+1's vol
            var = displaced_swap_variances[i + 1].variances()
            for j in range(i + 2):
                second_rate_vols[j] = math.sqrt(var[j])

            for k in range(i + 1):
                correlation = 0.0
                for ll in range(number_of_factors):
                    term1 = corr_pseudo[k][i][ll]
                    term2 = corr_pseudo[k][i + 1][ll]
                    correlation += term1 * term2
                correlations[k] = correlation

            w0 = inverted_zed_matrix[i][i]
            w1 = inverted_zed_matrix[i][i + 1]
            # w0 adjustment
            for k in range(i + 2, inverted_zed_matrix.shape[1]):
                w0 += inverted_zed_matrix[i][k]

            target_caplet_variance = (
                caplet_vols[i] * caplet_vols[i] * rate_times[i]
            )

            success, this_swaption_error, _this_caplet_error = (
                _single_rate_closest_point_finder(
                    i,
                    second_rate_vols,
                    first_rate_vols,
                    target_caplet_variance,
                    correlations,
                    w0,
                    w1,
                    caplet0_swaption1_priority,
                    max_iterations,
                    tolerance,
                    these_new_vols,
                    this_final_weight,
                )
            )

            total_swaption_error += this_swaption_error * this_swaption_error

            if not success:
                failures += 1

            for _j in range(i + 2):
                deformation_size += (these_new_vols[i] - second_rate_vols[i]) * (
                    these_new_vols[i] - second_rate_vols[i]
                )

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

        return failures, deformation_size, total_swaption_error, (
            swap_covariance_pseudo_roots
        )

    def calibration_impl(
        self,
        number_of_factors: int,
        inner_max_iterations: int,
        inner_tolerance: float,
    ) -> int:
        # C++ parity: CTSMMCapletMaxHomogeneityCalibration::calibrationImpl_.
        (
            failures,
            deformation_size,
            total_swaption_error,
            roots,
        ) = self.caplet_max_homogeneity_calibration(
            self.evolution,
            self._corr,
            self._displaced_swap_variances,
            self._used_caplet_vols,  # not mktCapletVols_ but...
            self._cs,
            self._displacement,
            self._caplet0_swaption1_priority,
            number_of_factors,
            inner_max_iterations,
            inner_tolerance,
        )
        self._deformation_size = deformation_size
        self._total_swaption_error = total_swaption_error
        self._swap_covariance_pseudo_roots = roots
        return failures
