"""capletSwaptionPeriodicCalibration — periodic max-homogeneity calibration.

# C++ parity: ql/models/marketmodels/models/capletcoterminalperiodic.{hpp,cpp}
# (v1.42.1).

A wrapper around :class:`CTSMMCapletMaxHomogeneityCalibration` that calibrates a
*periodic* (longer-tenor) coterminal swap market model. The small (caplet-tenor)
swap variances are produced by interpolating the supplied big-rate variances via
a :class:`VolatilityInterpolationSpecifier`; an outer loop rescales each big-rate
vol so the periodic swaption vols match the market, re-running the inner
max-homogeneity caplet calibration each iteration until the periodic swaption
RMS error stops improving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.models.marketmodels.models.caplet_coterminal_max_homogeneity import (
    CTSMMCapletMaxHomogeneityCalibration,
)
from pquantlib.models.marketmodels.models.cot_swap_to_fwd_adapter import (
    CotSwapToFwdAdapter,
)
from pquantlib.models.marketmodels.models.fwd_period_adapter import FwdPeriodAdapter
from pquantlib.models.marketmodels.models.fwd_to_cot_swap_adapter import (
    FwdToCotSwapAdapter,
)
from pquantlib.models.marketmodels.models.pseudo_root_facade import PseudoRootFacade

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
    from pquantlib.models.marketmodels.models.volatility_interpolation_specifier import (
        VolatilityInterpolationSpecifier,
    )
    from pquantlib.models.marketmodels.piecewise_constant_correlation import (
        PiecewiseConstantCorrelation,
    )


@dataclass
class PeriodicCalibrationResult:
    """The bundle of out-parameters that C++ writes through references.

    # C++ parity: the trailing ``Real&`` / vector / matrix out-params of
    capletSwaptionPeriodicCalibration.
    """

    failures: int
    deformation_size: float  # not set yet in C++ either
    total_swaption_error: float
    swap_covariance_pseudo_roots: list[Matrix]
    final_scales: list[float]
    iterations_done: int
    error_improvement: float
    model_swaption_vols_matrix: Matrix = field(default_factory=lambda: np.zeros((0, 0)))


def caplet_swaption_periodic_calibration(
    evolution: EvolutionDescription,
    corr: PiecewiseConstantCorrelation,
    displaced_swap_variances: VolatilityInterpolationSpecifier,
    caplet_vols: list[float],
    cs: CurveState,
    displacement: float,
    caplet0_swaption1_priority: float,
    number_of_factors: int,
    period: int,
    max1d_iterations: int,
    tolerance1d: float,
    max_unperiodic_iterations: int,
    tolerance_unperiodic: float,
    max_period_iterations: int,
    period_tolerance: float,
) -> PeriodicCalibrationResult:
    """Periodic caplet/coterminal-swaption calibration.

    # C++ parity: capletSwaptionPeriodicCalibration (free function).
    """
    number_small_rates = evolution.number_of_rates()
    number_small_steps = evolution.number_of_steps()
    qassert.require(
        number_small_steps == number_small_rates,
        "periodic calibration class requires evolution to the reset of each rate",
    )

    number_big_rates = number_small_rates // period
    offset = number_small_rates % period

    new_displacements = [displacement] * number_big_rates

    qassert.require(
        displaced_swap_variances.get_no_big_rates() == number_big_rates,
        "mismatch between number of swap variances given and number of rates "
        "and period",
    )

    failures = 0
    scaling_factors = [1.0] * number_big_rates

    displaced_swap_variances.set_last_caplet_vol(caplet_vols[-1])

    market_swaption_vols = [
        displaced_swap_variances.original_variances()[i].total_volatility(i)
        for i in range(number_big_rates)
    ]
    model_swaption_vols = [0.0] * number_big_rates

    iterations_done = 0
    previous_error = 1.0e10  # very large number
    model_swaption_vols_matrix = np.zeros(
        (max_period_iterations, number_big_rates), dtype=np.float64
    )

    period_swaption_rms_error = 0.0
    error_improvement = 0.0
    swap_covariance_pseudo_roots: list[Matrix] = []

    while True:
        displaced_swap_variances.set_scaling_factors(scaling_factors)

        unperiodic_calibrator = CTSMMCapletMaxHomogeneityCalibration(
            evolution,
            corr,
            displaced_swap_variances.interpolated_variances(),
            caplet_vols,
            cs,
            displacement,
            caplet0_swaption1_priority,
        )

        failures = int(
            unperiodic_calibrator.calibrate(
                number_of_factors,
                max_unperiodic_iterations,
                tolerance_unperiodic,
                max1d_iterations,
                tolerance1d,
            )
        )

        swap_covariance_pseudo_roots = unperiodic_calibrator.swap_pseudo_roots()

        smm = PseudoRootFacade(
            swap_covariance_pseudo_roots,
            evolution.rate_times(),
            cs.coterminal_swap_rates(),
            [displacement] * evolution.number_of_rates(),
        )
        flmm = CotSwapToFwdAdapter(smm)
        _caplet_tot_covariance = flmm.total_covariance(number_small_rates - 1)

        periodflmm = FwdPeriodAdapter(flmm, period, offset, new_displacements)
        periodsmm = FwdToCotSwapAdapter(periodflmm)

        swaption_tot_covariance = periodsmm.total_covariance(
            periodsmm.number_of_steps() - 1
        )

        total_swaption_error = 0.0
        for i in range(number_big_rates):
            model_swaption_vols[i] = math.sqrt(
                swaption_tot_covariance[i, i]
                / periodsmm.evolution().rate_times()[i]
            )
            scale = market_swaption_vols[i] / model_swaption_vols[i]
            scaling_factors[i] *= scale  # since applied to vol
            total_swaption_error += (
                market_swaption_vols[i] - model_swaption_vols[i]
            ) * (market_swaption_vols[i] - model_swaption_vols[i])

        for i in range(number_big_rates):
            model_swaption_vols_matrix[iterations_done][i] = model_swaption_vols[i]

        period_swaption_rms_error = math.sqrt(
            total_swaption_error / number_big_rates
        )
        error_improvement = previous_error - period_swaption_rms_error
        previous_error = period_swaption_rms_error

        # C++ parity: while (errorImprovement > periodTolerance/10.0 &&
        #   periodSwaptionRmsError > periodTolerance &&
        #   ++iterationsDone < maxPeriodIterations);
        # ``++iterationsDone`` is only evaluated (and the counter only
        # incremented) when the first two conditions hold — short-circuit.
        if not (
            error_improvement > period_tolerance / 10.0
            and period_swaption_rms_error > period_tolerance
        ):
            break
        iterations_done += 1
        if not iterations_done < max_period_iterations:
            break

    return PeriodicCalibrationResult(
        failures=failures,
        deformation_size=0.0,
        total_swaption_error=total_swaption_error,
        swap_covariance_pseudo_roots=swap_covariance_pseudo_roots,
        final_scales=list(scaling_factors),
        iterations_done=iterations_done,
        error_improvement=error_improvement,
        model_swaption_vols_matrix=model_swaption_vols_matrix,
    )
