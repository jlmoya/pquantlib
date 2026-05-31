"""CTSMMCapletCalibration — coterminal-swap-market-model caplet calibration base.

# C++ parity: ql/models/marketmodels/models/ctsmmcapletcalibration.{hpp,cpp}
# (v1.42.1).

The abstract base of the caplet-coterminal calibration family. Concrete
subclasses implement :meth:`calibration_impl` (the single-pass calibration that
fills ``swap_covariance_pseudo_roots``); the base :meth:`calibrate` wraps it in
the outer iteration that rescales the used caplet vols until the implied caplet
vols match the market caplet vols (while the swaption fit stays perfect by
construction).

``PseudoRootFacade.from_calibrator`` reads ``swap_pseudo_roots()`` +
``curve_state()`` + ``displacements()`` off this class.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.models.marketmodels.models.cot_swap_to_fwd_adapter import (
    CotSwapToFwdAdapter,
)
from pquantlib.models.marketmodels.models.pseudo_root_facade import PseudoRootFacade

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

# C++ parity: a positive large number used as the un-calibrated sentinel
# (ctsmmcapletcalibration.cpp calibrate()).
_LARGE = 987654321.0


class CTSMMCapletCalibration(ABC):
    """Abstract caplet-coterminal calibration.

    # C++ parity: CTSMMCapletCalibration.
    """

    def __init__(
        self,
        evolution: EvolutionDescription,
        corr: PiecewiseConstantCorrelation,
        displaced_swap_variances: list[PiecewiseConstantVariance],
        mkt_caplet_vols: list[float],
        cs: CurveState,
        displacement: float,
    ) -> None:
        # C++ parity: CTSMMCapletCalibration ctor.
        self._evolution = evolution
        self._corr = corr
        self._displaced_swap_variances = displaced_swap_variances
        self._mkt_caplet_vols = list(mkt_caplet_vols)
        self._cs = cs
        self._displacement = displacement
        self._number_of_rates = evolution.number_of_rates()

        self._mdl_caplet_vols = [0.0] * self._number_of_rates
        self._mkt_swaption_vols = [0.0] * self._number_of_rates
        self._mdl_swaption_vols = [0.0] * self._number_of_rates
        self._time_dependent_calibrated_swaption_vols: list[list[float]] = []

        # working variables
        self._used_caplet_vols: list[float] = []
        # results
        self._calibrated = False
        self._failures = 0
        self._deformation_size = _LARGE
        self._caplet_rms_error = _LARGE
        self._caplet_max_error = _LARGE
        self._swaption_rms_error = _LARGE
        self._swaption_max_error = _LARGE
        self._swap_covariance_pseudo_roots: list[Matrix] = []

        self.perform_checks(
            evolution, corr, displaced_swap_variances, self._mkt_caplet_vols, cs
        )

    # --- abstract hook -----------------------------------------------------

    @abstractmethod
    def calibration_impl(
        self,
        number_of_factors: int,
        inner_max_iterations: int,
        inner_tolerance: float,
    ) -> int:
        """Single-pass calibration; fill ``_swap_covariance_pseudo_roots`` and
        return the number of failures.

        # C++ parity: CTSMMCapletCalibration::calibrationImpl_ (pure virtual).
        """
        ...

    # --- static checks -----------------------------------------------------

    @staticmethod
    def perform_checks(
        evolution: EvolutionDescription,
        corr: PiecewiseConstantCorrelation,
        displaced_swap_variances: list[PiecewiseConstantVariance],
        mkt_caplet_vols: list[float],
        cs: CurveState,
    ) -> None:
        # C++ parity: CTSMMCapletCalibration::performChecks.
        evolution_times = evolution.evolution_times()
        qassert.require(
            evolution_times == corr.times(),
            f"evolutionTimes {evolution_times} not equal to correlation times "
            f"{corr.times()}",
        )
        rate_times = evolution.rate_times()
        qassert.require(
            rate_times == cs.rate_times(),
            "mismatch between EvolutionDescription and CurveState rate times",
        )
        number_of_rates = evolution.number_of_rates()
        qassert.require(
            number_of_rates == len(displaced_swap_variances),
            f"mismatch between EvolutionDescription number of rates "
            f"({number_of_rates}) and displacedSwapVariances size "
            f"({len(displaced_swap_variances)})",
        )
        qassert.require(
            number_of_rates == corr.number_of_rates(),
            f"mismatch between EvolutionDescription number of rates "
            f"({number_of_rates}) and corr number of rates "
            f"({corr.number_of_rates()})",
        )
        qassert.require(
            number_of_rates == len(mkt_caplet_vols),
            f"mismatch between EvolutionDescription number of rates "
            f"({number_of_rates}) and mktCapletVols size ({len(mkt_caplet_vols)})",
        )
        qassert.require(
            number_of_rates == cs.number_of_rates(),
            f"mismatch between EvolutionDescription number of rates "
            f"({number_of_rates}) and CurveState number of rates "
            f"({cs.number_of_rates()})",
        )
        temp = rate_times[:-1]
        qassert.require(
            temp == evolution_times,
            "mismatch between evolutionTimes and rateTimes",
        )
        last_swaption_vol = displaced_swap_variances[-1].total_volatility(
            number_of_rates - 1
        )
        qassert.require(
            close(last_swaption_vol, mkt_caplet_vols[number_of_rates - 1]),
            f"last caplet vol ({mkt_caplet_vols[number_of_rates - 1]!r}) must be "
            f"equal to last swaption vol ({last_swaption_vol!r}); discrepancy is "
            f"{last_swaption_vol - mkt_caplet_vols[number_of_rates - 1]}",
        )

    # --- calibration loop --------------------------------------------------

    def calibrate(
        self,
        number_of_factors: int,
        max_iterations: int,
        tolerance: float,
        inner_max_iterations: int = 100,
        inner_tolerance: float = 1e-8,
    ) -> bool:
        """Run the outer caplet-rescaling iteration. Returns ``failures == 0``.

        # C++ parity: CTSMMCapletCalibration::calibrate.
        """
        n = self._number_of_rates
        # initialise results
        self._calibrated = False
        self._failures = 987654321
        self._deformation_size = _LARGE
        self._caplet_rms_error = self._swaption_rms_error = _LARGE
        self._caplet_max_error = self._swaption_max_error = _LARGE

        # working variables
        self._used_caplet_vols = list(self._mkt_caplet_vols)

        for i in range(n):
            self._mkt_swaption_vols[i] = self._displaced_swap_variances[
                i
            ].total_volatility(i)

        displacements = [self._displacement] * n
        rate_times = self._evolution.rate_times()
        iterations = 0

        # calibration loop
        while True:
            self._failures = self.calibration_impl(
                number_of_factors, inner_max_iterations, inner_tolerance
            )

            ctsmm = PseudoRootFacade(
                self._swap_covariance_pseudo_roots,
                rate_times,
                self._cs.coterminal_swap_rates(),
                displacements,
            )
            swaption_tot_covariance = ctsmm.total_covariance(n - 1)

            flmm = CotSwapToFwdAdapter(ctsmm)
            caplet_tot_covariance = flmm.total_covariance(n - 1)

            # check fit
            self._caplet_rms_error = self._swaption_rms_error = 0.0
            self._caplet_max_error = self._swaption_max_error = -1.0

            for i in range(n):
                self._mdl_swaption_vols[i] = math.sqrt(
                    swaption_tot_covariance[i, i] / rate_times[i]
                )
                swaption_error = math.fabs(
                    self._mkt_swaption_vols[i] - self._mdl_swaption_vols[i]
                )
                self._swaption_rms_error += swaption_error * swaption_error
                self._swaption_max_error = max(
                    self._swaption_max_error, swaption_error
                )

                self._mdl_caplet_vols[i] = math.sqrt(
                    caplet_tot_covariance[i, i] / rate_times[i]
                )
                caplet_error = math.fabs(
                    self._mkt_caplet_vols[i] - self._mdl_caplet_vols[i]
                )
                self._caplet_rms_error += caplet_error * caplet_error
                self._caplet_max_error = max(self._caplet_max_error, caplet_error)

                if i < n - 1:
                    self._used_caplet_vols[i] *= (
                        self._mkt_caplet_vols[i] / self._mdl_caplet_vols[i]
                    )

            self._swaption_rms_error = math.sqrt(self._swaption_rms_error / n)
            self._caplet_rms_error = math.sqrt(self._caplet_rms_error / n)
            iterations += 1
            if not (
                iterations < max_iterations
                and self._caplet_rms_error > tolerance
            ):
                break

        ctsmm = PseudoRootFacade(
            self._swap_covariance_pseudo_roots,
            rate_times,
            self._cs.coterminal_swap_rates(),
            displacements,
        )
        self._time_dependent_calibrated_swaption_vols = [
            ctsmm.time_dependent_volatility(i) for i in range(n)
        ]

        self._calibrated = True
        return self._failures == 0

    # --- inspectors --------------------------------------------------------

    def failures(self) -> int:
        # C++ parity: CTSMMCapletCalibration::failures.
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._failures

    def deformation_size(self) -> float:
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._deformation_size

    def caplet_rms_error(self) -> float:
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._caplet_rms_error

    def caplet_max_error(self) -> float:
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._caplet_max_error

    def swaption_rms_error(self) -> float:
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._swaption_rms_error

    def swaption_max_error(self) -> float:
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._swaption_max_error

    def swap_pseudo_roots(self) -> list[Matrix]:
        # C++ parity: CTSMMCapletCalibration::swapPseudoRoots.
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._swap_covariance_pseudo_roots

    def swap_pseudo_root(self, i: int) -> Matrix:
        # C++ parity: CTSMMCapletCalibration::swapPseudoRoot.
        qassert.require(self._calibrated, "not successfully calibrated yet")
        qassert.require(
            i < len(self._swap_covariance_pseudo_roots),
            f"{i} is an invalid index, must be less than "
            f"{len(self._swap_covariance_pseudo_roots)}",
        )
        return self._swap_covariance_pseudo_roots[i]

    def mkt_caplet_vols(self) -> list[float]:
        # C++ parity: CTSMMCapletCalibration::mktCapletVols.
        return self._mkt_caplet_vols

    def mdl_caplet_vols(self) -> list[float]:
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._mdl_caplet_vols

    def mkt_swaption_vols(self) -> list[float]:
        return self._mkt_swaption_vols

    def mdl_swaption_vols(self) -> list[float]:
        qassert.require(self._calibrated, "not successfully calibrated yet")
        return self._mdl_swaption_vols

    def time_dependent_calibrated_swaption_vols(self, i: int) -> list[float]:
        # C++ parity: CTSMMCapletCalibration::timeDependentCalibratedSwaptionVols.
        qassert.require(
            i < self._number_of_rates,
            f"index ({i}) must less than number of rates "
            f"({self._number_of_rates})",
        )
        return self._time_dependent_calibrated_swaption_vols[i]

    def time_dependent_un_calibrated_swaption_vols(self, i: int) -> list[float]:
        # C++ parity: CTSMMCapletCalibration::timeDependentUnCalibratedSwaptionVols.
        qassert.require(
            i < self._number_of_rates,
            f"index ({i}) must less than number of rates "
            f"({self._number_of_rates})",
        )
        return self._displaced_swap_variances[i].volatilities()

    def curve_state(self) -> CurveState:
        # C++ parity: CTSMMCapletCalibration::curveState.
        return self._cs

    def displacements(self) -> list[float]:
        # C++ parity: CTSMMCapletCalibration::displacements.
        return [self._displacement] * self._number_of_rates

    # --- protected accessors for subclasses --------------------------------

    @property
    def number_of_rates(self) -> int:
        return self._number_of_rates

    @property
    def evolution(self) -> EvolutionDescription:
        return self._evolution
