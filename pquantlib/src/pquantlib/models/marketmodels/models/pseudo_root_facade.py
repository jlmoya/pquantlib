"""PseudoRootFacade — wrap a set of pseudo-root matrices as a MarketModel.

# C++ parity: ql/models/marketmodels/models/pseudorootfacade.{hpp,cpp}
# (v1.42.1).

Adapts an externally-supplied set of per-step covariance pseudo-root matrices
(typically the calibrated swap pseudo-roots from a caplet-coterminal
calibration) into the ``MarketModel`` interface.

The C++ class has two constructors:

1. ``PseudoRootFacade(calibrator)`` taking a ``CTSMMCapletCalibration`` — pulls
   the swap pseudo-roots, coterminal swap rates, displacements and rate times
   out of the calibrator's curve state. ``CTSMMCapletCalibration`` is a W10-C
   class, so this overload is provided as a thin classmethod
   (``from_calibrator``) that the W10-C calibration cluster wires up; until then
   it raises ``NotImplementedError`` if no calibrator object is supplied.
2. ``PseudoRootFacade(covariancePseudoRoots, rateTimes, initialRates,
   displacements)`` — the direct constructor, fully ported here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.market_model import MarketModel
from pquantlib.models.marketmodels.utilities import check_increasing_times

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.models.marketmodels.curve_state import CurveState


class _CalibratorLike(Protocol):
    """Structural interface for the W10-C ``CTSMMCapletCalibration``.

    # C++ parity: the subset of CTSMMCapletCalibration that
    PseudoRootFacade(calibrator) reads. Defined here as a Protocol so the
    cross-cluster wiring point stays statically typed before W10-C lands.
    """

    def swap_pseudo_roots(self) -> Sequence[Matrix]: ...
    def curve_state(self) -> CurveState: ...
    def displacements(self) -> list[float]: ...


class PseudoRootFacade(MarketModel):
    """A ``MarketModel`` facade over a fixed set of pseudo-root matrices.

    # C++ parity: pseudorootfacade.hpp/.cpp PseudoRootFacade.
    """

    def __init__(
        self,
        covariance_pseudo_roots: Sequence[Matrix],
        rate_times: list[float],
        initial_rates: list[float],
        displacements: list[float],
    ) -> None:
        super().__init__()
        prs = [np.asarray(m, dtype=np.float64) for m in covariance_pseudo_roots]
        self._covariance_pseudo_roots: list[Matrix] = prs
        self._number_of_factors = prs[0].shape[1]
        self._number_of_rates = prs[0].shape[0]
        self._number_of_steps = len(prs)
        self._initial_rates = list(initial_rates)
        self._displacements = list(displacements)
        self._evolution = EvolutionDescription(rate_times)

        n = self._number_of_rates
        check_increasing_times(rate_times)
        qassert.require(
            len(rate_times) > 1,
            "Rate times must contain at least two values",
        )
        qassert.require(
            n == len(rate_times) - 1,
            f"mismatch between number of rates ({n}) and rate times",
        )
        qassert.require(
            n == len(displacements),
            f"mismatch between number of rates ({n}) and displacements "
            f"({len(displacements)})",
        )
        qassert.require(
            n <= self._number_of_factors * self._number_of_steps,
            f"number of rates ({n}) greater than number of factors "
            f"({self._number_of_factors}) times number of steps "
            f"({self._number_of_steps})",
        )
        # evolutionTimes are not given for the time being
        qassert.require(
            n == len(prs),
            f"number of rates ({n}) must be equal to covariancePseudoRoots.size() "
            f"({len(prs)})",
        )
        for k in range(self._number_of_steps):
            qassert.require(
                prs[k].shape[0] == n,
                f"step {k}: pseudoRoot has wrong number of rows: {prs[k].shape[0]} "
                f"instead of {n}",
            )
            qassert.require(
                prs[k].shape[1] == self._number_of_factors,
                f"step {k}: pseudoRoot has wrong number of columns: "
                f"{prs[k].shape[1]} instead of {self._number_of_factors}",
            )

    @classmethod
    def from_calibrator(cls, calibrator: _CalibratorLike) -> PseudoRootFacade:
        """Build a facade from a ``CTSMMCapletCalibration`` (W10-C).

        # C++ parity: PseudoRootFacade(ext::shared_ptr<CTSMMCapletCalibration>).

        ``CTSMMCapletCalibration`` is a W10-C class; this classmethod is the
        wiring point for that cluster. It reads ``swap_pseudo_roots()`` +
        ``curve_state().coterminal_swap_rates()`` + ``displacements()`` +
        ``curve_state().rate_times()`` off the calibrator (typed via the
        ``_CalibratorLike`` Protocol so this stays statically checkable until
        the concrete class lands in W10-C).
        """
        curve_state = calibrator.curve_state()
        return cls(
            calibrator.swap_pseudo_roots(),
            curve_state.rate_times(),
            curve_state.coterminal_swap_rates(),
            calibrator.displacements(),
        )

    def initial_rates(self) -> list[float]:
        """The initial rates."""
        return self._initial_rates

    def displacements(self) -> list[float]:
        """The displacement (shift) of each rate."""
        return self._displacements

    def evolution(self) -> EvolutionDescription:
        """The evolution description."""
        return self._evolution

    def number_of_rates(self) -> int:
        """Number of rates."""
        return self._number_of_rates

    def number_of_factors(self) -> int:
        """Number of driving factors."""
        return self._number_of_factors

    def number_of_steps(self) -> int:
        """Number of evolution steps."""
        return self._number_of_steps

    def pseudo_root(self, i: int) -> Matrix:
        """Pseudo-square-root of the covariance matrix for step ``i``.

        # C++ parity: PseudoRootFacade::pseudoRoot.
        """
        qassert.require(
            i < self._number_of_steps,
            f"the index {i} is invalid: it must be less than number of steps "
            f"({self._number_of_steps})",
        )
        return self._covariance_pseudo_roots[i]
