"""FwdPeriodAdapter — fine-grid -> coarse-period forward MarketModel.

# C++ parity: ql/models/marketmodels/models/fwdperiodadapter.{hpp,cpp}
# (v1.42.1).

Adapts a fine-grid (e.g. semi-annual) forward ``MarketModel`` onto a coarser
periodic grid (e.g. annual), keeping every ``period`` rate starting at
``offset``. The coarse-grid pseudo-roots are re-expressed via the
forward-forward ``Y``-matrix (``smallPseudoRoot[k] = Y @ largePseudoRoot[k]``),
with already-expired rates zeroed out per step.

Displacements: if ``new_displacements`` is empty, the per-coarse-rate
displacement is the average of the underlying fine-grid displacements; if it
has length 1, that value is broadcast; otherwise it must already match the
coarse-rate count.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.forward_forward_mappings import ForwardForwardMappings
from pquantlib.models.marketmodels.market_model import MarketModel


class FwdPeriodAdapter(MarketModel):
    """Adapts a fine-grid forward model onto a coarser periodic grid.

    # C++ parity: fwdperiodadapter.hpp/.cpp FwdPeriodAdapter.
    """

    def __init__(
        self,
        large_model: MarketModel,
        period: int,
        offset: int,
        new_displacements: list[float],
    ) -> None:
        super().__init__()
        self._number_of_factors = large_model.number_of_factors()
        self._number_of_rates = (large_model.number_of_rates() - offset) // (
            period if period > 0 else 1
        )
        self._number_of_steps = large_model.number_of_steps()
        displacements: list[float] = list(new_displacements)

        qassert.require(
            period > 0, "period must  be greater than zero in fwdperiodadapter"
        )
        qassert.require(
            period > offset, "period must be greater than offset in fwdperiodadapter"
        )

        large_displacements = large_model.displacements()

        if len(displacements) == 1:
            dis = displacements[0]
            displacements = [dis] * self._number_of_rates
        elif len(displacements) == 0:
            # if not specified use average across rate.
            # C++ parity: fwdperiodadapter.cpp declares ``sum`` OUTSIDE the
            # coarse-rate loop and never resets it, so the k-th coarse
            # displacement is the running *cumulative* sum of all fine
            # displacements up to (k+1)*period, divided by period (not a true
            # per-period average). We replicate this quirk verbatim.
            displacements = []
            m = 0
            total = 0.0
            for _k in range(self._number_of_rates):
                for _ell in range(period):
                    total += large_displacements[m]
                    m += 1
                displacements.append(total / period)
        qassert.require(
            len(displacements) == self._number_of_rates,
            "newDisplacements should be empty,1, or number of new rates in "
            "fwdperiodadapter",
        )
        self._displacements = displacements

        large_cs = LMMCurveState(large_model.evolution().rate_times())
        large_cs.set_on_forward_rates(large_model.initial_rates())

        small_cs = ForwardForwardMappings.restrict_curve_state(
            large_cs, period, offset
        )

        self._initial_rates = small_cs.forward_rates()

        # C++ parity: smallCS.rateTimes()[smallCS.numberOfRates()-1].
        final_reset = small_cs.rate_times()[small_cs.number_of_rates() - 1]
        old_evolution_times = large_model.evolution().evolution_times()
        new_evolution_times = [
            t for t in old_evolution_times if t <= final_reset
        ]

        self._evolution = EvolutionDescription(
            small_cs.rate_times(), new_evolution_times
        )
        self._number_of_steps = len(new_evolution_times)

        rate_times = small_cs.rate_times()
        evolution_times = self._evolution.evolution_times()
        set_times = set(evolution_times)
        for i in range(len(rate_times) - 1):
            qassert.require(
                rate_times[i] in set_times,
                "every new rate time except last must be an evolution time in "
                "fwdperiod adapter",
            )

        y_matrix = ForwardForwardMappings.y_matrix(
            large_cs, large_displacements, self._displacements, period, offset
        )

        alive = self._evolution.first_alive_rate()
        self._pseudo_roots: list[Matrix] = []
        for k in range(self._number_of_steps):
            pr = y_matrix @ large_model.pseudo_root(k)
            for i in range(alive[k]):
                pr[i, :] = 0.0
            self._pseudo_roots.append(np.asarray(pr, dtype=np.float64))

    def initial_rates(self) -> list[float]:
        """The initial (coarse-grid) forward rates."""
        return self._initial_rates

    def displacements(self) -> list[float]:
        """The displacement (shift) of each coarse rate."""
        return self._displacements

    def evolution(self) -> EvolutionDescription:
        """The (coarse-grid) evolution description."""
        return self._evolution

    def number_of_rates(self) -> int:
        """Number of coarse rates."""
        return self._number_of_rates

    def number_of_factors(self) -> int:
        """Number of driving factors."""
        return self._number_of_factors

    def number_of_steps(self) -> int:
        """Number of evolution steps."""
        return self._number_of_steps

    def pseudo_root(self, i: int) -> Matrix:
        """Pseudo-square-root of the covariance matrix for step ``i``.

        # C++ parity: FwdPeriodAdapter::pseudoRoot.
        """
        return self._pseudo_roots[i]
