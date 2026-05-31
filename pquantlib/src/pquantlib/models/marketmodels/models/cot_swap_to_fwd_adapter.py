"""CotSwapToFwdAdapter — coterminal-swap-rate -> forward-rate MarketModel.

# C++ parity: ql/models/marketmodels/models/cotswaptofwdadapter.{hpp,cpp}
# (v1.42.1).

The inverse of ``FwdToCotSwapAdapter``: adapts a coterminal-swap-rate
``MarketModel`` into forward-rate dynamics. The pseudo-roots are re-expressed
in forward-rate coordinates via the inverse coterminal-swap Z-matrix
(``fwdPseudoRoot[k] = Z^{-1} @ swapPseudoRoot[k]``), with already-expired rates
zeroed out per step.

``CotSwapToFwdAdapterFactory`` wraps a coterminal-swap ``MarketModelFactory``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.curvestates.coterminal_swap_curve_state import (
    CoterminalSwapCurveState,
)
from pquantlib.models.marketmodels.market_model import MarketModel, MarketModelFactory
from pquantlib.models.marketmodels.swap_forward_mappings import SwapForwardMappings

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription


class CotSwapToFwdAdapter(MarketModel):
    """Adapts a coterminal-swap-rate model to forward-rate dynamics.

    # C++ parity: cotswaptofwdadapter.hpp/.cpp CotSwapToFwdAdapter.
    """

    def __init__(self, coterminal_model: MarketModel) -> None:
        super().__init__()
        self._coterminal_model = coterminal_model
        self._number_of_factors = coterminal_model.number_of_factors()
        self._number_of_rates = coterminal_model.number_of_rates()
        self._number_of_steps = coterminal_model.number_of_steps()

        displacements = coterminal_model.displacements()
        for i in range(1, len(displacements)):
            qassert.require(
                displacements[i] == displacements[0],
                f"{i + 1}th displacement ({displacements[i]}) not equal to the "
                f"previous ones ({displacements[0]})",
            )

        rate_times = coterminal_model.evolution().rate_times()
        # we must ensure we step through all rateTimes
        evolution_times = coterminal_model.evolution().evolution_times()
        i = 0
        while i < len(rate_times) and rate_times[i] <= evolution_times[-1]:
            qassert.require(
                rate_times[i] in evolution_times,
                f"skipping {i + 1}th rate time",
            )
            i += 1

        cs = CoterminalSwapCurveState(rate_times)
        initial_coterminal_swap_rates = coterminal_model.initial_rates()
        cs.set_on_coterminal_swap_rates(initial_coterminal_swap_rates)
        self._initial_rates = cs.forward_rates()

        zed_matrix = SwapForwardMappings.coterminal_swap_zed_matrix(
            cs, displacements[0]
        )
        inverted_zed_matrix = np.linalg.inv(np.asarray(zed_matrix, dtype=np.float64))

        alive = coterminal_model.evolution().first_alive_rate()
        self._pseudo_roots: list[Matrix] = []
        for k in range(self._number_of_steps):
            pr = inverted_zed_matrix @ coterminal_model.pseudo_root(k)
            for i in range(alive[k]):
                pr[i, :] = 0.0
            self._pseudo_roots.append(np.asarray(pr, dtype=np.float64))

    def initial_rates(self) -> list[float]:
        """The initial forward rates."""
        return self._initial_rates

    def displacements(self) -> list[float]:
        """The displacement (shift) of each rate (from the coterminal model)."""
        return self._coterminal_model.displacements()

    def evolution(self) -> EvolutionDescription:
        """The evolution description (from the coterminal model)."""
        return self._coterminal_model.evolution()

    def number_of_rates(self) -> int:
        """Number of rates."""
        return self._coterminal_model.number_of_rates()

    def number_of_factors(self) -> int:
        """Number of driving factors."""
        return self._coterminal_model.number_of_factors()

    def number_of_steps(self) -> int:
        """Number of evolution steps."""
        return self._coterminal_model.number_of_steps()

    def pseudo_root(self, i: int) -> Matrix:
        """Pseudo-square-root of the (forward) covariance matrix for step ``i``.

        # C++ parity: CotSwapToFwdAdapter::pseudoRoot.
        """
        return self._pseudo_roots[i]


class CotSwapToFwdAdapterFactory(MarketModelFactory):
    """Wraps a coterminal-swap ``MarketModelFactory`` as a forward factory.

    # C++ parity: cotswaptofwdadapter.hpp/.cpp CotSwapToFwdAdapterFactory.
    """

    def __init__(self, coterminal_factory: MarketModelFactory) -> None:
        self._coterminal_factory = coterminal_factory

    def create(
        self, evolution: EvolutionDescription, number_of_factors: int
    ) -> MarketModel:
        """Build a ``CotSwapToFwdAdapter`` around the wrapped factory's model.

        # C++ parity: CotSwapToFwdAdapterFactory::create.
        """
        coterminal_model = self._coterminal_factory.create(
            evolution, number_of_factors
        )
        return CotSwapToFwdAdapter(coterminal_model)
