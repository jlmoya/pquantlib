"""FwdToCotSwapAdapter — forward-rate -> coterminal-swap-rate MarketModel.

# C++ parity: ql/models/marketmodels/models/fwdtocotswapadapter.{hpp,cpp}
# (v1.42.1).

Adapts a forward-rate ``MarketModel`` into the dynamics of the coterminal swap
rates: the pseudo-roots are re-expressed in swap-rate coordinates via the
coterminal-swap Z-matrix ``Z`` (``swapPseudoRoot[k] = Z @ fwdPseudoRoot[k]``),
with already-expired rates zeroed out per step.

``FwdToCotSwapAdapterFactory`` wraps a forward ``MarketModelFactory``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.market_model import MarketModel, MarketModelFactory
from pquantlib.models.marketmodels.swap_forward_mappings import SwapForwardMappings

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription


class FwdToCotSwapAdapter(MarketModel):
    """Adapts a forward-rate model to coterminal-swap-rate dynamics.

    # C++ parity: fwdtocotswapadapter.hpp/.cpp FwdToCotSwapAdapter.
    """

    def __init__(self, forward_model: MarketModel) -> None:
        super().__init__()
        self._fwd_model = forward_model
        self._number_of_factors = forward_model.number_of_factors()
        self._number_of_rates = forward_model.number_of_rates()
        self._number_of_steps = forward_model.number_of_steps()

        displacements = forward_model.displacements()
        for i in range(1, len(displacements)):
            qassert.require(
                displacements[i] == displacements[0],
                f"{i + 1}th displacement ({displacements[i]}) not equal to the "
                f"previous ones ({displacements[0]})",
            )

        rate_times = forward_model.evolution().rate_times()
        # we must ensure we step through all rateTimes
        evolution_times = forward_model.evolution().evolution_times()
        i = 0
        while i < len(rate_times) and rate_times[i] <= evolution_times[-1]:
            qassert.require(
                rate_times[i] in evolution_times,
                f"skipping {i + 1}th rate time",
            )
            i += 1

        cs = LMMCurveState(rate_times)
        initial_fwd_rates = forward_model.initial_rates()
        cs.set_on_forward_rates(initial_fwd_rates)
        self._initial_rates = cs.coterminal_swap_rates()

        zed_matrix = SwapForwardMappings.coterminal_swap_zed_matrix(
            cs, displacements[0]
        )

        alive = forward_model.evolution().first_alive_rate()
        self._pseudo_roots: list[Matrix] = []
        for k in range(self._number_of_steps):
            pr = zed_matrix @ forward_model.pseudo_root(k)
            for i in range(alive[k]):
                pr[i, :] = 0.0
            self._pseudo_roots.append(np.asarray(pr, dtype=np.float64))

    def initial_rates(self) -> list[float]:
        """The initial coterminal swap rates."""
        return self._initial_rates

    def displacements(self) -> list[float]:
        """The displacement (shift) of each rate (from the forward model)."""
        return self._fwd_model.displacements()

    def evolution(self) -> EvolutionDescription:
        """The evolution description (from the forward model)."""
        return self._fwd_model.evolution()

    def number_of_rates(self) -> int:
        """Number of rates."""
        return self._fwd_model.number_of_rates()

    def number_of_factors(self) -> int:
        """Number of driving factors."""
        return self._fwd_model.number_of_factors()

    def number_of_steps(self) -> int:
        """Number of evolution steps."""
        return self._fwd_model.number_of_steps()

    def pseudo_root(self, i: int) -> Matrix:
        """Pseudo-square-root of the (swap) covariance matrix for step ``i``.

        # C++ parity: FwdToCotSwapAdapter::pseudoRoot.
        """
        return self._pseudo_roots[i]


class FwdToCotSwapAdapterFactory(MarketModelFactory):
    """Wraps a forward ``MarketModelFactory`` as a coterminal-swap factory.

    # C++ parity: fwdtocotswapadapter.hpp/.cpp FwdToCotSwapAdapterFactory.
    """

    def __init__(self, forward_factory: MarketModelFactory) -> None:
        self._forward_factory = forward_factory

    def create(
        self, evolution: EvolutionDescription, number_of_factors: int
    ) -> MarketModel:
        """Build a ``FwdToCotSwapAdapter`` around the wrapped factory's model.

        # C++ parity: FwdToCotSwapAdapterFactory::create.
        """
        forward_model = self._forward_factory.create(evolution, number_of_factors)
        return FwdToCotSwapAdapter(forward_model)
