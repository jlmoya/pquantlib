"""VolatilityBumpInstrumentJacobian + OrthogonalizedBumpFinder.

# C++ parity: ql/models/marketmodels/pathwisegreeks/bumpinstrumentjacobian.
# {hpp,cpp} (v1.42.1).

``VolatilityBumpInstrumentJacobian`` gives, for each instrument (swaption or
cap) and each vega-bump cluster, the derivative of the instrument implied vol
with respect to that bump, plus the "one-percent bump" — the smallest bump
(``0.01 v / <v,v>``) that shifts the instrument implied vol by one percent.

``OrthogonalizedBumpFinder`` takes a market model, a list of instruments, and a
set of possible bumps, and produces orthogonalised pseudo-root bump directions
that shift each instrument's implied vol by one percent while leaving the others
fixed (discarding instruments too correlated with the rest). Its output is
exactly the ``std::vector<std::vector<Matrix>>`` consumed by
``PathwiseVegasAccountingEngine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrixutilities.basis_incomplete_ordered import (
    OrthogonalProjections,
)
from pquantlib.models.marketmodels.pathwisegreeks.swaption_pseudo_jacobian import (
    CapPseudoDerivative,
    SwaptionPseudoDerivative,
)

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix
    from pquantlib.models.marketmodels.pathwisegreeks.vega_bump_cluster import (
        VegaBumpCollection,
    )


@dataclass(frozen=True, slots=True)
class Swaption:
    """A coterminal swaption defined by its rate-index range.

    # C++ parity: VolatilityBumpInstrumentJacobian::Swaption.
    """

    start_index: int
    end_index: int


@dataclass(frozen=True, slots=True)
class Cap:
    """A cap defined by its rate-index range and strike.

    # C++ parity: VolatilityBumpInstrumentJacobian::Cap.
    """

    start_index: int
    end_index: int
    strike: float


class VolatilityBumpInstrumentJacobian:
    """Instrument-level vega-bump Jacobian + one-percent bumps.

    # C++ parity: VolatilityBumpInstrumentJacobian.
    """

    def __init__(
        self,
        bumps: VegaBumpCollection,
        swaptions: list[Swaption],
        caps: list[Cap],
    ) -> None:
        # C++ parity: VolatilityBumpInstrumentJacobian ctor.
        self._bumps = bumps
        self._swaptions = list(swaptions)
        self._caps = list(caps)
        n_instruments = len(swaptions) + len(caps)
        n_bumps = bumps.number_bumps()
        self._computed = [False] * n_instruments
        self._all_computed = False
        self._derivatives: list[list[float]] = [
            [0.0] * n_bumps for _ in range(n_instruments)
        ]
        self._one_percent_bumps: list[list[float]] = [
            [0.0] * n_bumps for _ in range(n_instruments)
        ]
        self._bump_matrix: Matrix = np.zeros((n_instruments, n_bumps), dtype=np.float64)

    def get_input_bumps(self) -> VegaBumpCollection:
        # C++ parity: VolatilityBumpInstrumentJacobian::getInputBumps.
        return self._bumps

    def derivatives_volatility(self, j: int) -> list[float]:
        # C++ parity: VolatilityBumpInstrumentJacobian::derivativesVolatility.
        qassert.require(
            j < len(self._swaptions) + len(self._caps),
            "too high index passed to "
            "VolatilityBumpInstrumentJacobian::derivativesVolatility",
        )

        if self._computed[j]:
            return self._derivatives[j]

        n_bumps = self._bumps.number_bumps()
        sizesq = 0.0
        self._computed[j] = True
        init_j = j
        all_bumps = self._bumps.all_bumps()

        if j < len(self._swaptions):  # it's a swaption
            this_pseudo = SwaptionPseudoDerivative(
                self._bumps.associated_model(),
                self._swaptions[j].start_index,
                self._swaptions[j].end_index,
            )
            for k in range(n_bumps):
                v = 0.0
                cl = all_bumps[k]
                for i in range(cl.step_begin(), cl.step_end()):
                    full_derivative = this_pseudo.volatility_derivative(i)
                    for f in range(cl.factor_begin(), cl.factor_end()):
                        for r in range(cl.rate_begin(), cl.rate_end()):
                            v += full_derivative[r][f]
                self._derivatives[j][k] = v
                sizesq += v * v
        else:  # it's a cap
            j -= len(self._swaptions)
            # first df shouldn't make any difference
            this_pseudo = CapPseudoDerivative(  # type: ignore[assignment]
                self._bumps.associated_model(),
                self._caps[j].strike,
                self._caps[j].start_index,
                self._caps[j].end_index,
                1.0,
            )
            for k in range(n_bumps):
                v = 0.0
                cl = all_bumps[k]
                for i in range(cl.step_begin(), cl.step_end()):
                    full_derivative = this_pseudo.volatility_derivative(i)
                    for f in range(cl.factor_begin(), cl.factor_end()):
                        for r in range(cl.rate_begin(), cl.rate_end()):
                            v += full_derivative[r][f]
                sizesq += v * v
                self._derivatives[init_j][k] = v

        for k in range(n_bumps):
            self._bump_matrix[init_j][k] = self._one_percent_bumps[init_j][k] = (
                0.01 * self._derivatives[init_j][k] / sizesq
            )

        return self._derivatives[init_j]

    def one_percent_bump(self, j: int) -> list[float]:
        # C++ parity: VolatilityBumpInstrumentJacobian::onePercentBump.
        self.derivatives_volatility(j)
        return self._one_percent_bumps[j]

    def get_all_one_percent_bumps(self) -> Matrix:
        # C++ parity: VolatilityBumpInstrumentJacobian::getAllOnePercentBumps.
        if not self._all_computed:
            for i in range(len(self._swaptions) + len(self._caps)):
                self.derivatives_volatility(i)
        self._all_computed = True
        return self._bump_matrix


class OrthogonalizedBumpFinder:
    """Orthogonalised pseudo-root bump directions for pathwise vegas.

    # C++ parity: OrthogonalizedBumpFinder.
    """

    def __init__(
        self,
        bumps: VegaBumpCollection,
        swaptions: list[Swaption],
        caps: list[Cap],
        multiplier_cut_off: float,
        tolerance: float,
    ) -> None:
        # C++ parity: OrthogonalizedBumpFinder ctor.
        self._derivatives_producer = VolatilityBumpInstrumentJacobian(
            bumps, swaptions, caps
        )
        self._multiplier_cut_off = multiplier_cut_off
        self._tolerance = tolerance

    def get_vega_bumps(self) -> list[list[Matrix]]:
        """Return ``the_bumps[step][bump]`` pseudo-root bump matrices.

        # C++ parity: OrthogonalizedBumpFinder::GetVegaBumps.

        The outer index is the time step, the inner index the (restricted)
        vega; this is precisely the structure consumed by
        ``PathwiseVegasAccountingEngine``.
        """
        projector = OrthogonalProjections(
            self._derivatives_producer.get_all_one_percent_bumps(),
            self._multiplier_cut_off,
            self._tolerance,
        )

        number_restricted_bumps = projector.number_valid_vectors()

        marketmodel = self._derivatives_producer.get_input_bumps().associated_model()
        evolution = marketmodel.evolution()
        number_steps = evolution.number_of_steps()
        number_rates = evolution.number_of_rates()
        factors = marketmodel.number_of_factors()

        # outer by time step, inner by which vega
        the_bumps: list[list[Matrix]] = [
            [
                np.zeros((number_rates, factors), dtype=np.float64)
                for _ in range(number_restricted_bumps)
            ]
            for _ in range(number_steps)
        ]

        bump_clusters = self._derivatives_producer.get_input_bumps().all_bumps()
        valid = projector.valid_vectors()

        bump_index = 0
        for instrument in range(len(valid)):
            if valid[instrument]:
                vector = projector.get_vector(instrument)
                for cluster in range(len(bump_clusters)):
                    magnitude = vector[cluster]
                    cl = bump_clusters[cluster]
                    for step in range(cl.step_begin(), cl.step_end()):
                        for rate in range(cl.rate_begin(), cl.rate_end()):
                            for factor in range(cl.factor_begin(), cl.factor_end()):
                                the_bumps[step][bump_index][rate][factor] = magnitude
                bump_index += 1

        return the_bumps
