"""TsiveriotisFernandesLattice — credit-adjusted binomial lattice.

# C++ parity: ql/methods/lattices/tflattice.hpp (v1.42.1) —
#             ``template <class T> class TsiveriotisFernandesLattice``.

A :class:`~pquantlib.methods.lattices.bsm_lattice.BlackScholesLattice`
specialisation implementing the Tsiveriotis-Fernandes (1998) convertible-bond
model: the bond value is rolled back with a *blended* discount rate that
interpolates between the risk-free rate (when conversion is certain) and the
risk-free + credit-spread rate (when conversion is impossible), weighted by
the backward-induced conversion probability.

This lattice only works with :class:`DiscretizedConvertible` — its
``rollback`` / ``partial_rollback`` drive the convertible's three coupled
arrays (``values`` / ``conversion_probability`` / ``spread_adjusted_rate``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice
from pquantlib.pricingengines.bond.discretized_convertible import (
    DiscretizedConvertible,
)

if TYPE_CHECKING:
    from pquantlib.methods.lattices.binomial_tree import BinomialTree


class TsiveriotisFernandesLattice(BlackScholesLattice):
    """Binomial lattice approximating the Tsiveriotis-Fernandes model.

    # C++ parity: ``class TsiveriotisFernandesLattice<T>`` (tflattice.hpp).
    """

    def __init__(
        self,
        tree: BinomialTree,
        risk_free_rate: float,
        end: float,
        steps: int,
        credit_spread: float,
        volatility: float,
        div_yield: float,
    ) -> None:
        # C++ parity: tflattice.hpp:69-84.
        super().__init__(tree, risk_free_rate, end, steps)
        self._credit_spread: float = credit_spread
        qassert.require(self._pu <= 1.0, f"probability ({self._pu}) higher than one")
        qassert.require(self._pu >= 0.0, f"negative ({self._pu}) probability")

    def credit_spread(self) -> float:
        """C++ parity: ``creditSpread()`` (tflattice.hpp:49)."""
        return self._credit_spread

    # -- TF-specific stepback ---------------------------------------------

    def _tf_stepback(
        self,
        i: int,
        values: Array,
        conversion_probability: Array,
        spread_adjusted_rate: Array,
    ) -> tuple[Array, Array, Array]:
        # C++ parity: tflattice.hpp:86-116.
        n_i = self.size(i)
        pd = self._pd
        pu = self._pu
        rf = self._risk_free_rate
        dt = self._dt

        cp_down = conversion_probability[0:n_i]
        cp_up = conversion_probability[1 : n_i + 1]
        new_conv_prob = pd * cp_down + pu * cp_up

        new_spread_adjusted_rate = (
            new_conv_prob * rf + (1.0 - new_conv_prob) * (rf + self._credit_spread)
        )

        sar_down = spread_adjusted_rate[0:n_i]
        sar_up = spread_adjusted_rate[1 : n_i + 1]
        v_down = values[0:n_i]
        v_up = values[1 : n_i + 1]
        new_values = pd * v_down / (1.0 + sar_down * dt) + pu * v_up / (
            1.0 + sar_up * dt
        )

        return (
            np.asarray(new_values, dtype=np.float64),
            np.asarray(new_conv_prob, dtype=np.float64),
            np.asarray(new_spread_adjusted_rate, dtype=np.float64),
        )

    # -- Lattice rollback overrides ---------------------------------------

    def rollback(self, asset: object, to_t: float) -> None:
        # C++ parity: tflattice.hpp:118-123.
        self.partial_rollback(asset, to_t)
        _as_convertible(asset).adjust_values()

    def partial_rollback(self, asset: object, to_t: float) -> None:
        # C++ parity: tflattice.hpp:126-164.
        convertible = _as_convertible(asset)
        from_t = convertible.time
        if close(from_t, to_t):
            return
        qassert.require(
            from_t > to_t,
            f"cannot roll the asset back to {to_t} (it is already at t = {from_t})",
        )

        i_from = self._time_grid.index(from_t)
        i_to = self._time_grid.index(to_t)

        for i in range(i_from - 1, i_to - 1, -1):
            new_values, new_conv_prob, new_sar = self._tf_stepback(
                i,
                convertible.values,
                convertible.conversion_probability,
                convertible.spread_adjusted_rate,
            )
            convertible.set_time(self._time_grid[i])
            convertible.set_values(new_values)
            convertible.spread_adjusted_rate = new_sar
            convertible.conversion_probability = new_conv_prob
            # skip the very last adjustment
            if i != i_to:
                convertible.adjust_values()


def _as_convertible(asset: object) -> DiscretizedConvertible:
    if not isinstance(asset, DiscretizedConvertible):
        raise TypeError("asset must be a DiscretizedConvertible")
    return asset


__all__ = ["TsiveriotisFernandesLattice"]
