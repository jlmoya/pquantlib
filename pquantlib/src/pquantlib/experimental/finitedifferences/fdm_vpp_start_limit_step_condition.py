"""FdmVPPStartLimitStepCondition — VPP step condition with optional start limit.

# C++ parity:
# ql/experimental/finitedifferences/fdmvppstartlimitstepcondition.{hpp,cpp}
# (v1.42.1).

Concrete subclass of :class:`FdmVPPStepCondition` that supports a
finite ``n_starts`` budget. The state machine is replicated
``n_starts + 1`` times along the state axis — each replica indexes
the residual start budget remaining — so the size is
``(2*t_min_up + t_min_down) * (n_starts + 1)``. When ``n_starts``
is ``None`` (no start limit), the state machine has just one
replica, identical to the vanilla case.

:meth:`change_state` implements the Bellman backward update used by
both :class:`DynProgVPPIntrinsicValueEngine` (intrinsic) and the
FD storage/swing/VPP engines (full pricing). The formula is exact
to C++.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.fdm_vpp_step_condition import (
    FdmVPPStepCondition,
    FdmVPPStepConditionMesher,
    FdmVPPStepConditionParams,
    InnerValueCalculator,
)
from pquantlib.math.array import Array


class FdmVPPStartLimitStepCondition(FdmVPPStepCondition):
    """VPP step condition with optional start-count budget.

    # C++ parity: ``class FdmVPPStartLimitStepCondition : public
    # FdmVPPStepCondition``.
    """

    def __init__(
        self,
        params: FdmVPPStepConditionParams,
        n_starts: int | None,
        mesh: FdmVPPStepConditionMesher,
        gas_price: InnerValueCalculator,
        spark_spread_price: InnerValueCalculator,
    ) -> None:
        super().__init__(
            params=params,
            n_states=FdmVPPStartLimitStepCondition.compute_n_states(
                params.t_min_up, params.t_min_down, n_starts
            ),
            mesh=mesh,
            gas_price=gas_price,
            spark_spread_price=spark_spread_price,
        )
        qassert.require(
            self._t_min_up > 0, "minimum up time must be greater than one"
        )
        qassert.require(
            self._t_min_down > 0, "minimum down time must be greater than one"
        )
        self._n_starts: int | None = n_starts

    @staticmethod
    def compute_n_states(
        t_min_up: int, t_min_down: int, n_starts: int | None
    ) -> int:
        """Compute the state-axis cardinality.

        # C++ parity: ``FdmVPPStartLimitStepCondition::nStates`` (static).
        Python uses ``compute_n_states`` to avoid a name clash with the
        instance ``n_states()`` accessor inherited from
        :class:`FdmVPPStepCondition`.

        Formula: ``(2*t_min_up + t_min_down) * (1 if n_starts is None
        else n_starts + 1)``.
        """
        cycle = 2 * t_min_up + t_min_down
        if n_starts is None:
            return cycle
        return cycle * (n_starts + 1)

    def max_value(self, states: Array) -> float:
        """Maximum value across the state vector.

        # C++ parity: ``maxValue`` — ``*max_element(states)``.
        """
        return float(states.max())

    def change_state(self, gas_price: float, state: Array, t: float) -> Array:
        """Bellman backward update over the state machine.

        # C++ parity: ``FdmVPPStartLimitStepCondition::changeState``.

        The state-machine cycle has length ``sss = 2*t_min_up + t_min_down``.
        The state indices encode (residual-start-replica * sss + phase):

        * ``j = i mod sss < t_min_up - 1``: PMin sub-phase (just-started,
          not yet exited the up-window). Transition: ``max(state[i+1],
          state[t_min_up + i + 1])``.
        * ``j == t_min_up - 1``: PMin -> Down switch. Transition:
          ``max(state[i + t_min_up + 1], state[i], state[i + t_min_up])``.
        * ``j < 2*t_min_up``: PMax sub-phase. Transition: copy from
          ``retVal[i - t_min_up]`` (lock to the corresponding PMin path).
        * ``j < sss - 1``: down sub-phase. Transition: ``state[i + 1]``.
        * ``j == sss - 1`` (end-of-down):
            * No start limit (``n_starts is None``): ``max(state[i],
              max(state.front(), state[t_min_up]) - start_up_cost)``.
            * Start limit + ``i >= sss`` (still have starts left):
              ``max(state[i], max(state[i+1-2*sss], state[i+1-2*sss+t_min_up])
              - start_up_cost)``.
            * Start limit + ``i < sss`` (no starts left): ``state[i]``.

        ``start_up_cost = start_up_fix_cost + (gas_price + fuel_cost_addon)
        * start_up_fuel``.
        """
        start_up_cost = (
            self._start_up_fix_cost
            + (gas_price + self._fuel_cost_addon) * self._start_up_fuel
        )
        ret_val = np.empty(state.size, dtype=np.float64)
        sss = 2 * self._t_min_up + self._t_min_down
        t_min_up = self._t_min_up

        for i in range(self._n_states):
            j = i % sss

            if j < t_min_up - 1:
                ret_val[i] = max(state[i + 1], state[t_min_up + i + 1])
            elif j == t_min_up - 1:
                ret_val[i] = max(
                    state[i + t_min_up + 1],
                    state[i],
                    state[i + t_min_up],
                )
            elif j < 2 * t_min_up:
                # C++ uses retVal[i - t_min_up] (already-written PMin
                # cell within this same iteration); preserves the
                # in-order data dependency from j < t_min_up branch.
                ret_val[i] = ret_val[i - t_min_up]
            elif j < 2 * t_min_up + self._t_min_down - 1:
                ret_val[i] = state[i + 1]
            elif self._n_starts is None:
                ret_val[i] = max(
                    state[i],
                    max(state[0], state[t_min_up]) - start_up_cost,
                )
            elif i >= sss:
                # Have starts left in the budget: pay start_up_cost
                # to jump back to (residual_starts - 1) PMin/PMax entry.
                idx = i + 1 - 2 * sss
                ret_val[i] = max(
                    state[i],
                    max(state[idx], state[idx + t_min_up]) - start_up_cost,
                )
            else:
                # Start budget exhausted: must stay off.
                ret_val[i] = state[i]

        return ret_val


__all__ = ["FdmVPPStartLimitStepCondition"]
