"""FdmVPPStepCondition — backward-induction step condition for VPP.

# C++ parity:
# ql/experimental/finitedifferences/fdmvppstepcondition.{hpp,cpp} (v1.42.1).

At each backward step the VPP step condition:

1. Adds an "evolve" term to each grid node — the per-hour profit if
   the plant is in a running state (PMin or PMax phase of the
   state-machine cycle), zero otherwise. The state-machine cycle has
   length ``2 * t_min_up + t_min_down`` per ``j = i mod cycle``:

   * ``0 <= j < t_min_up`` — PMin phase (just-started, must run at
     ``p_min``).
   * ``t_min_up <= j < 2 * t_min_up`` — PMax phase (running at
     ``p_max``).
   * ``2 * t_min_up <= j < cycle`` — Down phase (must remain off).

2. For each grid point with state-coordinate 0, reads off the full
   state vector along the state axis, applies the ``change_state``
   transition (subclass-specific — the "change_state" matrix), and
   writes the values back. This is the Bellman backward update over
   the discrete state machine.

The C++ class is templated on ``Array``. The Python port specialises
on :class:`numpy.ndarray` (the existing :class:`StepCondition` base).

The inner-value calculators are :class:`InnerValueCalculator`
callables (``(iter, t) -> float``) — matching the existing
:class:`FdmSolverDesc` collapse of C++'s
``FdmInnerValueCalculator`` abstract interface.

Subclasses MUST implement:

* ``change_state(gas_price, state, t) -> Array`` — the
  state-transition matrix update.
* ``max_value(states) -> float`` — extract the option NPV at a
  single grid point's state vector.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)

InnerValueCalculator = Callable[[FdmLinearOpIterator, float], float]


@dataclass(frozen=True, slots=True)
class FdmVPPStepConditionParams:
    """Aggregated VPP plant parameters used by the step condition.

    # C++ parity: ``struct FdmVPPStepConditionParams``.
    """

    heat_rate: float
    p_min: float
    p_max: float
    t_min_up: int
    t_min_down: int
    start_up_fuel: float
    start_up_fix_cost: float
    fuel_cost_addon: float


@dataclass(frozen=True, slots=True)
class FdmVPPStepConditionMesher:
    """Mesher + state-axis selection for the step condition.

    # C++ parity: ``struct FdmVPPStepConditionMesher``.
    """

    state_direction: int
    mesher: FdmMesher


class FdmVPPStepCondition(StepCondition):
    """Abstract VPP step condition.

    # C++ parity: ``class FdmVPPStepCondition : public StepCondition<Array>``.

    Subclasses (:class:`FdmVPPStartLimitStepCondition`) implement
    :meth:`change_state` and :meth:`max_value`.
    """

    def __init__(
        self,
        params: FdmVPPStepConditionParams,
        n_states: int,
        mesh: FdmVPPStepConditionMesher,
        gas_price: InnerValueCalculator,
        spark_spread_price: InnerValueCalculator,
    ) -> None:
        self._heat_rate: float = params.heat_rate
        self._p_min: float = params.p_min
        self._p_max: float = params.p_max
        self._t_min_up: int = params.t_min_up
        self._t_min_down: int = params.t_min_down
        self._start_up_fuel: float = params.start_up_fuel
        self._start_up_fix_cost: float = params.start_up_fix_cost
        self._fuel_cost_addon: float = params.fuel_cost_addon
        self._state_direction: int = mesh.state_direction
        self._n_states: int = n_states
        self._mesher: FdmMesher = mesh.mesher
        self._gas_price: InnerValueCalculator = gas_price
        self._spark_spread_price: InnerValueCalculator = spark_spread_price

        layout_dim = self._mesher.layout().dim()
        qassert.require(
            n_states == layout_dim[self._state_direction],
            "mesher does not fit to vpp arguments",
        )
        # Pre-populate per-state evolve function = phase classifier.
        # C++ parity: ``stateEvolveFcts_`` vector; entries are either
        # ``evolveAtPMin``, ``evolveAtPMax`` (capturing ``this``), or
        # null. The Python port uses ``None`` for the null case and a
        # plain int phase tag (-1 / 0 / 1) for fast dispatch; the
        # actual evolve formulas are inlined in :meth:`_evolve`.
        cycle = 2 * self._t_min_up + self._t_min_down
        # phase[i] = 0 → PMin, 1 → PMax, -1 → idle/down.
        self._phase: list[int] = [-1] * n_states
        for i in range(n_states):
            j = i % cycle
            if j < self._t_min_up:
                self._phase[i] = 0  # PMin
            elif j < 2 * self._t_min_up:
                self._phase[i] = 1  # PMax
            # else: idle/down phase, no evolve term.

    def n_states(self) -> int:
        """Number of state-machine states along the state axis."""
        return self._n_states

    @abstractmethod
    def max_value(self, states: Array) -> float:
        """Reduce a per-state value vector to a single NPV.

        # C++ parity: pure-virtual ``maxValue(const Array&) const``.
        """

    @abstractmethod
    def change_state(self, gas_price: float, state: Array, t: float) -> Array:
        """Apply the Bellman state-transition matrix at time ``t``.

        # C++ parity: pure-virtual ``changeState(Real, const Array&,
        # Time) const``.
        """

    def evolve_at_p_min(self, spark_spread: float) -> float:
        """Per-hour profit at PMin phase.

        # C++ parity: ``evolveAtPMin``.
        Formula: ``p_min * (spark_spread - heat_rate * fuel_cost_addon)``.
        """
        return self._p_min * (spark_spread - self._heat_rate * self._fuel_cost_addon)

    def evolve_at_p_max(self, spark_spread: float) -> float:
        """Per-hour profit at PMax phase.

        # C++ parity: ``evolveAtPMax``.
        Formula: ``p_max * (spark_spread - heat_rate * fuel_cost_addon)``.
        """
        return self._p_max * (spark_spread - self._heat_rate * self._fuel_cost_addon)

    def _evolve(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Per-node evolve term used by :meth:`apply_to`."""
        state = iterator.coordinates[self._state_direction]
        phase = self._phase[state]
        if phase < 0:
            return 0.0
        spark_spread = self._spark_spread_price(iterator, t)
        if phase == 0:
            return self.evolve_at_p_min(spark_spread)
        return self.evolve_at_p_max(spark_spread)

    def apply_to(self, a: Array, t: float) -> None:
        """One backward induction step at time ``t``.

        # C++ parity: ``FdmVPPStepCondition::applyTo``.
        """
        layout = self._mesher.layout()
        state_dim = layout.dim()[self._state_direction]

        # Phase 1: add the per-hour profit term in place.
        for iter_ in layout.iter():
            a[iter_.index] += self._evolve(iter_, t)

        # Phase 2: apply the state-transition matrix at each grid point
        # whose state-coordinate is 0 (the matrix touches every state
        # along the axis simultaneously).
        for iter_ in layout.iter():
            if iter_.coordinates[self._state_direction] != 0:
                continue
            # Gather the full state vector along the state axis.
            x = np.empty(state_dim, dtype=np.float64)
            for i in range(state_dim):
                x[i] = a[layout.neighbourhood(iter_, self._state_direction, i)]
            gas_price = self._gas_price(iter_, t)
            x = self.change_state(gas_price, x, t)
            for i in range(state_dim):
                a[layout.neighbourhood(iter_, self._state_direction, i)] = x[i]


__all__ = [
    "FdmVPPStepCondition",
    "FdmVPPStepConditionMesher",
    "FdmVPPStepConditionParams",
    "InnerValueCalculator",
]
