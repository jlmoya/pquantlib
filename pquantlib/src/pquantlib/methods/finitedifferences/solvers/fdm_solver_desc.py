"""FdmSolverDesc — config DTO for the backward FD solver.

# C++ parity: ql/methods/finitedifferences/solvers/fdmsolverdesc.hpp
# (v1.42.1) — ``struct FdmSolverDesc``.

Bundles a mesher, a step-condition composite, an inner-value
calculator, the maturity time, and the (time-step / damping-step)
counts into a single immutable config object.

The C++ struct also carries a ``FdmBoundaryConditionSet bcSet`` —
the Python port omits boundary conditions (BC handling is deferred
to Phase 6 alongside the multi-asset FD work).

The C++ ``calculator`` field is an ``FdmInnerValueCalculator``
abstract; the Python port collapses it to a callable
``calculator(iter, t) -> float`` (typically wrapping a
``Payoff`` evaluated at ``exp(log-spot)``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)

InnerValueCalculator = Callable[[FdmLinearOpIterator, float], float]


@dataclass(frozen=True, slots=True)
class FdmSolverDesc:
    """Backward-solver configuration bundle.

    # C++ parity: ``struct FdmSolverDesc``.
    """

    mesher: FdmMesher
    condition: FdmStepConditionComposite
    calculator: InnerValueCalculator
    maturity: float
    time_steps: int
    damping_steps: int


__all__ = ["FdmSolverDesc", "InnerValueCalculator"]
