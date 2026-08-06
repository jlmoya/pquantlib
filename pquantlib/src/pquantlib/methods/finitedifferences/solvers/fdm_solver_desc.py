"""FdmSolverDesc — config DTO for the backward FD solver.

# C++ parity: ql/methods/finitedifferences/solvers/fdmsolverdesc.hpp
# @ v1.43 (6b57206e0) — ``struct FdmSolverDesc``.

Bundles a mesher, a boundary-condition set, a step-condition composite, an
inner-value calculator, the maturity time, and the (time-step /
damping-step) counts into a single immutable config object.

**Field order.** C++ declares ``bcSet`` second, right after ``mesher``.
Python cannot: a dataclass field with a default must come after every field
without one, and ``bc_set`` defaults to the empty set so that call sites
which do not use boundary conditions stay positional. The field is
otherwise the same thing — every barrier, rebate and swing engine in
QuantLib fills it, and each solver hands it straight to
``FdmBackwardSolver``.

The C++ ``calculator`` field is an ``FdmInnerValueCalculator``
abstract; the Python port collapses it to a callable
``calculator(iter, t) -> float`` (typically wrapping a
``Payoff`` evaluated at ``exp(log-spot)``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    FdmBoundaryConditionSet,
)
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
    #: # C++ parity: ``const FdmBoundaryConditionSet bcSet``.
    bc_set: FdmBoundaryConditionSet = field(default=())


__all__ = ["FdmSolverDesc", "InnerValueCalculator"]
