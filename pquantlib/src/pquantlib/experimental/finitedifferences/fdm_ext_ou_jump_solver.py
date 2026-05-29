"""FdmExtOUJumpSolver — solver wrapper for the FdmExtOUJumpOp.

# C++ parity: ql/experimental/finitedifferences/fdmextoujumpsolver.{hpp,cpp}
# (v1.42.1).

The C++ class is a thin lazy-object wrapper around ``Fdm2DimSolver``
(generic 2-D backward FDM with cubic-spline interpolation of the
final values to query an arbitrary (x, y) state).

**Carve-out (Phase 11 W5-A):** the multi-D backward FDM framework
(``Fdm2DimSolver`` / ``Fdm3DimSolver`` / ``FdmNdimSolver<N>``) and
the Hundsdorfer / ModifiedHundsdorfer / Craig-Sneyd / Douglas /
ModifiedCraigSneyd / TR-BDF2 / ImplicitEuler / ExplicitEuler
schemes for multi-D composite operators are deferred to a follow-up
cluster. The W5-A scope covers the **operators + inner-value
calculators**; the runtime solvers will be wired in once the
multi-D scheme infrastructure lands.

For now the class is wired with the same constructor signature as
the C++ class to allow downstream code to *type-check* against it,
but ``value_at(x, y)`` raises ``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final

from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

InnerValueCalculator = Callable[[FdmLinearOpIterator, float], float]


@final
class FdmExtOUJumpSolver:
    """Backward FD solver for the ExtOU + jump op.

    # C++ parity: ``class FdmExtOUJumpSolver : public LazyObject``.

    **Carve-out:** runtime ``value_at`` requires the multi-D backward
    FDM framework (deferred). The constructor accepts the same
    parameters as the C++ class to allow downstream wiring.
    """

    def __init__(
        self,
        process: ExtOUWithJumpsProcess,
        r_ts: YieldTermStructure,
        mesher: FdmMesher,
        condition: FdmStepConditionComposite | None,
        calculator: InnerValueCalculator,
        maturity: float,
        time_steps: int,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        self._process: ExtOUWithJumpsProcess = process
        self._r_ts: YieldTermStructure = r_ts
        self._mesher: FdmMesher = mesher
        self._condition: FdmStepConditionComposite | None = condition
        self._calculator: InnerValueCalculator = calculator
        self._maturity: float = maturity
        self._time_steps: int = time_steps
        self._damping_steps: int = damping_steps
        self._scheme_desc: FdmSchemeDesc | None = scheme_desc

    def value_at(self, x: float, y: float) -> float:
        """Interpolate the rolled-back value at (x, y).

        # C++ parity: ``FdmExtOUJumpSolver::valueAt``.

        **Carve-out:** raises ``NotImplementedError`` until the
        multi-D backward FDM framework is ported.
        """
        raise NotImplementedError(
            "FdmExtOUJumpSolver.value_at requires the multi-D backward FDM "
            "framework (Fdm2DimSolver + multi-direction Hundsdorfer scheme) "
            "which is deferred to a follow-up Phase 11 cluster."
        )


__all__ = ["FdmExtOUJumpSolver"]
