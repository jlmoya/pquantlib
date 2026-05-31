"""FdmSimple3dExtOUJumpSolver — 3-D ExtOU + jump FD solver.

# C++ parity: ql/experimental/finitedifferences/fdmsimple3dextoujumpsolver.hpp
# (v1.42.1).

Used to price simple swing options whose underlying is a Kluge
ExtOU + jump (electricity) process embedded in a 3-D mesh (e.g.,
(x, y, time)). The C++ class wraps ``Fdm3DimSolver`` with an
``FdmExtOUJumpOp``.

**Carve-out (Phase 11 W5-A):** the 3-D backward FDM solver is
deferred; see ``fdm_ext_ou_jump_solver.py`` for the rationale.
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
class FdmSimple3dExtOUJumpSolver:
    """Simple 3-D backward FD solver for the Kluge ExtOU + jump op.

    # C++ parity: ``class FdmSimple3dExtOUJumpSolver : public LazyObject``.

    **Carve-out:** runtime ``value_at`` requires the multi-D
    backward FDM framework (deferred).
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

    def value_at(self, x: float, y: float, z: float) -> float:
        """Interpolate the rolled-back value at (x, y, z).

        # C++ parity: ``FdmSimple3dExtOUJumpSolver::valueAt``.

        **Carve-out:** raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "FdmSimple3dExtOUJumpSolver.value_at requires the 3-D backward FDM "
            "framework which is deferred to a follow-up Phase 11 cluster."
        )


__all__ = ["FdmSimple3dExtOUJumpSolver"]
