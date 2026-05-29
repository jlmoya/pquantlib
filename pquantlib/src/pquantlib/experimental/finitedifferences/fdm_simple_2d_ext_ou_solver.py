"""FdmSimple2dExtOUSolver — 2-D ExtOU FD solver.

# C++ parity: ql/experimental/finitedifferences/fdmsimple2dextousolver.hpp
# (v1.42.1).

Used to price simple swing options whose underlying is an ExtOU
process. The C++ class wraps ``Fdm2DimSolver`` with an
``FdmExtendedOrnsteinUhlenbeckOp`` configured for the 2-D mesh.

**Carve-out (Phase 11 W5-A):** the 2-D backward FDM solver is
deferred; see ``fdm_ext_ou_jump_solver.py`` for the rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final

from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
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
class FdmSimple2dExtOUSolver:
    """Simple 2-D backward FD solver for the ExtOU op.

    # C++ parity: ``class FdmSimple2dExtOUSolver : public LazyObject``.

    **Carve-out:** runtime ``value_at`` requires the multi-D
    backward FDM framework (deferred).
    """

    def __init__(
        self,
        process: ExtendedOrnsteinUhlenbeckProcess,
        r_ts: YieldTermStructure,
        mesher: FdmMesher,
        condition: FdmStepConditionComposite | None,
        calculator: InnerValueCalculator,
        maturity: float,
        time_steps: int,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        self._process: ExtendedOrnsteinUhlenbeckProcess = process
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

        # C++ parity: ``FdmSimple2dExtOUSolver::valueAt``.

        **Carve-out:** raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "FdmSimple2dExtOUSolver.value_at requires the 2-D backward FDM "
            "framework which is deferred to a follow-up Phase 11 cluster."
        )


__all__ = ["FdmSimple2dExtOUSolver"]
