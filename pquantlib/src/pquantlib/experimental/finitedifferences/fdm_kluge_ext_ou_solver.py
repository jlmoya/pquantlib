"""FdmKlugeExtOUSolver — solver wrapper for the FdmKlugeExtOUOp.

# C++ parity: ql/experimental/finitedifferences/fdmklugeextousolver.hpp
# (v1.42.1).

The C++ class is a thin lazy-object wrapper around
``FdmNdimSolver<N>`` (default N=3) for the 3-D Kluge + ExtOU op.

**Carve-out (Phase 11 W5-A):** the multi-D backward FDM framework
is deferred (see ``fdm_ext_ou_jump_solver.py`` for rationale). The
class accepts the same parameters as the C++ class but
``value_at(x)`` raises ``NotImplementedError`` for now.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import final

from pquantlib import qassert
from pquantlib.experimental.processes.kluge_ext_ou_process import KlugeExtOUProcess
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
class FdmKlugeExtOUSolver:
    """Backward FD solver for the correlated Kluge + ExtOU op.

    # C++ parity: ``template <Size N=3> class FdmKlugeExtOUSolver``.

    The Python port collapses the N=3 default and exposes the same
    constructor signature. ``N`` is restricted to ``>= 3`` (matching
    the C++ ``BOOST_STATIC_ASSERT``).

    **Carve-out:** runtime ``value_at`` requires the multi-D
    backward FDM framework (deferred).
    """

    def __init__(
        self,
        kluge_ext_ou_process: KlugeExtOUProcess,
        r_ts: YieldTermStructure,
        mesher: FdmMesher,
        condition: FdmStepConditionComposite | None,
        calculator: InnerValueCalculator,
        maturity: float,
        time_steps: int,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
        n: int = 3,
    ) -> None:
        qassert.require(n >= 3, f"KlugeExtOU solver requires N >= 3, got {n}")
        self._process: KlugeExtOUProcess = kluge_ext_ou_process
        self._r_ts: YieldTermStructure = r_ts
        self._mesher: FdmMesher = mesher
        self._condition: FdmStepConditionComposite | None = condition
        self._calculator: InnerValueCalculator = calculator
        self._maturity: float = maturity
        self._time_steps: int = time_steps
        self._damping_steps: int = damping_steps
        self._scheme_desc: FdmSchemeDesc | None = scheme_desc
        self._n: int = n

    def value_at(self, x: Sequence[float]) -> float:
        """Interpolate the rolled-back value at the given multi-D state.

        # C++ parity: ``FdmKlugeExtOUSolver::valueAt(const std::vector<Real>&)``.

        **Carve-out:** raises ``NotImplementedError`` until the
        multi-D backward FDM framework is ported.
        """
        raise NotImplementedError(
            "FdmKlugeExtOUSolver.value_at requires the multi-D backward FDM "
            "framework (FdmNdimSolver + multi-direction Hundsdorfer scheme) "
            "which is deferred to a follow-up Phase 11 cluster."
        )


__all__ = ["FdmKlugeExtOUSolver"]
