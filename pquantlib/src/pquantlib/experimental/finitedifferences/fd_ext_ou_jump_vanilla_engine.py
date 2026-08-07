"""FdExtOUJumpVanillaEngine — FD vanilla engine under the Kluge spot model.

# C++ parity: ql/experimental/finitedifferences/fdextoujumpvanillaengine.{hpp,cpp}
# (v1.43).

Prices a :class:`VanillaOption` when the log spot price is the Kluge
process ``X(t) + Y(t)``: an extended Ornstein-Uhlenbeck factor plus an
exponential-jump factor, with a deterministic seasonal ``shape`` added on
top. The PDE is solved on a 2-D grid (OU factor x jump factor) by
:class:`FdmExtOUJumpSolver`.

The two axes are meshed differently on purpose. Direction 0 is
process-driven (:class:`FdmSimpleProcess1dMesher` over the embedded
extended-OU process), direction 1 is
:class:`ExponentialJump1dMesher`, which places nodes by the jump-size
distribution rather than by a diffusion's standard deviation.

The inner value is NOT ``payoff(exp(x))``: it is
``payoff(exp(shape(t) + x + y))`` via
:class:`FdmExtOUJumpModelInnerValue`, and the reported NPV is read off the
solver at the process's own initial values ``(x0, y0)``.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_model_inner_value import (
    FdmExtOUJumpModelInnerValue,
)
from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_model_inner_value import (
    Shape as InnerValueShape,
)
from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_solver import (
    FdmExtOUJumpSolver,
)
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.meshers.exponential_jump_1d_mesher import (
    ExponentialJump1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.fdm_simple_process_1d_mesher import (
    FdmSimpleProcess1dMesher,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

#: # C++ parity: ``typedef FdmExtOUJumpModelInnerValue::Shape Shape;``
#: (fdextoujumpvanillaengine.hpp:42) — a list of ``(t, value)`` knots. Aliased
#: rather than redefined so the two names are the SAME type, as the C++ typedef
#: makes them.
Shape = InnerValueShape


@final
class FdExtOUJumpVanillaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """FD vanilla engine over the extended-OU + jumps process.

    # C++ parity: ``class FdExtOUJumpVanillaEngine : public
    # GenericEngine<VanillaOption::arguments, VanillaOption::results>``.
    """

    def __init__(
        self,
        process: ExtOUWithJumpsProcess,
        r_ts: YieldTermStructure,
        t_grid: int = 50,
        x_grid: int = 200,
        y_grid: int = 50,
        shape: Shape | None = None,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: ExtOUWithJumpsProcess = process
        self._r_ts: YieldTermStructure = r_ts
        self._t_grid: int = int(t_grid)
        self._x_grid: int = int(x_grid)
        self._y_grid: int = int(y_grid)
        self._shape: Shape | None = shape
        # C++ default argument: FdmSchemeDesc::Hundsdorfer().
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )

    def calculate(self) -> None:
        """# C++ parity: ``FdExtOUJumpVanillaEngine::calculate`` (cpp:54-98)."""
        args = self._arguments
        results = self._results
        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None
        payoff: Payoff = args.payoff

        # 1. Mesher.
        maturity = self._r_ts.day_counter().year_fraction(
            self._r_ts.reference_date(), args.exercise.last_date()
        )
        ou_process = self._process.get_extended_ornstein_uhlenbeck_process()
        x_mesher = FdmSimpleProcess1dMesher(self._x_grid, ou_process, maturity)
        y_mesher = ExponentialJump1dMesher(
            self._y_grid,
            self._process.beta(),
            self._process.jump_intensity(),
            self._process.eta(),
        )
        mesher = FdmMesherComposite(x_mesher, y_mesher)

        # 2. Calculator.
        calculator = FdmExtOUJumpModelInnerValue(payoff, mesher, self._shape)

        # 3. Step conditions. C++ passes an empty DividendSchedule().
        conditions = FdmStepConditionComposite.vanilla_composite(
            [],
            args.exercise,
            mesher,
            calculator,
            self._r_ts.reference_date(),
            self._r_ts.day_counter(),
        )

        # 4. Boundary conditions: C++ builds an empty FdmBoundaryConditionSet.
        # 5. Solver.
        solver_desc = FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=calculator.inner_value,
            maturity=maturity,
            time_steps=self._t_grid,
            damping_steps=0,
        )
        solver = FdmExtOUJumpSolver(
            self._process, self._r_ts, solver_desc, self._scheme_desc
        )

        x0 = self._process.initial_values()
        results.value = solver.value_at(float(x0[0]), float(x0[1]))


__all__ = ["FdExtOUJumpVanillaEngine", "Shape"]
