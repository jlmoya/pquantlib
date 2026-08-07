"""FdKlugeExtOUSpreadEngine — FD engine for a power/gas spread option.

# C++ parity: ql/experimental/finitedifferences/fdklugeextouspreadengine.{hpp,cpp}
# (v1.43).

Prices a spark-spread option — the right to burn gas into power — on a 3-D
grid built from the :class:`KlugeExtOUProcess`:

* directions 0 and 1 are the Kluge power process (extended-OU factor +
  exponential-jump factor),
* direction 2 is the extended-OU gas factor.

Power price is ``powerShape(t) * exp(x + y)`` and gas price is
``gasShape(t) * exp(u)``; the payoff is a
:class:`~pquantlib.payoffs.BasketPayoff` evaluated on that pair. Both
prices are produced by a zero-strike CALL inner-value calculator — that is
how C++ turns a "payoff" calculator into a plain price surface — and
:class:`FdmSpreadPayoffInnerValue` feeds the pair into the basket payoff.

Note the two calculators are NOT the same class: gas uses
:class:`FdmExpExtOUInnerValueCalculator` on direction 2, power uses
:class:`FdmExtOUJumpModelInnerValue`, which sums directions 0 and 1.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.fdm_exp_ext_ou_inner_value_calculator import (
    FdmExpExtOUInnerValueCalculator,
)
from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_model_inner_value import (
    FdmExtOUJumpModelInnerValue,
)
from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_model_inner_value import (
    Shape as InnerValueShape,
)
from pquantlib.experimental.finitedifferences.fdm_kluge_ext_ou_solver import (
    FdmKlugeExtOUSolver,
)
from pquantlib.experimental.finitedifferences.fdm_spread_payoff_inner_value import (
    FdmSpreadPayoffInnerValue,
)
from pquantlib.experimental.processes.kluge_ext_ou_process import KlugeExtOUProcess
from pquantlib.instruments.basket_option import BasketPayoff
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
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

#: # C++ parity: ``typedef FdmExtOUJumpModelInnerValue::Shape GasShape;`` and
#: ``... PowerShape;`` (fdklugeextouspreadengine.hpp:45-46) — two names for the
#: same type, aliased rather than redefined for exactly that reason.
GasShape = InnerValueShape
PowerShape = InnerValueShape


@final
class FdKlugeExtOUSpreadEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """FD engine for a power/gas spread option under Kluge + extended OU.

    # C++ parity: ``class FdKlugeExtOUSpreadEngine : public
    # GenericEngine<VanillaOption::arguments, VanillaOption::results>``.
    """

    def __init__(
        self,
        kluge_ou_process: KlugeExtOUProcess,
        r_ts: YieldTermStructure,
        t_grid: int = 25,
        x_grid: int = 50,
        y_grid: int = 10,
        u_grid: int = 25,
        gas_shape: GasShape | None = None,
        power_shape: PowerShape | None = None,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: KlugeExtOUProcess = kluge_ou_process
        self._r_ts: YieldTermStructure = r_ts
        self._t_grid: int = int(t_grid)
        self._x_grid: int = int(x_grid)
        self._y_grid: int = int(y_grid)
        self._u_grid: int = int(u_grid)
        self._gas_shape: GasShape | None = gas_shape
        self._power_shape: PowerShape | None = power_shape
        # C++ default argument: FdmSchemeDesc::Hundsdorfer().
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )

    def calculate(self) -> None:
        """# C++ parity: ``FdKlugeExtOUSpreadEngine::calculate`` (cpp:54-123)."""
        args = self._arguments
        results = self._results
        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        # 1. Mesher.
        maturity = self._r_ts.day_counter().year_fraction(
            self._r_ts.reference_date(), args.exercise.last_date()
        )
        kluge_process = self._process.get_kluge_process()
        ou_process = kluge_process.get_extended_ornstein_uhlenbeck_process()

        x_mesher = FdmSimpleProcess1dMesher(self._x_grid, ou_process, maturity)
        y_mesher = ExponentialJump1dMesher(
            self._y_grid,
            kluge_process.beta(),
            kluge_process.jump_intensity(),
            kluge_process.eta(),
        )
        u_mesher = FdmSimpleProcess1dMesher(
            self._u_grid, self._process.get_ext_ou_process(), maturity
        )
        mesher = FdmMesherComposite(x_mesher, y_mesher, u_mesher)

        # 2. Calculator.
        qassert.require(
            isinstance(args.payoff, BasketPayoff), " basket payoff expected"
        )
        assert isinstance(args.payoff, BasketPayoff)
        basket_payoff: BasketPayoff = args.payoff

        # C++ prices each leg with a ZERO-STRIKE CALL, which turns the
        # payoff-shaped calculator into a plain price surface.
        zero_strike_call = PlainVanillaPayoff(OptionType.Call, 0.0)
        gas_price = FdmExpExtOUInnerValueCalculator(
            zero_strike_call, mesher, self._gas_shape, 2
        )
        power_price = FdmExtOUJumpModelInnerValue(
            zero_strike_call, mesher, self._power_shape
        )
        calculator = FdmSpreadPayoffInnerValue(
            basket_payoff, power_price, gas_price
        )

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
        solver = FdmKlugeExtOUSolver(
            self._process, self._r_ts, solver_desc, self._scheme_desc, 3
        )

        x0 = self._process.initial_values()
        results.value = solver.value_at(
            [float(x0[0]), float(x0[1]), float(x0[2])]
        )


__all__ = ["FdKlugeExtOUSpreadEngine", "GasShape", "PowerShape"]
