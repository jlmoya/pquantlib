"""FdCIRVanillaEngine — finite-difference vanilla engine with CIR stochastic rates.

# C++ parity: ql/pricingengines/vanilla/fdcirvanillaengine.{hpp,cpp}
# (v1.43) — ``class FdCIRVanillaEngine : public VanillaOption::engine`` and
# ``class MakeFdCIRVanillaEngine``.

Two state variables: log-spot on an ``FdmBlackScholesMesher`` and the CIR
short rate on an ``FdmSimpleProcess1dMesher``, coupled with correlation
``rho`` through
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_cir_solver.FdmCIRSolver`.
The engine's default scheme is ``ModifiedHundsdorfer`` — a 2-D ADI splitting —
not ``Douglas``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend, dividend_vector
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.fdm_simple_process_1d_mesher import (
    FdmSimpleProcess1dMesher,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_cir_solver import FdmCIRSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
)
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.date import Date


@final
class FdCIRVanillaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Finite-difference vanilla engine under a CIR short-rate process.

    # C++ parity: ``class FdCIRVanillaEngine``.

    C++ has two constructors differing only by a ``DividendSchedule`` in third
    position; Python folds that into the keyword-only ``dividends``. Note that
    C++ gives **no defaults** to ``tGrid``/``xGrid``/``rGrid``/
    ``dampingSteps``/``rho``, so neither does this port.
    """

    def __init__(
        self,
        cir_process: CoxIngersollRossProcess,
        bs_process: GeneralizedBlackScholesProcess,
        t_grid: int,
        x_grid: int,
        r_grid: int,
        damping_steps: int,
        rho: float,
        scheme_desc: FdmSchemeDesc | None = None,
        quanto_helper: FdmQuantoHelper | None = None,
        *,
        dividends: Sequence[Dividend] = (),
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._bs_process: GeneralizedBlackScholesProcess = bs_process
        self._cir_process: CoxIngersollRossProcess = cir_process
        self._quanto_helper: FdmQuantoHelper | None = quanto_helper
        self._dividends: list[Dividend] = list(dividends)
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._r_grid: int = r_grid
        self._damping_steps: int = damping_steps
        self._rho: float = rho
        # C++ parity: ``schemeDesc = FdmSchemeDesc::ModifiedHundsdorfer()``.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.modified_hundsdorfer()
        )
        # C++ parity: neither constructor calls ``registerWith``.

    def get_solver_desc(self, equity_scale_factor: float) -> FdmSolverDesc:
        """Build the 2-D (log-spot, short-rate) solver description.

        # C++ parity: ``FdCIRVanillaEngine::getSolverDesc(Real) const``
        # (cpp:66-107) — the ``Real`` parameter is unnamed and unused in C++.
        """
        del equity_scale_factor

        args = self._arguments
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None

        maturity = self._bs_process.time(args.exercise.last_date())

        # The short rate mesher
        short_rate_mesher = FdmSimpleProcess1dMesher(
            self._r_grid, self._cir_process, maturity, self._t_grid
        )

        # The equity mesher
        equity_mesher = FdmBlackScholesMesher(
            self._x_grid,
            self._bs_process,
            maturity,
            payoff.strike(),
            None,
            None,
            0.0001,
            1.5,
            (payoff.strike(), 0.1),
            self._dividends,
            self._quanto_helper,
            0.0,
        )

        mesher = FdmMesherComposite(equity_mesher, short_rate_mesher)

        # Calculator
        calculator = FdmLogInnerValue(payoff, mesher, 0)

        # Step conditions
        conditions = FdmStepConditionComposite.vanilla_composite(
            self._dividends,
            args.exercise,
            mesher,
            calculator,
            self._bs_process.risk_free_rate().reference_date(),
            self._bs_process.risk_free_rate().day_counter(),
        )

        # Boundary conditions: C++ passes a default-constructed set.
        return FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=calculator.avg_inner_value,
            maturity=maturity,
            time_steps=self._t_grid,
            damping_steps=self._damping_steps,
        )

    def calculate(self) -> None:
        """# C++ parity: ``FdCIRVanillaEngine::calculate`` (cpp:109-124)."""
        args = self._arguments
        results = self._results

        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        solver = FdmCIRSolver(
            self._cir_process,
            self._bs_process,
            self.get_solver_desc(1.5),
            self._scheme_desc,
            self._rho,
            payoff.strike(),
        )

        r0 = self._cir_process.x0()
        spot = self._bs_process.x0()

        results.value = solver.value_at(spot, r0)
        results.delta = solver.delta_at(spot, r0)
        results.gamma = solver.gamma_at(spot, r0)
        results.theta = solver.theta_at(spot, r0)


@final
class MakeFdCIRVanillaEngine:
    """Fluent builder for :class:`FdCIRVanillaEngine`.

    # C++ parity: ``class MakeFdCIRVanillaEngine``
    # (fdcirvanillaengine.hpp:81-115, cpp:126-194).

    C++ ends with ``operator ext::shared_ptr<PricingEngine>()``; the Python
    terminal step is :meth:`engine` (also reachable by calling the builder).
    """

    def __init__(
        self,
        cir_process: CoxIngersollRossProcess,
        bs_process: GeneralizedBlackScholesProcess,
        rho: float,
    ) -> None:
        self._cir_process: CoxIngersollRossProcess = cir_process
        self._bs_process: GeneralizedBlackScholesProcess = bs_process
        self._dividends: list[Dividend] = []
        self._rho: float = rho
        # C++ parity: the in-class member initialisers.
        self._t_grid: int = 10
        self._x_grid: int = 100
        self._r_grid: int = 100
        self._damping_steps: int = 0
        self._scheme_desc: FdmSchemeDesc = FdmSchemeDesc.modified_hundsdorfer()
        self._quanto_helper: FdmQuantoHelper | None = None

    def with_quanto_helper(self, quanto_helper: FdmQuantoHelper) -> MakeFdCIRVanillaEngine:
        """# C++ parity: ``withQuantoHelper``."""
        self._quanto_helper = quanto_helper
        return self

    def with_t_grid(self, t_grid: int) -> MakeFdCIRVanillaEngine:
        """# C++ parity: ``withTGrid``."""
        self._t_grid = t_grid
        return self

    def with_x_grid(self, x_grid: int) -> MakeFdCIRVanillaEngine:
        """# C++ parity: ``withXGrid``."""
        self._x_grid = x_grid
        return self

    def with_r_grid(self, r_grid: int) -> MakeFdCIRVanillaEngine:
        """# C++ parity: ``withRGrid``."""
        self._r_grid = r_grid
        return self

    def with_damping_steps(self, damping_steps: int) -> MakeFdCIRVanillaEngine:
        """# C++ parity: ``withDampingSteps``."""
        self._damping_steps = damping_steps
        return self

    def with_fdm_scheme_desc(self, scheme_desc: FdmSchemeDesc) -> MakeFdCIRVanillaEngine:
        """# C++ parity: ``withFdmSchemeDesc``."""
        self._scheme_desc = scheme_desc
        return self

    def with_cash_dividends(
        self, dividend_dates: Sequence[Date], dividend_amounts: Sequence[float]
    ) -> MakeFdCIRVanillaEngine:
        """# C++ parity: ``withCashDividends`` — ``DividendVector(dates, amounts)``."""
        self._dividends = list(dividend_vector(list(dividend_dates), list(dividend_amounts)))
        return self

    def engine(self) -> PricingEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``."""
        return FdCIRVanillaEngine(
            self._cir_process,
            self._bs_process,
            self._t_grid,
            self._x_grid,
            self._r_grid,
            self._damping_steps,
            self._rho,
            self._scheme_desc,
            self._quanto_helper,
            dividends=self._dividends,
        )

    def __call__(self) -> PricingEngine:
        """Alias for :meth:`engine` — the C++ conversion operator's call site."""
        return self.engine()


__all__ = ["FdCIRVanillaEngine", "MakeFdCIRVanillaEngine"]
