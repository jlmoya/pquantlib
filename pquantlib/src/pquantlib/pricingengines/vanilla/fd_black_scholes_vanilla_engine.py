"""FdBlackScholesVanillaEngine — finite-difference Black-Scholes vanilla engine.

# C++ parity: ql/pricingengines/vanilla/fdblackscholesvanillaengine.{hpp,cpp}
# (v1.43) — ``class FdBlackScholesVanillaEngine : public VanillaOption::engine``
# and ``class MakeFdBlackScholesVanillaEngine``.

``calculate()`` builds an ``FdmSolverDesc`` (mesher / calculator /
step-condition composite / boundaries) and hands it to
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_black_scholes_solver.FdmBlackScholesSolver`,
then reads value, delta, gamma and theta straight off the solver — one for
one with the C++ body.

C++ declares four constructors, distinguished only by whether a
``DividendSchedule`` and/or an ``FdmQuantoHelper`` precede the grid sizes.
Python cannot overload, so the two are keyword parameters ``dividends`` and
``quanto_helper`` placed where C++ puts them; the C++ positional order of
everything else (``t_grid``, ``x_grid``, ``damping_steps``, ``scheme_desc``,
``local_vol``, ``illegal_local_vol_overwrite``, ``cash_dividend_model``) is
preserved, so ``FdBlackScholesVanillaEngine(process, 100, 100, 0)`` reads the
same in both languages.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend, FixedDividend, dividend_vector
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import NULL_REAL
from pquantlib.methods.finitedifferences.solvers.fdm_black_scholes_solver import (
    FdmBlackScholesSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.escrowed_dividend_adjustment import (
    EscrowedDividendAdjustment,
)
from pquantlib.methods.finitedifferences.utilities.fdm_escrowed_log_inner_value_calculator import (
    FdmEscrowedLogInnerValueCalculator,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmInnerValueCalculator,
    FdmLogInnerValue,
)
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.cash_dividend_european_engine import (
    CashDividendModel,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.date import Date


@final
class FdBlackScholesVanillaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Finite-difference Black-Scholes vanilla option engine.

    # C++ parity: ``class FdBlackScholesVanillaEngine``.
    """

    #: # C++ parity: ``enum CashDividendModel { Spot = ..., Escrowed = ... }``
    #: aliased to ``CashDividendEuropeanEngine``'s own enum, exactly as C++ does.
    Spot = CashDividendModel.Spot
    Escrowed = CashDividendModel.Escrowed

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        t_grid: int = 100,
        x_grid: int = 100,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
        local_vol: bool = False,
        illegal_local_vol_overwrite: float = -NULL_REAL,
        cash_dividend_model: CashDividendModel = CashDividendModel.Spot,
        *,
        dividends: Sequence[Dividend] = (),
        quanto_helper: FdmQuantoHelper | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._dividends: list[Dividend] = list(dividends)
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._damping_steps: int = damping_steps
        # C++ parity: every one of the four constructors defaults
        # ``schemeDesc`` to ``FdmSchemeDesc::Douglas()``
        # (fdblackscholesvanillaengine.hpp:57, 67, 79, 91).
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )
        self._local_vol: bool = local_vol
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        self._quanto_helper: FdmQuantoHelper | None = quanto_helper
        self._cash_dividend_model: CashDividendModel = cash_dividend_model

        # C++ parity: ``registerWith(process_)`` / ``registerWith(quantoHelper_)``.
        process.register_with(self)
        if quanto_helper is not None:
            quanto_helper.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``FdBlackScholesVanillaEngine::calculate`` (cpp:109-215)."""
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        exercise: Exercise = args.exercise

        # 0. Cash dividend model
        exercise_date = exercise.last_date()
        maturity = self._process.time(exercise_date)
        settlement_date = self._process.risk_free_rate().reference_date()

        spot_adjustment = 0.0
        dividend_schedule: list[Dividend] = []
        escrowed_div_adj: EscrowedDividendAdjustment | None = None

        if self._cash_dividend_model == CashDividendModel.Spot:
            dividend_schedule = list(self._dividends)
        elif self._cash_dividend_model == CashDividendModel.Escrowed:
            if exercise.type() != Exercise.Type.European:
                # add dividend dates as stopping times
                dividend_schedule = [FixedDividend(0.0, cf.date()) for cf in self._dividends]

            qassert.require(
                self._quanto_helper is None,
                "Escrowed dividend model is not supported for Quanto-Options",
            )

            escrowed_div_adj = EscrowedDividendAdjustment(
                self._dividends,
                self._process.risk_free_rate(),
                self._process.dividend_yield(),
                self._process.time,
                maturity,
            )
            spot_adjustment = escrowed_div_adj.dividend_adjustment(
                self._process.time(settlement_date)
            )
            qassert.require(
                self._process.x0() + spot_adjustment > 0.0,
                "spot minus dividends becomes negative",
            )
        else:
            qassert.fail("unknwon cash dividend model")

        # 1. Mesher
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff),
            "non-striked payoff given",
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        equity_mesher = FdmBlackScholesMesher(
            self._x_grid,
            self._process,
            maturity,
            payoff.strike(),
            None,
            None,
            0.0001,
            1.5,
            (payoff.strike(), 0.1),
            dividend_schedule,
            self._quanto_helper,
            spot_adjustment,
        )
        mesher = FdmMesherComposite(equity_mesher)

        # 2. Calculator
        calculator = FdmLogInnerValue(payoff, mesher, 0)

        early_exercise_calculator: FdmInnerValueCalculator
        if self._cash_dividend_model == CashDividendModel.Spot:
            early_exercise_calculator = calculator
        elif self._cash_dividend_model == CashDividendModel.Escrowed:
            assert escrowed_div_adj is not None
            early_exercise_calculator = FdmEscrowedLogInnerValueCalculator(
                escrowed_div_adj, payoff, mesher, 0
            )
        else:
            qassert.fail("unknwon cash dividend model")

        # 3. Step conditions
        conditions = FdmStepConditionComposite.vanilla_composite(
            dividend_schedule,
            exercise,
            mesher,
            early_exercise_calculator,
            self._process.risk_free_rate().reference_date(),
            self._process.risk_free_rate().day_counter(),
        )

        # 4. Boundary conditions — C++ passes a default-constructed
        # ``FdmBoundaryConditionSet``; ``FdmSolverDesc.bc_set`` defaults to it.
        # 5. Solver
        solver_desc = FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=calculator.avg_inner_value,
            maturity=maturity,
            time_steps=self._t_grid,
            damping_steps=self._damping_steps,
        )
        solver = FdmBlackScholesSolver(
            self._process,
            payoff.strike(),
            solver_desc,
            self._scheme_desc,
            self._local_vol,
            self._illegal_local_vol_overwrite,
            self._quanto_helper,
        )

        spot = self._process.x0() + spot_adjustment

        results.value = solver.value_at(spot)
        results.delta = solver.delta_at(spot)
        results.gamma = solver.gamma_at(spot)
        results.theta = solver.theta_at(spot)


@final
class MakeFdBlackScholesVanillaEngine:
    """Fluent builder for :class:`FdBlackScholesVanillaEngine`.

    # C++ parity: ``class MakeFdBlackScholesVanillaEngine``
    # (fdblackscholesvanillaengine.hpp:110-147, cpp:217-294).

    C++ ends with ``operator ext::shared_ptr<PricingEngine>()``; Python has no
    implicit conversion operator, so the terminal step is :meth:`engine`
    (also reachable by calling the builder, ``Make...(p)()``).
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        self._process: GeneralizedBlackScholesProcess = process
        self._dividends: list[Dividend] = []
        # C++ parity: the in-class member initialisers.
        self._t_grid: int = 100
        self._x_grid: int = 100
        self._damping_steps: int = 0
        self._scheme_desc: FdmSchemeDesc = FdmSchemeDesc.douglas()
        self._local_vol: bool = False
        self._illegal_local_vol_overwrite: float = -NULL_REAL
        self._quanto_helper: FdmQuantoHelper | None = None
        self._cash_dividend_model: CashDividendModel = CashDividendModel.Spot

    def with_quanto_helper(self, quanto_helper: FdmQuantoHelper) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withQuantoHelper``."""
        self._quanto_helper = quanto_helper
        return self

    def with_t_grid(self, t_grid: int) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withTGrid``."""
        self._t_grid = t_grid
        return self

    def with_x_grid(self, x_grid: int) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withXGrid``."""
        self._x_grid = x_grid
        return self

    def with_damping_steps(self, damping_steps: int) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withDampingSteps``."""
        self._damping_steps = damping_steps
        return self

    def with_fdm_scheme_desc(self, scheme_desc: FdmSchemeDesc) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withFdmSchemeDesc``."""
        self._scheme_desc = scheme_desc
        return self

    def with_local_vol(self, local_vol: bool) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withLocalVol``."""
        self._local_vol = local_vol
        return self

    def with_illegal_local_vol_overwrite(
        self, illegal_local_vol_overwrite: float
    ) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withIllegalLocalVolOverwrite``."""
        self._illegal_local_vol_overwrite = illegal_local_vol_overwrite
        return self

    def with_cash_dividends(
        self, dividend_dates: Sequence[Date], dividend_amounts: Sequence[float]
    ) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withCashDividends`` — ``DividendVector(dates, amounts)``."""
        self._dividends = list(dividend_vector(list(dividend_dates), list(dividend_amounts)))
        return self

    def with_cash_dividend_model(
        self, cash_dividend_model: CashDividendModel
    ) -> MakeFdBlackScholesVanillaEngine:
        """# C++ parity: ``withCashDividendModel``."""
        self._cash_dividend_model = cash_dividend_model
        return self

    def engine(self) -> PricingEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``."""
        return FdBlackScholesVanillaEngine(
            self._process,
            self._t_grid,
            self._x_grid,
            self._damping_steps,
            self._scheme_desc,
            self._local_vol,
            self._illegal_local_vol_overwrite,
            self._cash_dividend_model,
            dividends=self._dividends,
            quanto_helper=self._quanto_helper,
        )

    def __call__(self) -> PricingEngine:
        """Alias for :meth:`engine` — the C++ conversion operator's call site."""
        return self.engine()


__all__ = ["FdBlackScholesVanillaEngine", "MakeFdBlackScholesVanillaEngine"]
