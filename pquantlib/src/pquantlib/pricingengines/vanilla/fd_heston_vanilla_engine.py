"""FdHestonVanillaEngine / MakeFdHestonVanillaEngine — FD Heston vanilla engine.

# C++ parity: ql/pricingengines/vanilla/fdhestonvanillaengine.{hpp,cpp} (v1.43)
# — ``class FdHestonVanillaEngine : public GenericModelEngine<HestonModel,
#    VanillaOption::arguments, VanillaOption::results>`` and
#    ``class MakeFdHestonVanillaEngine``.

Solves the two-dimensional Heston PDE on a (log-spot, variance) mesh and
reads value, delta, gamma and theta off the resulting bicubic surface.

Three details of the C++ engine are easy to lose in a port and are
reproduced here verbatim:

* ``getSolverDesc(Real)`` **ignores its argument**. ``calculate()`` calls it
  with ``1.5`` and :class:`FdBatesVanillaEngine` calls it with ``2.0``; both
  values are discarded and the equity mesher's scale factor is the hard-coded
  ``2.0`` (single strike) or ``1.5`` (multi strike).
* The equity mesher is built through
  ``FdmBlackScholesMesher::processHelper(s0, dividendYield, riskFreeRate,
  vol)``, whose own parameters are named ``(rTS, qTS)`` but which forwards
  them as ``GeneralizedBlackScholesProcess(s0, /*dividendTS=*/qTS,
  /*riskFreeTS=*/rTS, ...)``. Net effect: the helper process carries the
  **real risk-free curve in its dividend slot and vice versa**, so the
  mesher's forward walk drifts at ``q - r`` rather than ``r - q``. That is
  v1.43 behaviour, it moves the grid, and it is therefore reproduced rather
  than corrected — see :func:`process_helper`.
* ``enableMultipleStrikesCaching`` both switches the equity mesher to
  ``FdmBlackScholesMultiStrikeMesher`` *and* installs a per-strike result
  cache that is filled from the single solved surface by moneyness scaling
  (``value/d``, ``delta``, ``gamma*d``, ``theta/d`` with
  ``d = payoff.strike()/strikes[i]``). The cached entries are never
  re-solved.

``dividends`` and ``quanto_helper`` are threaded straight through to
``FdmBlackScholesMesher`` and, for the dividends, to
``FdmStepConditionComposite.vanilla_composite``, exactly as in C++.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, final

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend, dividend_vector
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_multi_strike_mesher import (
    FdmBlackScholesMultiStrikeMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_heston_variance_mesher import (
    FdmHestonLocalVolatilityVarianceMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_heston_solver import FdmHestonSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
)
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.quotes.quote import Quote


def process_helper(
    s0: Quote,
    r_ts: YieldTermStructure,
    q_ts: YieldTermStructure,
    vol: float,
) -> GeneralizedBlackScholesProcess:
    """Constant-vol BSM process used to drive the equity mesher.

    # C++ parity: ``FdmBlackScholesMesher::processHelper``
    # (fdmblackscholesmesher.cpp:133-149).

    The C++ body is

    .. code-block:: c++

        return ext::make_shared<GeneralizedBlackScholesProcess>(
            s0, qTS, rTS,
            Handle<BlackVolTermStructure>(new BlackConstantVol(
                rTS->referenceDate(), Calendar(), vol, rTS->dayCounter())));

    i.e. the parameter named ``qTS`` lands in the *dividend* slot and the one
    named ``rTS`` in the *risk-free* slot, and the ``BlackConstantVol``
    reference date and day counter come from ``rTS``. Every Heston-family
    call site passes ``(s0, process->dividendYield(), process->riskFreeRate(),
    vol)``, so the helper process ends up with ``dividendTS`` = the real
    risk-free curve and ``riskFreeTS`` = the real dividend curve. The
    argument order is preserved here so the quirk survives the port intact.
    """
    return GeneralizedBlackScholesProcess(
        x0=s0,
        dividend_ts=q_ts,
        risk_free_ts=r_ts,
        black_vol_ts=BlackConstantVol(
            reference_date=r_ts.reference_date(),
            calendar=NullCalendar(),
            volatility=vol,
            day_counter=r_ts.day_counter(),
        ),
    )


class FdHestonVanillaEngine(
    GenericModelEngine[HestonModel, OptionArguments, OneAssetOptionResults]
):
    """Finite-difference Heston vanilla option engine.

    # C++ parity: ``class FdHestonVanillaEngine``.

    C++ declares four constructors that differ only in whether
    ``DividendSchedule`` and/or ``FdmQuantoHelper`` are present; Python
    expresses all four as keyword defaults on one signature.
    """

    def __init__(
        self,
        model: HestonModel,
        dividends: Sequence[Dividend] | None = None,
        quanto_helper: FdmQuantoHelper | None = None,
        t_grid: int = 100,
        x_grid: int = 100,
        v_grid: int = 50,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
        leverage_fct: LocalVolTermStructure | None = None,
        mixing_factor: float = 1.0,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults(), model)
        self._dividends: list[Dividend] = list(dividends) if dividends else []
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._v_grid: int = v_grid
        self._damping_steps: int = damping_steps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._leverage_fct: LocalVolTermStructure | None = leverage_fct
        self._quanto_helper: FdmQuantoHelper | None = quanto_helper
        self._mixing_factor: float = mixing_factor

        self._strikes: list[float] = []
        #: (exercise, PlainVanillaPayoff, results) triples.
        # C++ parity: ``mutable std::vector<std::pair<VanillaOption::arguments,
        # VanillaOption::results> > cachedArgs2results_``.
        self._cached_args2results: list[tuple[Exercise, PlainVanillaPayoff, OneAssetOptionResults]] = []

    # --- inspectors (no C++ counterpart; used by the tests) ---------------

    def dividends(self) -> list[Dividend]:
        """The dividend schedule the engine was built with."""
        return list(self._dividends)

    def strikes(self) -> list[float]:
        """The cached-strike vector; empty unless caching was enabled."""
        return list(self._strikes)

    # --- multiple strikes caching ----------------------------------------

    def update(self) -> None:
        """Drop the per-strike cache, then propagate.

        # C++ parity: ``FdHestonVanillaEngine::update``
        # (fdhestonvanillaengine.cpp:231-236).
        """
        self._cached_args2results.clear()
        super().update()

    def enable_multiple_strikes_caching(self, strikes: Sequence[float]) -> None:
        """# C++ parity: ``FdHestonVanillaEngine::enableMultipleStrikesCaching``."""
        self._strikes = list(strikes)
        self._cached_args2results.clear()

    # --- solver description ----------------------------------------------

    def get_solver_desc(self, equity_scale_factor: float) -> FdmSolverDesc:
        """Build the mesher / calculator / conditions bundle.

        # C++ parity: ``FdmSolverDesc FdHestonVanillaEngine::getSolverDesc(Real)
        # const`` (fdhestonvanillaengine.cpp:103-173). The parameter is
        # unnamed in C++ and never read; it is kept here for signature parity
        # (``FdBatesVanillaEngine`` passes 2.0, ``calculate()`` passes 1.5, and
        # neither reaches the mesher).
        """
        del equity_scale_factor  # C++ parity: the argument is discarded.

        args = self._arguments
        model = self.model()
        qassert.require(model is not None, "no model specified")
        assert model is not None
        assert args.exercise is not None

        # 1. Mesher
        process = model.process()
        maturity = process.time(args.exercise.last_date())

        # 1.1 The variance mesher
        t_grid_min = 5
        t_avg_steps = max(t_grid_min, self._t_grid // 50)
        v_mesher = FdmHestonLocalVolatilityVarianceMesher(
            self._v_grid,
            process,
            self._leverage_fct,
            maturity,
            t_avg_steps,
            0.0001,
            self._mixing_factor,
        )
        avg_vola_estimate = v_mesher.vola_estimate()

        # 1.2 The equity mesher
        payoff = args.payoff
        qassert.require(isinstance(payoff, StrikedTypePayoff), "wrong payoff type given")
        assert isinstance(payoff, StrikedTypePayoff)

        helper = process_helper(
            process.s0(), process.dividend_yield(), process.risk_free_rate(), avg_vola_estimate
        )

        equity_mesher: Fdm1dMesher
        if not self._strikes:
            equity_mesher = FdmBlackScholesMesher(
                self._x_grid,
                helper,
                maturity,
                payoff.strike(),
                None,
                None,
                0.0001,
                2.0,
                (payoff.strike(), 0.1),
                self._dividends,
                self._quanto_helper,
            )
        else:
            qassert.require(
                not self._dividends,
                "multiple strikes engine does not work with discrete dividends",
            )
            equity_mesher = FdmBlackScholesMultiStrikeMesher(
                self._x_grid,
                helper,
                maturity,
                self._strikes,
                0.0001,
                1.5,
                (payoff.strike(), 0.075),
            )

        mesher = FdmMesherComposite(equity_mesher, v_mesher)

        # 2. Calculator
        calculator = FdmLogInnerValue(payoff, mesher, 0)

        # 3. Step conditions
        conditions = FdmStepConditionComposite.vanilla_composite(
            self._dividends,
            args.exercise,
            mesher,
            calculator,
            process.risk_free_rate().reference_date(),
            process.risk_free_rate().day_counter(),
        )

        # 4. Boundary conditions — none. 5. Solver description.
        return FdmSolverDesc(
            mesher,
            conditions,
            calculator.avg_inner_value,
            maturity,
            self._t_grid,
            self._damping_steps,
        )

    # --- pricing ----------------------------------------------------------

    def calculate(self) -> None:
        """Solve once, then fill the per-strike cache.

        # C++ parity: ``FdHestonVanillaEngine::calculate``
        # (fdhestonvanillaengine.cpp:175-229).
        """
        args = self._arguments
        results = self._results
        assert args.exercise is not None

        # cache lookup for precalculated results
        for cached_exercise, cached_payoff, cached_results in self._cached_args2results:
            if (
                cached_exercise.type() == args.exercise.type()
                and cached_exercise.dates() == args.exercise.dates()
                and isinstance(args.payoff, PlainVanillaPayoff)
                and args.payoff.strike() == cached_payoff.strike()
                and args.payoff.option_type() == cached_payoff.option_type()
            ):
                qassert.require(
                    not self._dividends,
                    "multiple strikes engine does not work with discrete dividends",
                )
                results.value = cached_results.value
                results.delta = cached_results.delta
                results.gamma = cached_results.gamma
                results.theta = cached_results.theta
                return

        model = self.model()
        qassert.require(model is not None, "no model specified")
        assert model is not None
        process = model.process()

        solver = FdmHestonSolver(
            process,
            self.get_solver_desc(1.5),
            self._scheme_desc,
            self._quanto_helper,
            self._leverage_fct,
            self._mixing_factor,
        )

        v0 = process.v0
        spot = process.s0().value()

        results.value = solver.value_at(spot, v0)
        results.delta = solver.delta_at(spot, v0)
        results.gamma = solver.gamma_at(spot, v0)
        results.theta = solver.theta_at(spot, v0)

        payoff = args.payoff
        assert isinstance(payoff, StrikedTypePayoff)
        self._cached_args2results = []
        for strike in self._strikes:
            cached_results = OneAssetOptionResults()
            d = payoff.strike() / strike
            cached_results.value = solver.value_at(spot * d, v0) / d
            cached_results.delta = solver.delta_at(spot * d, v0)
            cached_results.gamma = solver.gamma_at(spot * d, v0) * d
            cached_results.theta = solver.theta_at(spot * d, v0) / d
            self._cached_args2results.append(
                (
                    args.exercise,
                    PlainVanillaPayoff(payoff.option_type(), strike),
                    cached_results,
                )
            )


@final
class MakeFdHestonVanillaEngine:
    """Fluent builder for :class:`FdHestonVanillaEngine`.

    # C++ parity: ``class MakeFdHestonVanillaEngine``
    # (fdhestonvanillaengine.{hpp:116-148,cpp:245-310}).

    Every ``with*`` method assigns and returns ``*this``; there is **no**
    validation anywhere in the C++ builder and none is invented here. The
    defaults are ``tGrid = 100``, ``xGrid = 100``, ``vGrid = 50``,
    ``dampingSteps = 0``, ``schemeDesc = Hundsdorfer()``, no leverage
    function, no quanto helper, no dividends.
    """

    def __init__(self, heston_model: HestonModel) -> None:
        self._heston_model: HestonModel = heston_model
        self._dividends: list[Dividend] = []
        self._t_grid: int = 100
        self._x_grid: int = 100
        self._v_grid: int = 50
        self._damping_steps: int = 0
        self._scheme_desc: FdmSchemeDesc = FdmSchemeDesc.hundsdorfer()
        self._leverage_fct: LocalVolTermStructure | None = None
        self._quanto_helper: FdmQuantoHelper | None = None

    def with_quanto_helper(self, quanto_helper: FdmQuantoHelper) -> MakeFdHestonVanillaEngine:
        """# C++ parity: ``MakeFdHestonVanillaEngine::withQuantoHelper``."""
        self._quanto_helper = quanto_helper
        return self

    def with_t_grid(self, t_grid: int) -> MakeFdHestonVanillaEngine:
        """# C++ parity: ``MakeFdHestonVanillaEngine::withTGrid``."""
        self._t_grid = t_grid
        return self

    def with_x_grid(self, x_grid: int) -> MakeFdHestonVanillaEngine:
        """# C++ parity: ``MakeFdHestonVanillaEngine::withXGrid``."""
        self._x_grid = x_grid
        return self

    def with_v_grid(self, v_grid: int) -> MakeFdHestonVanillaEngine:
        """# C++ parity: ``MakeFdHestonVanillaEngine::withVGrid``."""
        self._v_grid = v_grid
        return self

    def with_damping_steps(self, damping_steps: int) -> MakeFdHestonVanillaEngine:
        """# C++ parity: ``MakeFdHestonVanillaEngine::withDampingSteps``."""
        self._damping_steps = damping_steps
        return self

    def with_fdm_scheme_desc(self, scheme_desc: FdmSchemeDesc) -> MakeFdHestonVanillaEngine:
        """# C++ parity: ``MakeFdHestonVanillaEngine::withFdmSchemeDesc``."""
        self._scheme_desc = scheme_desc
        return self

    def with_leverage_function(
        self, leverage_fct: LocalVolTermStructure
    ) -> MakeFdHestonVanillaEngine:
        """# C++ parity: ``MakeFdHestonVanillaEngine::withLeverageFunction``."""
        self._leverage_fct = leverage_fct
        return self

    def with_cash_dividends(
        self, dividend_dates: Sequence[Date], dividend_amounts: Sequence[float]
    ) -> MakeFdHestonVanillaEngine:
        """# C++ parity: ``MakeFdHestonVanillaEngine::withCashDividends`` —
        # ``dividends_ = DividendVector(dividendDates, dividendAmounts)``.
        """
        self._dividends = dividend_vector(list(dividend_dates), list(dividend_amounts))
        return self

    def engine(self) -> FdHestonVanillaEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``.

        Note that C++ does **not** forward ``mixingFactor`` here, so the built
        engine always uses the default ``1.0``.
        """
        return FdHestonVanillaEngine(
            self._heston_model,
            self._dividends,
            self._quanto_helper,
            self._t_grid,
            self._x_grid,
            self._v_grid,
            self._damping_steps,
            self._scheme_desc,
            self._leverage_fct,
        )


__all__ = [
    "FdHestonVanillaEngine",
    "MakeFdHestonVanillaEngine",
    "process_helper",
]
