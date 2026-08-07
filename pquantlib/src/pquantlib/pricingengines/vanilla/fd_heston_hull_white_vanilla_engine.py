"""FdHestonHullWhiteVanillaEngine — 3-D FD Heston / Hull-White vanilla engine.

# C++ parity: ql/pricingengines/vanilla/fdhestonhullwhitevanillaengine.{hpp,cpp}
# (v1.43) — ``class FdHestonHullWhiteVanillaEngine : public
#  GenericModelEngine<HestonModel, VanillaOption::arguments,
#  VanillaOption::results>``.

Solves the three-dimensional (log-spot, variance, short-rate) PDE. Four points
that a port typically gets wrong and that are reproduced verbatim here:

* the variance axis uses the plain
  :class:`~pquantlib.methods.finitedifferences.meshers.fdm_heston_variance_mesher.FdmHestonVarianceMesher`,
  **not** the local-volatility subclass the 2-D Heston engine uses, and there
  is no leverage function or mixing factor anywhere in this engine;
* the equity mesher's scale factor is ``1.5`` here (the 2-D engine hard-codes
  ``2.0``), with the strike-concentrated critical point ``(strike, 0.1)``;
* delta and gamma are **bumped**, not read off the spline —
  ``deltaAt(spot, v0, 0, spot*0.01)`` — so they carry the solver's own
  central-difference error;
* ``controlVariate`` adds ``AnalyticHestonEngine(model, 164)`` minus a 2-D
  :class:`FdHestonVanillaEngine` priced on the *same* grid parameters, and the
  correction is applied to the cached per-strike results too.

``enableMultipleStrikesCaching`` calls ``update()`` here (the 2-D engine only
clears the cache), which additionally notifies the engine's observers.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_multi_strike_mesher import (
    FdmBlackScholesMultiStrikeMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_heston_variance_mesher import (
    FdmHestonVarianceMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.fdm_simple_process_1d_mesher import (
    FdmSimpleProcess1dMesher,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_heston_hull_white_solver import (
    FdmHestonHullWhiteSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
)
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.pricingengines.vanilla.analytic_heston_engine import AnalyticHestonEngine
from pquantlib.pricingengines.vanilla.fd_heston_vanilla_engine import (
    FdHestonVanillaEngine,
    process_helper,
)
from pquantlib.processes.hull_white_process import HullWhiteProcess
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess


class FdHestonHullWhiteVanillaEngine(
    GenericModelEngine[HestonModel, OptionArguments, OneAssetOptionResults]
):
    """Finite-differences Heston Hull-White vanilla option engine.

    # C++ parity: ``class FdHestonHullWhiteVanillaEngine``.
    """

    def __init__(
        self,
        model: HestonModel,
        hw_process: HullWhiteProcess,
        corr_equity_short_rate: float,
        dividends: Sequence[Dividend] | None = None,
        t_grid: int = 50,
        x_grid: int = 100,
        v_grid: int = 40,
        r_grid: int = 20,
        damping_steps: int = 0,
        control_variate: bool = True,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults(), model)
        self._hw_process: HullWhiteProcess = hw_process
        self._dividends: list[Dividend] = list(dividends) if dividends else []
        self._corr_equity_short_rate: float = corr_equity_short_rate
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._v_grid: int = v_grid
        self._r_grid: int = r_grid
        self._damping_steps: int = damping_steps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._control_variate: bool = control_variate

        self._strikes: list[float] = []
        self._cached_args2results: list[
            tuple[Exercise, PlainVanillaPayoff, OneAssetOptionResults]
        ] = []

    # --- multiple strikes caching ----------------------------------------

    def update(self) -> None:
        """# C++ parity: ``FdHestonHullWhiteVanillaEngine::update``."""
        self._cached_args2results.clear()
        super().update()

    def enable_multiple_strikes_caching(self, strikes: Sequence[float]) -> None:
        """# C++ parity: ``enableMultipleStrikesCaching`` — note this one calls
        # ``update()``, unlike the 2-D engine which only clears the cache.
        """
        self._strikes = list(strikes)
        self.update()

    def strikes(self) -> list[float]:
        """The cached-strike vector; empty unless caching was enabled."""
        return list(self._strikes)

    # --- pricing ----------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915 — one C++ function, kept whole
        """# C++ parity: ``FdHestonHullWhiteVanillaEngine::calculate``
        # (fdhestonhullwhitevanillaengine.cpp:75-230).
        """
        args = self._arguments
        results = self._results
        assert args.exercise is not None

        # 1. cache lookup for precalculated results
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

        # 2. Mesher
        heston_process = model.process()
        maturity = heston_process.time(args.exercise.last_date())

        # 2.1 The variance mesher
        t_grid_min = 5
        variance_mesher = FdmHestonVarianceMesher(
            self._v_grid, heston_process, maturity, max(t_grid_min, self._t_grid // 50)
        )

        # 2.2 The equity mesher
        payoff = args.payoff
        qassert.require(isinstance(payoff, StrikedTypePayoff), "wrong payoff type given")
        assert isinstance(payoff, StrikedTypePayoff)

        helper = process_helper(
            heston_process.s0(),
            heston_process.dividend_yield(),
            heston_process.risk_free_rate(),
            variance_mesher.vola_estimate(),
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
                1.5,
                (payoff.strike(), 0.1),
                self._dividends,
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

        # 2.3 The short rate mesher
        ou_process = OrnsteinUhlenbeckProcess(self._hw_process.a(), self._hw_process.sigma())
        short_rate_mesher = FdmSimpleProcess1dMesher(self._r_grid, ou_process, maturity)

        mesher = FdmMesherComposite(equity_mesher, variance_mesher, short_rate_mesher)

        # 3. Calculator
        calculator = FdmLogInnerValue(payoff, mesher, 0)

        # 4. Step conditions
        conditions = FdmStepConditionComposite.vanilla_composite(
            self._dividends,
            args.exercise,
            mesher,
            calculator,
            heston_process.risk_free_rate().reference_date(),
            heston_process.risk_free_rate().day_counter(),
        )

        # 5. Boundary conditions — none. 6. Solver.
        solver_desc = FdmSolverDesc(
            mesher,
            conditions,
            calculator.avg_inner_value,
            maturity,
            self._t_grid,
            self._damping_steps,
        )
        solver = FdmHestonHullWhiteSolver(
            heston_process,
            self._hw_process,
            self._corr_equity_short_rate,
            solver_desc,
            self._scheme_desc,
        )

        spot = heston_process.s0().value()
        v0 = heston_process.v0
        results.value = solver.value_at(spot, v0, 0.0)
        results.delta = solver.delta_at(spot, v0, 0.0, spot * 0.01)
        results.gamma = solver.gamma_at(spot, v0, 0.0, spot * 0.01)
        results.theta = solver.theta_at(spot, v0, 0.0)

        self._cached_args2results = []
        for strike in self._strikes:
            cached_results = OneAssetOptionResults()
            d = payoff.strike() / strike
            cached_results.value = solver.value_at(spot * d, v0, 0.0) / d
            cached_results.delta = solver.delta_at(spot * d, v0, 0.0, spot * d * 0.01)
            cached_results.gamma = solver.gamma_at(spot * d, v0, 0.0, spot * d * 0.01) * d
            cached_results.theta = solver.theta_at(spot * d, v0, 0.0) / d
            self._cached_args2results.append(
                (args.exercise, PlainVanillaPayoff(payoff.option_type(), strike), cached_results)
            )

        if self._control_variate:
            analytic_engine = AnalyticHestonEngine(model, 164)
            exercise = EuropeanExercise(args.exercise.last_date())

            option = VanillaOption(payoff, exercise)
            option.set_pricing_engine(analytic_engine)
            analytic_npv = option.npv()

            fd_engine = FdHestonVanillaEngine(
                model,
                None,
                None,
                self._t_grid,
                self._x_grid,
                self._v_grid,
                self._damping_steps,
                self._scheme_desc,
            )
            fd_engine.enable_multiple_strikes_caching(self._strikes)
            option.set_pricing_engine(fd_engine)

            fd_npv = option.npv()
            assert results.value is not None
            results.value += analytic_npv - fd_npv

            for i, strike in enumerate(self._strikes):
                control_variate_option = VanillaOption(
                    PlainVanillaPayoff(payoff.option_type(), strike), exercise
                )
                control_variate_option.set_pricing_engine(analytic_engine)
                analytic_npv = control_variate_option.npv()

                control_variate_option.set_pricing_engine(fd_engine)
                fd_npv = control_variate_option.npv()
                cached = self._cached_args2results[i][2]
                assert cached.value is not None
                cached.value += analytic_npv - fd_npv


__all__ = ["FdHestonHullWhiteVanillaEngine"]
