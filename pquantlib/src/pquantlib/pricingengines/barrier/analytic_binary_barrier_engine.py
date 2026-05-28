# ruff: noqa: SIM108
# Reason: the 8-way (BarrierType x OptionType x strike-vs-barrier)
# branch table mirrors C++ ``analyticbinarybarrierengine.cpp`` switch
# cases line-by-line. The inline ``# B3``, ``# B1-B2+B4`` etc. comments
# anchor cross-reference to the Reiner-Rubinstein 4-term decomposition
# and would be lost under a ternary collapse.
"""AnalyticBinaryBarrierEngine — one-touch / no-touch binary barrier.

# C++ parity:
# ql/pricingengines/barrier/analyticbinarybarrierengine.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for American-exercise binary (cash-or-nothing or
asset-or-nothing) barrier options under Black-Scholes dynamics.
Reference: Haug "The Complete Guide to Option Pricing Formulas" 2nd ed,
pp. 176 ff. (Reiner-Rubinstein 1991).

The engine handles three branches:
  1. Knock-out KO degenerate: spot is on the KO side of the barrier;
     value = 0.
  2. Knock-in KI degenerate: spot is on the KI side of the barrier;
     value reduces to a European binary (delegate to
     ``AnalyticEuropeanEngine``).
  3. General case: 4-term Reiner-Rubinstein-style closed-form
     determined by (BarrierType, OptionType, strike vs barrier).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise, Exercise
from pquantlib.instruments.barrier_option import (
    BarrierOptionArguments,
    BarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
    OptionType,
    StrikedTypePayoff,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class AnalyticBinaryBarrierEngine(
    GenericEngine[BarrierOptionArguments, OneAssetOptionResults]
):
    """Reiner-Rubinstein binary-barrier closed-form engine.

    # C++ parity: ``AnalyticBinaryBarrierEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(BarrierOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:
        """Three-branch binary-barrier closed form.

        # C++ parity:
        # ``AnalyticBinaryBarrierEngine::calculate`` +
        # ``AnalyticBinaryBarrierEngine_helper::payoffAtExpiry``.
        """
        args = self._arguments
        results = self._results

        # American exercise with payoff-at-expiry (one-touch / no-touch).
        ex = args.exercise
        qassert.require(ex is not None, "no exercise given")
        assert ex is not None
        qassert.require(
            isinstance(ex, AmericanExercise), "non-American exercise given"
        )
        assert isinstance(ex, AmericanExercise)
        qassert.require(ex.payoff_at_expiry(), "payoff must be at expiry")
        qassert.require(
            ex.dates()[0]
            <= self._process.black_volatility().reference_date(),
            "American option with window exercise not handled yet",
        )

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(payoff, StrikedTypePayoff)

        spot = self._process.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying given")

        variance = self._process.black_volatility().black_variance(
            ex.last_date(), payoff.strike(), extrapolate=True
        )
        assert args.barrier is not None
        barrier = args.barrier
        qassert.require(barrier > 0.0, "positive barrier value required")
        qassert.require(args.barrier_type is not None, "no barrier type given")
        assert args.barrier_type is not None
        bt: BarrierType = args.barrier_type

        # ---------- KO degenerate cases (already knocked out) ------------
        if (bt == BarrierType.DownOut and spot <= barrier) or (
            bt == BarrierType.UpOut and spot >= barrier
        ):
            results.value = 0.0
            results.delta = 0.0
            results.gamma = 0.0
            results.vega = 0.0
            results.theta = 0.0
            results.rho = 0.0
            results.dividend_rho = 0.0
            return

        # ---------- KI degenerate cases (already knocked in) -------------
        if (bt == BarrierType.DownIn and spot <= barrier) or (
            bt == BarrierType.UpIn and spot >= barrier
        ):
            # Knocked in - reduces to a digital European.
            eu_exercise = EuropeanExercise(ex.last_date())
            opt = VanillaOption(payoff, eu_exercise)
            opt.set_pricing_engine(AnalyticEuropeanEngine(self._process))
            results.value = opt.npv()
            results.delta = opt.delta()
            results.gamma = opt.gamma()
            results.vega = opt.vega()
            try:
                results.theta = opt.theta()
            except LibraryException:
                # theta may not be supplied for some special cases.
                results.theta = None
            results.rho = opt.rho()
            results.dividend_rho = opt.dividend_rho()
            return

        # ---------- General case: 4-term Reiner-Rubinstein closed form ---
        risk_free_discount = self._process.risk_free_rate().discount(
            ex.last_date()
        )
        results.value = self._payoff_at_expiry(
            payoff, spot, variance, risk_free_discount, barrier, bt, ex
        )

    def _payoff_at_expiry(  # noqa: PLR0915
        self,
        payoff: StrikedTypePayoff,
        spot: float,
        variance: float,
        discount: float,
        barrier: float,
        bt: BarrierType,
        exercise: Exercise,
    ) -> float:
        """Closed form for the general (not-yet-triggered) case.

        # C++ parity: ``AnalyticBinaryBarrierEngine_helper::payoffAtExpiry``.
        """
        dividend_discount = self._process.dividend_yield().discount(
            exercise.last_date()
        )
        qassert.require(spot > 0.0, "positive spot value required")
        qassert.require(discount > 0.0, "positive discount required")
        qassert.require(
            dividend_discount > 0.0, "positive dividend discount required"
        )
        qassert.require(variance >= 0.0, "negative variance not allowed")

        opt_type = payoff.option_type()
        strike = payoff.strike()
        qassert.require(barrier > 0.0, "positive barrier value required")

        std_dev = math.sqrt(variance)
        mu = (
            math.log(dividend_discount / discount) / variance - 0.5
            if variance > 0.0
            else -0.5
        )
        cash_payoff = 0.0

        # Branch on payoff subtype.
        coo: CashOrNothingPayoff | None = (
            payoff if isinstance(payoff, CashOrNothingPayoff) else None
        )
        if coo is not None:
            cash_payoff = coo.cash_payoff()

        aoo: AssetOrNothingPayoff | None = (
            payoff if isinstance(payoff, AssetOrNothingPayoff) else None
        )
        if aoo is not None:
            mu += 1.0
            cash_payoff = spot * dividend_discount / discount

        log_s_x = math.log(spot / strike)
        log_s_h = math.log(spot / barrier)
        log_h_s = math.log(barrier / spot)
        log_h2_sx = math.log(barrier * barrier / (spot * strike))
        h_s_2mu = (barrier / spot) ** (2.0 * mu)

        eta = 1.0 if bt in (BarrierType.DownIn, BarrierType.DownOut) else -1.0
        phi = 1.0 if opt_type == OptionType.Call else -1.0
        _ = eta, phi  # both used below via x1/x2/y1/y2

        if variance >= QL_EPSILON:
            x1 = phi * (log_s_x / std_dev + mu * std_dev)
            x2 = phi * (log_s_h / std_dev + mu * std_dev)
            y1 = eta * (log_h2_sx / std_dev + mu * std_dev)
            y2 = eta * (log_h_s / std_dev + mu * std_dev)

            cnd = CumulativeNormalDistribution()
            cum_x1 = cnd(x1)
            cum_x2 = cnd(x2)
            cum_y1 = cnd(y1)
            cum_y2 = cnd(y2)
        else:
            cum_x1 = 1.0 if log_s_x > 0.0 else 0.0
            cum_x2 = 1.0 if log_s_h > 0.0 else 0.0
            cum_y1 = 1.0 if log_h2_sx > 0.0 else 0.0
            cum_y2 = 1.0 if log_h_s > 0.0 else 0.0

        alpha = 0.0
        # Eight (barrier_type, option_type, strike vs barrier) branches.
        if bt == BarrierType.DownIn:
            if opt_type == OptionType.Call:
                if strike >= barrier:
                    # B3
                    alpha = h_s_2mu * cum_y1
                else:
                    # B1-B2+B4
                    alpha = cum_x1 - cum_x2 + h_s_2mu * cum_y2
            elif strike >= barrier:
                # B2-B3+B4
                alpha = cum_x2 + h_s_2mu * (-cum_y1 + cum_y2)
            else:
                # B1
                alpha = cum_x1
        elif bt == BarrierType.UpIn:
            if opt_type == OptionType.Call:
                if strike >= barrier:
                    # B1
                    alpha = cum_x1
                else:
                    # B2-B3+B4
                    alpha = cum_x2 + h_s_2mu * (-cum_y1 + cum_y2)
            elif strike >= barrier:
                # B1-B2+B4
                alpha = cum_x1 - cum_x2 + h_s_2mu * cum_y2
            else:
                # B3
                alpha = h_s_2mu * cum_y1
        elif bt == BarrierType.DownOut:
            if opt_type == OptionType.Call:
                if strike >= barrier:
                    # B1-B3
                    alpha = cum_x1 - h_s_2mu * cum_y1
                else:
                    # B2-B4
                    alpha = cum_x2 - h_s_2mu * cum_y2
            elif strike >= barrier:
                # B1-B2+B3-B4
                alpha = cum_x1 - cum_x2 + h_s_2mu * (cum_y1 - cum_y2)
            else:
                alpha = 0.0
        elif opt_type == OptionType.Call:
            if strike >= barrier:
                alpha = 0.0
            else:
                # B1-B2+B3-B4
                alpha = cum_x1 - cum_x2 + h_s_2mu * (cum_y1 - cum_y2)
        elif strike >= barrier:
            # B2-B4
            alpha = cum_x2 - h_s_2mu * cum_y2
        else:
            # B1-B3
            alpha = cum_x1 - h_s_2mu * cum_y1

        return discount * cash_payoff * alpha


__all__ = ["AnalyticBinaryBarrierEngine"]
