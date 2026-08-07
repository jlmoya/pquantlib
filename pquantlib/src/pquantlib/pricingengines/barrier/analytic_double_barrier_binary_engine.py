"""AnalyticDoubleBarrierBinaryEngine — C.H. Hui one-touch double-barrier series.

# C++ parity: ql/pricingengines/barrier/analyticdoublebarrierbinaryengine.{hpp,cpp}
# (v1.43) — ``class AnalyticDoubleBarrierBinaryEngine :
# public DoubleBarrierOption::engine``.

Implements C.H. Hui, "One-Touch Double Barrier Binary Option Values",
*Applied Financial Economics* 6/1996, as reproduced in Haug, *The Complete
Guide to Option Pricing Formulas*, 2nd ed., p. 180.

Two different series, chosen by barrier type:

* ``KnockIn`` / ``KnockOut`` pay **at expiry**, so they need a European
  exercise and use :meth:`_AnalyticDoubleBarrierBinaryEngineHelper.payoff_at_expiry`
  (100 iterations, i.e. 99 terms).
* ``KIKO`` / ``KOKI`` have a knock-**in** leg that pays **at hit**, so they
  need an American exercise whose first date is not after the vol
  reference date, and use
  :meth:`_AnalyticDoubleBarrierBinaryEngineHelper.payoff_kiko`
  (1000 iterations, i.e. 999 terms). ``KOKI`` swaps the two barriers before
  entering the series.

Both series carry a hard convergence check: if the last term computed is
not below ``1e-8`` in absolute value the engine raises rather than
returning a truncated sum. That fires for wide barriers with a small
volatility, where ``alpha`` is large and ``(S/H_hi)^alpha`` is
astronomically big.

Result contract (pinned in the cross-validation):

* On the normal path **only** ``value`` is assigned; ``option.delta()``
  and friends raise. The class docstring in C++ claims "greeks are
  calculated by simple numeric derivation", but ``calculate()`` does no
  such thing.
* On the four degenerate early returns (spot at or beyond a barrier)
  ``value``, ``delta``, ``gamma``, ``vega`` and ``rho`` are set — the last
  four to hard zeros — while ``theta`` and ``dividend_rho`` stay unset.
* The payoff's option **type** and **strike** are ignored; only its cash
  payoff enters. (The C++ source still carries the commented-out
  ``Option::Type type = payoff_->optionType(); // this is not used ?``.)
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.exercise import AmericanExercise, Exercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import CashOrNothingPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

# C++ declares its own `static Real PI = 3.14159265358979323846264338327950;`
# rather than using M_PI. The literal rounds to the same double as math.pi.
_PI: Final[float] = 3.14159265358979323846264338327950


class _AnalyticDoubleBarrierBinaryEngineHelper:
    """Series evaluator for the two Hui payoff shapes.

    # C++ parity: ``class AnalyticDoubleBarrierBinaryEngine_helper``
    # (analyticdoublebarrierbinaryengine.cpp:31-56) — a file-local helper
    # with no header declaration, kept here as a private class so the two
    # series stay one-for-one with C++.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        payoff: CashOrNothingPayoff,
        arguments: DoubleBarrierOptionArguments,
    ) -> None:
        self._process = process
        self._payoff = payoff
        self._arguments = arguments

    def _common(self, variance: float) -> tuple[float, float, float, float, float]:
        """Residual time, sigma^2, alpha, beta and the log barrier ratio Z."""
        args = self._arguments
        assert args.exercise is not None
        residual_time = self._process.time(args.exercise.last_date())
        qassert.require(residual_time > 0.0, "expiration time must be > 0")

        sigmaq = variance / residual_time
        r = (
            self._process.risk_free_rate()
            .zero_rate(residual_time, Compounding.Continuous, Frequency.NoFrequency)
            .rate()
        )
        q = (
            self._process.dividend_yield()
            .zero_rate(residual_time, Compounding.Continuous, Frequency.NoFrequency)
            .rate()
        )
        b = r - q

        alpha = -0.5 * (2 * b / sigmaq - 1)
        beta = -0.25 * math.pow(2 * b / sigmaq - 1, 2) - 2 * r / sigmaq
        return residual_time, sigmaq, alpha, beta, r

    def payoff_at_expiry(
        self,
        spot: float,
        variance: float,
        barrier_type: DoubleBarrierType,
        max_iteration: int = 100,
        required_convergence: float = 1e-8,
    ) -> float:
        """# C++ parity: ``...helper::payoffAtExpiry`` (pays at expiry)."""
        qassert.require(spot > 0.0, "positive spot value required")
        qassert.require(variance >= 0.0, "negative variance not allowed")

        args = self._arguments
        assert args.exercise is not None
        assert args.barrier_lo is not None
        assert args.barrier_hi is not None

        _, _, alpha, beta, _ = self._common(variance)

        cash = self._payoff.cash_payoff()
        barrier_lo = args.barrier_lo
        barrier_hi = args.barrier_hi

        z = math.log(barrier_hi / barrier_lo)
        factor = (2 * _PI * cash) / math.pow(z, 2)
        lo_alpha = math.pow(spot / barrier_lo, alpha)
        hi_alpha = math.pow(spot / barrier_hi, alpha)

        tot = 0.0
        term = 0.0
        for i in range(1, max_iteration):
            term1 = (lo_alpha - math.pow(-1.0, i) * hi_alpha) / (
                math.pow(alpha, 2) + math.pow(i * _PI / z, 2)
            )
            term2 = math.sin(i * _PI / z * math.log(spot / barrier_lo))
            term3 = math.exp(-0.5 * (math.pow(i * _PI / z, 2) - beta) * variance)
            term = factor * i * term1 * term2 * term3
            tot += term

        # For extreme parameters (big alpha) the series converges very poorly;
        # C++ refuses rather than returning a truncated sum. See Hui (1996).
        qassert.require(
            abs(term) < required_convergence, "serie did not converge sufficiently fast"
        )

        if barrier_type == DoubleBarrierType.KnockOut:
            return max(tot, 0.0)
        discount = self._process.risk_free_rate().discount(args.exercise.last_date())
        qassert.require(discount > 0.0, "positive discount required")
        return max(cash * discount - tot, 0.0)

    def payoff_kiko(
        self,
        spot: float,
        variance: float,
        barrier_type: DoubleBarrierType,
        max_iteration: int = 1000,
        required_convergence: float = 1e-8,
    ) -> float:
        """# C++ parity: ``...helper::payoffKIKO`` (knock-in leg pays at hit)."""
        qassert.require(spot > 0.0, "positive spot value required")
        qassert.require(variance >= 0.0, "negative variance not allowed")

        args = self._arguments
        assert args.barrier_lo is not None
        assert args.barrier_hi is not None

        _, _, alpha, beta, _ = self._common(variance)

        cash = self._payoff.cash_payoff()
        barrier_lo = args.barrier_lo
        barrier_hi = args.barrier_hi
        if barrier_type == DoubleBarrierType.KOKI:
            barrier_lo, barrier_hi = barrier_hi, barrier_lo

        z = math.log(barrier_hi / barrier_lo)
        log_s_l = math.log(spot / barrier_lo)

        tot = 0.0
        term = 0.0
        for i in range(1, max_iteration):
            factor = math.pow(i * _PI / z, 2) - beta
            term1 = (
                beta - math.pow(i * _PI / z, 2) * math.exp(-0.5 * factor * variance)
            ) / factor
            term2 = math.sin(i * _PI / z * log_s_l)
            term = (2.0 / (i * _PI)) * term1 * term2
            tot += term
        tot += 1 - log_s_l / z
        tot *= cash * math.pow(spot / barrier_lo, alpha)

        qassert.require(
            abs(term) < required_convergence, "serie did not converge sufficiently fast"
        )

        return max(tot, 0.0)


class AnalyticDoubleBarrierBinaryEngine(
    GenericEngine[DoubleBarrierOptionArguments, OneAssetOptionResults]
):
    """Hui (1996) series engine for cash-or-nothing double-barrier options.

    # C++ parity: ``AnalyticDoubleBarrierBinaryEngine(process)``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(DoubleBarrierOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def _knocked(self, value: float) -> None:
        """Fill the degenerate (spot outside the barriers) result block.

        # C++ parity: the five assignments repeated in each of the four
        # degenerate switch arms. Note theta and dividend_rho are NOT set.
        """
        results = self._results
        results.value = value
        results.delta = 0.0
        results.gamma = 0.0
        results.vega = 0.0
        results.rho = 0.0

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticDoubleBarrierBinaryEngine::calculate``."""
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        exercise = args.exercise

        if args.barrier_type in (DoubleBarrierType.KIKO, DoubleBarrierType.KOKI):
            qassert.require(
                isinstance(exercise, AmericanExercise),
                "KIKO/KOKI options must have American exercise",
            )
            assert isinstance(exercise, AmericanExercise)
            qassert.require(
                exercise.dates()[0]
                <= self._process.black_volatility().reference_date(),
                "American option with window exercise not handled yet",
            )
        else:
            qassert.require(
                exercise.type() == Exercise.Type.European, "non-European exercise given"
            )

        qassert.require(
            isinstance(args.payoff, CashOrNothingPayoff),
            "a cash-or-nothing payoff must be given",
        )
        assert isinstance(args.payoff, CashOrNothingPayoff)
        payoff: CashOrNothingPayoff = args.payoff

        spot = self._process.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying given")

        variance = self._process.black_volatility().black_variance(
            exercise.last_date(), payoff.strike()
        )
        barrier_lo = args.barrier_lo
        barrier_hi = args.barrier_hi
        barrier_type = args.barrier_type
        assert barrier_lo is not None
        assert barrier_hi is not None
        qassert.require(barrier_lo > 0.0, "positive low barrier value required")
        qassert.require(barrier_hi > 0.0, "positive high barrier value required")
        qassert.require(barrier_lo < barrier_hi, "barrier_lo must be < barrier_hi")
        qassert.require(
            barrier_type
            in (
                DoubleBarrierType.KnockIn,
                DoubleBarrierType.KnockOut,
                DoubleBarrierType.KIKO,
                DoubleBarrierType.KOKI,
            ),
            "Unsupported barrier type",
        )

        # Degenerate cases: the spot is already at or beyond a barrier.
        if barrier_type == DoubleBarrierType.KnockOut:
            if spot <= barrier_lo or spot >= barrier_hi:
                self._knocked(0.0)  # knocked out, no value
                return
        elif barrier_type == DoubleBarrierType.KnockIn:
            if spot <= barrier_lo or spot >= barrier_hi:
                self._knocked(payoff.cash_payoff())  # knocked in - pays
                return
        elif barrier_type == DoubleBarrierType.KIKO:
            if spot >= barrier_hi:
                self._knocked(0.0)  # knocked out, no value
                return
            if spot <= barrier_lo:
                self._knocked(payoff.cash_payoff())  # knocked in, pays
                return
        elif barrier_type == DoubleBarrierType.KOKI:
            if spot <= barrier_lo:
                self._knocked(0.0)  # knocked out, no value
                return
            if spot >= barrier_hi:
                self._knocked(payoff.cash_payoff())  # knocked in, pays
                return

        assert barrier_type is not None  # narrowed by the require above
        helper = _AnalyticDoubleBarrierBinaryEngineHelper(self._process, payoff, args)
        if barrier_type in (DoubleBarrierType.KnockOut, DoubleBarrierType.KnockIn):
            results.value = helper.payoff_at_expiry(spot, variance, barrier_type)
        else:
            results.value = helper.payoff_kiko(spot, variance, barrier_type)


__all__ = ["AnalyticDoubleBarrierBinaryEngine"]
