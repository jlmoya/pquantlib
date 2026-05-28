"""AnalyticDoubleBarrierEngine — Ikeda-Kunitomo (1992) closed-form
double-barrier pricing.

# C++ parity:
# ql/pricingengines/barrier/analyticdoublebarrierengine.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for single-asset double-barrier european options
under Black-Scholes dynamics, following Ikeda-Kunitomo:

  Reference: Ikeda M., Kunitomo N., "Pricing Options with Curved
  Boundaries", Mathematical Finance 2/1992.
  Also: Haug "The Complete Guide to Option Pricing Formulas" 2nd ed.,
  §4.13.

The engine builds the price from a doubly-indexed series of standard
normals (n in [-series, +series]). Convergence is geometric — five
terms is well within machine precision for textbook setups.

Supported barrier types: ``KnockIn`` / ``KnockOut``.

KIKO / KOKI are **not** supported by the analytic series (the C++
engine ``QL_FAIL``s on those branches because the engine is rooted in
a single Knock-Out boundary-value problem). The instrument still
accepts those enum members; only this analytic engine refuses them.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

_PHI: Final[CumulativeNormalDistribution] = CumulativeNormalDistribution()


class AnalyticDoubleBarrierEngine(
    GenericEngine[DoubleBarrierOptionArguments, OneAssetOptionResults]
):
    """Ikeda-Kunitomo closed-form analytic double-barrier engine.

    # C++ parity: ``AnalyticDoubleBarrierEngine``.

    Parameters:
        process: GBSM process supplying spot + flat curves + vol.
        series: number of terms each side of zero in the
            Ikeda-Kunitomo series. C++ defaults to 5; the geometric
            convergence makes 5 numerically indistinguishable from 10
            for textbook barrier widths.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        series: int = 5,
    ) -> None:
        super().__init__(DoubleBarrierOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._series: int = series
        process.register_with(self)

    # --- helper accessors (mirror C++) ----------------------------------

    def _underlying(self) -> float:
        return self._process.x0()

    def _strike(self) -> float:
        payoff = self._arguments.payoff
        assert isinstance(payoff, PlainVanillaPayoff)
        return payoff.strike()

    def _exercise(self) -> Exercise:
        ex = self._arguments.exercise
        assert ex is not None
        return ex

    def _residual_time(self) -> float:
        return self._process.time(self._exercise().last_date())

    def _volatility(self) -> float:
        # C++ passes ``residualTime()`` here; the Python BlackVolTermStructure
        # ``black_vol`` overload binds ``Date`` (the time-typed overload
        # isn't exposed). The Python port matches the C++ NPV exactly by
        # passing the last_date directly — both routes resolve to the
        # same black variance via the day-counter's ``year_fraction``.
        return self._process.black_volatility().black_vol(
            self._exercise().last_date(), self._strike(), extrapolate=True
        )

    def _volatility_squared(self) -> float:
        v = self._volatility()
        return v * v

    def _barrier_lo(self) -> float:
        assert self._arguments.barrier_lo is not None
        return self._arguments.barrier_lo

    def _barrier_hi(self) -> float:
        assert self._arguments.barrier_hi is not None
        return self._arguments.barrier_hi

    def _std_deviation(self) -> float:
        return self._volatility() * math.sqrt(self._residual_time())

    def _risk_free_rate(self) -> float:
        return self._process.risk_free_rate().zero_rate(
            self._exercise().last_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
        ).rate()

    def _risk_free_discount(self) -> float:
        return self._process.risk_free_rate().discount(self._exercise().last_date())

    def _dividend_yield(self) -> float:
        return self._process.dividend_yield().zero_rate(
            self._exercise().last_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
        ).rate()

    def _dividend_discount(self) -> float:
        return self._process.dividend_yield().discount(self._exercise().last_date())

    def _cost_of_carry(self) -> float:
        return self._risk_free_rate() - self._dividend_yield()

    def _triggered(self, spot: float) -> bool:
        """Spot has touched (or crossed) one of the barriers.

        # C++ parity: ``DoubleBarrierOption::engine::triggered``.
        """
        return spot <= self._barrier_lo() or spot >= self._barrier_hi()

    # --- vanilla equivalent (used by KI = vanilla - KO) -----------------

    def _vanilla_equivalent(self) -> float:
        """European-vanilla equivalent under Black 1976.

        # C++ parity: ``AnalyticDoubleBarrierEngine::vanillaEquivalent``.
        """
        payoff = self._arguments.payoff
        assert isinstance(payoff, StrikedTypePayoff)
        forward_price = (
            self._underlying() * self._dividend_discount() / self._risk_free_discount()
        )
        black = BlackCalculator(
            payoff,
            forward_price,
            self._std_deviation(),
            self._risk_free_discount(),
        )
        return max(0.0, black.value())

    # --- Ikeda-Kunitomo series for KnockOut Call ------------------------

    def _call_ko(self) -> float:
        """Ikeda-Kunitomo series for Call Knock-Out.

        # C++ parity: ``AnalyticDoubleBarrierEngine::callKO``.

        Math-symbol variables (d1/d2/d3/d4, acc1/acc2, mu1, bsigma,
        L2n/U2n, rend, kov) intentionally mirror the C++ literal names.
        """
        # N.B. for flat barriers mu3=mu1 and mu2=0 (per C++ comment).
        sd = self._std_deviation()
        S = self._underlying()  # noqa: N806
        K = self._strike()  # noqa: N806
        L = self._barrier_lo()  # noqa: N806
        U = self._barrier_hi()  # noqa: N806
        vol_sq = self._volatility_squared()
        T = self._residual_time()  # noqa: N806
        b = self._cost_of_carry()
        mu1 = 2.0 * b / vol_sq + 1.0
        bsigma = (b + vol_sq / 2.0) * T / sd

        acc1 = 0.0
        acc2 = 0.0
        for n in range(-self._series, self._series + 1):
            L2n = L ** (2.0 * n)  # noqa: N806
            U2n = U ** (2.0 * n)  # noqa: N806
            d1 = math.log(S * U2n / (K * L2n)) / sd + bsigma
            d2 = math.log(S * U2n / (U * L2n)) / sd + bsigma
            d3 = math.log(L ** (2.0 * n + 2.0) / (K * S * U2n)) / sd + bsigma
            d4 = math.log(L ** (2.0 * n + 2.0) / (U * S * U2n)) / sd + bsigma

            acc1 += (U**n / L**n) ** mu1 * (_PHI(d1) - _PHI(d2)) - (
                L ** (n + 1) / (U**n * S)
            ) ** mu1 * (_PHI(d3) - _PHI(d4))

            acc2 += (U**n / L**n) ** (mu1 - 2.0) * (
                _PHI(d1 - sd) - _PHI(d2 - sd)
            ) - (L ** (n + 1) / (U**n * S)) ** (mu1 - 2.0) * (
                _PHI(d3 - sd) - _PHI(d4 - sd)
            )

        rend = math.exp(-self._dividend_yield() * T)
        kov = S * rend * acc1 - K * self._risk_free_discount() * acc2
        return max(0.0, kov)

    def _put_ko(self) -> float:
        """Ikeda-Kunitomo series for Put Knock-Out.

        # C++ parity: ``AnalyticDoubleBarrierEngine::putKO``.
        """
        sd = self._std_deviation()
        S = self._underlying()  # noqa: N806
        K = self._strike()  # noqa: N806
        L = self._barrier_lo()  # noqa: N806
        U = self._barrier_hi()  # noqa: N806
        vol_sq = self._volatility_squared()
        T = self._residual_time()  # noqa: N806
        b = self._cost_of_carry()
        mu1 = 2.0 * b / vol_sq + 1.0
        bsigma = (b + vol_sq / 2.0) * T / sd

        acc1 = 0.0
        acc2 = 0.0
        for n in range(-self._series, self._series + 1):
            L2n = L ** (2.0 * n)  # noqa: N806
            U2n = U ** (2.0 * n)  # noqa: N806
            y1 = math.log(S * U2n / (L ** (2.0 * n + 1.0))) / sd + bsigma
            y2 = math.log(S * U2n / (K * L2n)) / sd + bsigma
            y3 = math.log(L ** (2.0 * n + 2.0) / (L * S * U2n)) / sd + bsigma
            y4 = math.log(L ** (2.0 * n + 2.0) / (K * S * U2n)) / sd + bsigma

            acc1 += (U**n / L**n) ** (mu1 - 2.0) * (
                _PHI(y1 - sd) - _PHI(y2 - sd)
            ) - (L ** (n + 1) / (U**n * S)) ** (mu1 - 2.0) * (
                _PHI(y3 - sd) - _PHI(y4 - sd)
            )

            acc2 += (U**n / L**n) ** mu1 * (_PHI(y1) - _PHI(y2)) - (
                L ** (n + 1) / (U**n * S)
            ) ** mu1 * (_PHI(y3) - _PHI(y4))

        # rend kept for parity with C++ but unused in put final
        # combination (C++ computes it but the put formula multiplies S
        # by rend below):
        rend = math.exp(-self._dividend_yield() * T)
        kov = K * self._risk_free_discount() * acc1 - S * rend * acc2
        return max(0.0, kov)

    def _call_ki(self) -> float:
        """Call KI = vanilla - callKO, clipped at 0.

        # C++ parity: ``AnalyticDoubleBarrierEngine::callKI``.
        """
        return max(0.0, self._vanilla_equivalent() - self._call_ko())

    def _put_ki(self) -> float:
        """Put KI = vanilla - putKO, clipped at 0.

        # C++ parity: ``AnalyticDoubleBarrierEngine::putKI``.
        """
        return max(0.0, self._vanilla_equivalent() - self._put_ko())

    # --- main entry point -----------------------------------------------

    def calculate(self) -> None:
        """Dispatch on (option_type, barrier_type).

        # C++ parity: ``AnalyticDoubleBarrierEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "this engine handles only european options",
        )

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        qassert.require(payoff.strike() > 0.0, "strike must be positive")

        spot = self._underlying()
        qassert.require(spot > 0.0, "negative or null underlying given")
        qassert.require(not self._triggered(spot), "barrier(s) already touched")

        qassert.require(args.barrier_type is not None, "no barrier type given")
        assert args.barrier_type is not None
        barrier_type = args.barrier_type

        is_call = payoff.option_type() == OptionType.Call
        if is_call and barrier_type == DoubleBarrierType.KnockIn:
            results.value = self._call_ki()
        elif is_call and barrier_type == DoubleBarrierType.KnockOut:
            results.value = self._call_ko()
        elif not is_call and barrier_type == DoubleBarrierType.KnockIn:
            results.value = self._put_ki()
        elif not is_call and barrier_type == DoubleBarrierType.KnockOut:
            results.value = self._put_ko()
        else:
            # KIKO / KOKI: C++ QL_FAIL on these branches.
            qassert.fail(f"unsupported double-barrier type: {int(barrier_type)}")


__all__ = ["AnalyticDoubleBarrierEngine"]
