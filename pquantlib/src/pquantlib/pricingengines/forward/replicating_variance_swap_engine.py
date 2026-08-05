"""ReplicatingVarianceSwapEngine — Demeterfi-Derman-Kamal-Zou (1999).

# C++ parity:
# ql/pricingengines/forward/replicatingvarianceswapengine.hpp (v1.43) —
# header-only, there is no matching .cpp.

Prices a variance swap by the static replication of a log contract
described in Demeterfi, Derman, Kamal & Zou, "A Guide to Volatility and
Variance Swaps" (1999): the log payoff is approximated by a piecewise
linear function through the available strikes, and the slope changes at
each strike give the weight of the vanilla option struck there.

The construction, per side:

1. sort the strikes (calls ascending, puts descending) and append one
   end strike ``dk`` beyond the last, which exists only to give the last
   segment a slope and is then discarded;
2. drop duplicates;
3. for each strike take the absolute slope of the log payoff over the
   segment to the next strike, and weight the option by that slope minus
   the previous one (the first strike gets the bare slope).

The fair variance is then the cost of that portfolio plus the analytic
part of the log contract, and the NPV is

    +/-1 * df * notional * (fair_variance - variance_strike)

with the sign taken from the swap's :class:`~pquantlib.position.PositionType`.

The weight ladder is published as ``additional_results["optionWeights"]``
— a list of ``(payoff, weight)`` pairs in emission order (all calls
ascending, then all puts descending).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.variance_swap import (
    VarianceSwapArguments,
    VarianceSwapResults,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.position import PositionType
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

# C++ ``weights_type`` = std::vector<std::pair<shared_ptr<StrikedTypePayoff>, Real>>.
type OptionWeights = list[tuple[StrikedTypePayoff, float]]


class ReplicatingVarianceSwapEngine(GenericEngine[VarianceSwapArguments, VarianceSwapResults]):
    """Variance-swap engine using replicating cost.

    # C++ parity: ``ReplicatingVarianceSwapEngine(process, dk,
    # callStrikes, putStrikes)``.

    ``dk`` (default ``5.0``, as in C++) is the step beyond the outermost
    strike on each side used to close the piecewise approximation; it is
    not a strike of its own and no option is written on it, but it does
    set the last slope on each side and therefore moves every weight.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        dk: float = 5.0,
        call_strikes: Sequence[float] = (),
        put_strikes: Sequence[float] = (),
    ) -> None:
        super().__init__(VarianceSwapArguments(), VarianceSwapResults())
        # C++ also has ``QL_REQUIRE(process_, "no process given")``; a
        # None process cannot reach here under pyright strict.
        qassert.require(len(call_strikes) > 0 and len(put_strikes) > 0, "no strike(s) given")
        qassert.require(min(put_strikes) > 0.0, "min put strike must be positive")
        qassert.require(
            min(call_strikes) == max(put_strikes),
            "min call and max put strikes differ",
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._dk: float = dk
        self._call_strikes: list[float] = list(call_strikes)
        self._put_strikes: list[float] = list(put_strikes)
        process.register_with(self)

    # --- helpers (mirror C++) -------------------------------------------

    def _underlying(self) -> float:
        return self._process.x0()

    def _residual_time(self) -> float:
        return self._process.time(self._arguments.maturity_date)

    def _risk_free_rate(self) -> float:
        # C++ passes extrapolate=true here (and only here).
        return (
            self._process.risk_free_rate()
            .zero_rate(
                self._residual_time(),
                Compounding.Continuous,
                Frequency.NoFrequency,
                extrapolate=True,
            )
            .rate()
        )

    def _risk_free_discount(self) -> float:
        return self._process.risk_free_rate().discount(self._residual_time())

    def _compute_log_payoff(self, strike: float, call_put_strike_boundary: float) -> float:
        """Log-contract payoff at ``strike``, measured from the boundary.

        # C++ parity: ``computeLogPayoff``
        # (replicatingvarianceswapengine.hpp:141-147).
        """
        f = call_put_strike_boundary
        return (2.0 / self._residual_time()) * (((strike - f) / f) - math.log(strike / f))

    def _compute_option_weights(
        self,
        avail_strikes: Sequence[float],
        option_type: OptionType,
        option_weights: OptionWeights,
    ) -> None:
        """Append this side's replicating weights to ``option_weights``.

        # C++ parity: ``computeOptionWeights``
        # (replicatingvarianceswapengine.hpp:96-138).
        """
        if len(avail_strikes) == 0:
            return

        if option_type == OptionType.Call:
            strikes = sorted(avail_strikes)
            strikes.append(strikes[-1] + self._dk)
        else:
            strikes = sorted(avail_strikes, reverse=True)
            strikes.append(max(strikes[-1] - self._dk, 0.0))

        # C++ ``std::unique`` only collapses ADJACENT equal elements, but
        # the vector has just been sorted, so this removes every duplicate.
        deduped: list[float] = []
        for k in strikes:
            if not deduped or deduped[-1] != k:
                deduped.append(k)
        strikes = deduped

        f = strikes[0]
        prev_slope = 0.0
        # The appended end strike only supplies the last segment's slope
        # and gets no option of its own — hence ``len(strikes) - 1``.
        for i in range(len(strikes) - 1):
            slope = abs(
                (self._compute_log_payoff(strikes[i + 1], f) - self._compute_log_payoff(strikes[i], f))
                / (strikes[i + 1] - strikes[i])
            )
            payoff = PlainVanillaPayoff(option_type, strikes[i])
            option_weights.append((payoff, slope if i == 0 else slope - prev_slope))
            prev_slope = slope

    def _compute_replicating_portfolio(self, option_weights: OptionWeights) -> float:
        """Fair variance = strip cost + analytic part of the log contract.

        # C++ parity: ``computeReplicatingPortfolio``
        # (replicatingvarianceswapengine.hpp:150-172).
        """
        exercise = EuropeanExercise(self._arguments.maturity_date)
        option_engine = AnalyticEuropeanEngine(self._process)
        options_value = 0.0
        for payoff, weight in option_weights:
            option = EuropeanOption(payoff, exercise)
            option.set_pricing_engine(option_engine)
            options_value += option.npv() * weight

        f = option_weights[0][0].strike()
        underlying = self._underlying()
        rf_discount = self._risk_free_discount()
        return (
            2.0 * self._risk_free_rate()
            - 2.0 / self._residual_time() * (((underlying / rf_discount - f) / f) + math.log(f / underlying))
            + options_value / rf_discount
        )

    # --- main entry point -------------------------------------------------

    def calculate(self) -> None:
        """Compute the fair variance and the swap NPV.

        # C++ parity: ``ReplicatingVarianceSwapEngine::calculate``
        # (replicatingvarianceswapengine.hpp:175-201).
        """
        args = self._arguments
        results = self._results

        option_weights: OptionWeights = []
        self._compute_option_weights(self._call_strikes, OptionType.Call, option_weights)
        self._compute_option_weights(self._put_strikes, OptionType.Put, option_weights)

        variance = self._compute_replicating_portfolio(option_weights)
        results.variance = variance

        risk_free_discount = self._process.risk_free_rate().discount(args.maturity_date)
        multiplier = 1.0 if args.position == PositionType.Long else -1.0

        assert args.notional is not None
        assert args.strike is not None
        results.value = multiplier * risk_free_discount * args.notional * (variance - args.strike)

        results.additional_results["optionWeights"] = option_weights


__all__ = ["OptionWeights", "ReplicatingVarianceSwapEngine"]
