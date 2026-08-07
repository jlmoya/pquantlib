"""AnalyticPerformanceEngine — closed-form performance (ratchet) option engine.

# C++ parity: ql/pricingengines/cliquet/analyticperformanceengine.{hpp,cpp}
# (v1.43) — ``class AnalyticPerformanceEngine : public CliquetOption::engine``.

A performance option is the cliquet whose per-period payoff is normalised
by the reset spot: instead of ``max(S(t_i) - k*S(t_{i-1}), 0)`` it pays
``max(S(t_i)/S(t_{i-1}) - k, 0)``. Dividing by ``k*S(t_{i-1})`` turns each
period into a Black at unit strike, which is why the engine builds
``PlainVanillaPayoff(type, 1.0)`` and multiplies the result back by the
moneyness.

Compare with the sibling
:class:`~pquantlib.pricingengines.cliquet.analytic_cliquet_engine.AnalyticCliquetEngine`;
the two differ in exactly the places a port is most likely to conflate:

============  ===========================  ==============================
              AnalyticCliquetEngine        AnalyticPerformanceEngine
============  ===========================  ==============================
outer weight  ``qDF(0, t_{i-1})``          ``rDF(0, t_{i-1})`` * moneyness
Black strike  ``S * moneyness``            ``1.0``
Black forward ``S * qFwd / rFwd``          ``(1/moneyness) * qFwd / rFwd``
vol strike    ``S * moneyness``            ``S * moneyness`` (same)
theta rate    dividend curve fwd rate      risk-free curve fwd rate
rho           no ``- t * value`` term      carries ``- t * value``
dividend rho  carries ``- t * value``      no ``- t * value`` term
delta         Black delta + beta term      hard ``0.0``
============  ===========================  ==============================

Both ``delta`` and ``gamma`` are assigned a hard ``0.0`` here (not left
unset). The engine refuses started and capped/floored options.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.cliquet_option import CliquetOptionArguments
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import PercentageStrikePayoff, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.cliquet.analytic_cliquet_engine import (
    check_unsupported_cliquet_features,
    cliquet_reset_grid,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticPerformanceEngine(
    GenericEngine[CliquetOptionArguments, OneAssetOptionResults]
):
    """Closed-form performance-option engine.

    # C++ parity: ``AnalyticPerformanceEngine(process)``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(CliquetOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticPerformanceEngine::calculate``."""
        args = self._arguments
        results = self._results

        check_unsupported_cliquet_features(args)

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not an European option"
        )
        qassert.require(
            isinstance(args.payoff, PercentageStrikePayoff), "wrong payoff given"
        )
        assert isinstance(args.payoff, PercentageStrikePayoff)
        moneyness: PercentageStrikePayoff = args.payoff

        reset_dates = cliquet_reset_grid(args)

        process = self._process
        r_ts = process.risk_free_rate()
        q_ts = process.dividend_yield()
        vol_ts = process.black_volatility()

        underlying = process.state_variable().value()
        qassert.require(underlying > 0.0, "negative or null underlying")

        # Unit strike: the per-period payoff has been divided through by
        # moneyness * S(reset).
        payoff = PlainVanillaPayoff(moneyness.option_type(), 1.0)

        results.value = 0.0
        results.delta = 0.0
        results.gamma = 0.0
        results.theta = 0.0
        results.rho = 0.0
        results.dividend_rho = 0.0
        results.vega = 0.0

        rfdc = r_ts.day_counter()
        divdc = q_ts.day_counter()
        voldc = vol_ts.day_counter()

        for i in range(1, len(reset_dates)):
            start, end = reset_dates[i - 1], reset_dates[i]

            # Outer weight is a RISK-FREE discount to the period start (the
            # cliquet engine uses a dividend one).
            discount = r_ts.discount(start)
            r_discount = r_ts.discount(end) / r_ts.discount(start)
            q_discount = q_ts.discount(end) / q_ts.discount(start)
            forward = (1.0 / moneyness.strike()) * q_discount / r_discount
            # The vol is still read at the *absolute* re-strike level, not at
            # the normalised unit strike.
            variance = vol_ts.black_forward_variance(
                start, end, underlying * moneyness.strike()
            )

            black = BlackCalculator(payoff, forward, math.sqrt(variance), r_discount)

            scale = discount * moneyness.strike()

            results.value += scale * black.value()
            results.delta += 0.0
            results.gamma += 0.0
            results.theta += (
                r_ts.forward_rate(
                    start, end, Compounding.Continuous, Frequency.NoFrequency, False, rfdc
                ).rate()
                * scale
                * black.value()
            )

            dt = rfdc.year_fraction(start, end)
            t = rfdc.year_fraction(r_ts.reference_date(), start)
            results.rho += scale * (black.rho(dt) - t * black.value())

            dt = divdc.year_fraction(start, end)
            results.dividend_rho += scale * black.dividend_rho(dt)

            dt = voldc.year_fraction(start, end)
            results.vega += scale * black.vega(dt)


__all__ = ["AnalyticPerformanceEngine"]
