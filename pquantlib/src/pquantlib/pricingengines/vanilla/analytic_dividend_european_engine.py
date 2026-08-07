"""AnalyticDividendEuropeanEngine — European option with discrete cash dividends.

# C++ parity: ql/pricingengines/vanilla/analyticdividendeuropeanengine.{hpp,cpp}
# (v1.43) — ``class AnalyticDividendEuropeanEngine : public VanillaOption::engine``.

The engine subtracts the present value of every dividend falling in
``[settlement_date, expiry]`` from the spot, then prices a plain Black
European on the reduced spot. Each dividend is discounted with the
risk-free curve and *inflated* by the dividend curve
(``r_df(d) / q_df(d)``) because the escrowed amount would otherwise have
grown at the dividend yield.

Two things a port gets wrong easily and that the cross-validation pins:

* The dividend window is closed on **both** sides — a dividend on the
  settlement date counts, and so does one on the expiry date.
* The greeks read **three different day counters**: the vega time comes
  from the vol surface's day counter, while theta and rho use
  ``process.time()`` (the risk-free one), and the ``delta_theta``
  accumulation reads the zero rates on the risk-free and dividend
  curves' own day counters respectively.

``dividend_rho`` is never assigned, so ``option.dividend_rho()`` raises —
that absence is part of the C++ contract.

In v1.43 the dividend schedule is a constructor argument of the *engine*;
the old ``DividendVanillaOption`` instrument was removed, so this engine
prices a plain :class:`~pquantlib.instruments.vanilla_option.VanillaOption`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class AnalyticDividendEuropeanEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Analytic European engine with a discrete-dividend schedule.

    # C++ parity: ``AnalyticDividendEuropeanEngine(process, dividends)``.

    Args:
        process: the Black-Scholes process of the underlying.
        dividends: the discrete cash-dividend schedule (C++
            ``DividendSchedule`` = ``std::vector<shared_ptr<Dividend>>``).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        dividends: Sequence[Dividend] = (),
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._dividends: list[Dividend] = list(dividends)
        process.register_with(self)

    def dividends(self) -> list[Dividend]:
        """The dividend schedule handed to the constructor."""
        return list(self._dividends)

    def _in_window(self, d: Date, settlement_date: Date, last_date: Date) -> bool:
        """C++ parity: ``cashFlowDate >= settlementDate && <= exercise->lastDate()``.

        Both bounds are inclusive.
        """
        return settlement_date <= d <= last_date

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticDividendEuropeanEngine::calculate``."""
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not an European option"
        )
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        process = self._process
        r_ts = process.risk_free_rate()
        q_ts = process.dividend_yield()
        vol_ts = process.black_volatility()

        settlement_date = r_ts.reference_date()
        last_date = args.exercise.last_date()

        riskless = 0.0
        for div in self._dividends:
            cash_flow_date = div.date()
            if self._in_window(cash_flow_date, settlement_date, last_date):
                riskless += (
                    div.amount()
                    * r_ts.discount(cash_flow_date)
                    / q_ts.discount(cash_flow_date)
                )

        spot = process.state_variable().value() - riskless
        qassert.require(
            spot > 0.0, "negative or null underlying after subtracting dividends"
        )

        dividend_discount = q_ts.discount(last_date)
        risk_free_discount = r_ts.discount(last_date)
        forward_price = spot * dividend_discount / risk_free_discount

        variance = vol_ts.black_variance(last_date, payoff.strike())

        black = BlackCalculator(
            payoff, forward_price, math.sqrt(variance), risk_free_discount
        )

        results.value = black.value()
        results.delta = black.delta(spot)
        results.gamma = black.gamma(spot)

        rfdc = r_ts.day_counter()
        dydc = q_ts.day_counter()
        voldc = vol_ts.day_counter()
        # Vega is measured on the VOL surface's own day counter, unlike theta
        # and rho below which use process.time() (the risk-free one).
        t_vega = voldc.year_fraction(vol_ts.reference_date(), last_date)
        results.vega = black.vega(t_vega)

        delta_theta = 0.0
        delta_rho = 0.0
        for div in self._dividends:
            d = div.date()
            if not self._in_window(d, settlement_date, last_date):
                continue
            delta_theta -= (
                div.amount()
                * (
                    r_ts.zero_rate(
                        d, Compounding.Continuous, Frequency.Annual, False, rfdc
                    ).rate()
                    - q_ts.zero_rate(
                        d, Compounding.Continuous, Frequency.Annual, False, dydc
                    ).rate()
                )
                * r_ts.discount(d)
                / q_ts.discount(d)
            )
            # C++ switches to the TIME overloads of discount() here, using
            # process_->time(d) for both curves -- so the dividend curve is
            # queried at the risk-free curve's year fraction, not its own.
            t_div = process.time(d)
            delta_rho += (
                div.amount() * t_div * r_ts.discount(t_div) / q_ts.discount(t_div)
            )

        t = process.time(last_date)
        try:
            results.theta = black.theta(spot, t) + delta_theta * black.delta(spot)
        except LibraryException:
            results.theta = None

        results.rho = black.rho(t) + delta_rho * black.delta(spot)


__all__ = ["AnalyticDividendEuropeanEngine"]
