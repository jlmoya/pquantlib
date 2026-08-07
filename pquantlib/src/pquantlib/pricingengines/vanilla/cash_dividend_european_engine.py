"""CashDividendEuropeanEngine — spot / escrowed cash-dividend European engine.

# C++ parity: ql/pricingengines/vanilla/cashdividendeuropeanengine.{hpp,cpp}
# (v1.43) — ``class CashDividendEuropeanEngine : public VanillaOption::engine``.

Reference: Jherek Healy (2021), *The Pricing of Vanilla Options with Cash
Dividends as a Classic Vanilla Basket Option Problem*,
https://arxiv.org/pdf/2106.12971

The engine has four code paths, chosen in this order:

1. **Escrowed** model, or exactly one (filtered) dividend falling on the
   settlement date — delegate to
   :class:`~pquantlib.pricingengines.vanilla.analytic_dividend_european_engine.AnalyticDividendEuropeanEngine`
   and copy **only** the NPV. The delegate's greeks are deliberately
   discarded, so ``option.delta()`` raises for every case this engine
   prices.
2. Nothing left after filtering to ``settlement <= date <= maturity`` and
   ``amount > 0`` — the "underlyings" collapse to the strike alone and the
   option prices as a plain European.
3. A single dividend falling exactly **on** the maturity date — it is
   *merged into the strike* (``amount + strike``) and the underlyings again
   collapse to one, so this too prices as a plain European, at the bumped
   strike.
4. Everything else — the Healy basket representation, priced with
   :class:`~pquantlib.pricingengines.basket.choi_basket_engine.ChoiBasketEngine`
   over one synthetic lognormal leg per dividend plus one for the strike.

Path 4 needs some care to reproduce. Each leg is a ``BlackProcess``-shaped
:class:`GeneralizedBlackScholesProcess` whose *spot* is the dividend amount,
whose risk-free curve is flat **zero**, whose dividend curve carries the
synthetic rate ``q_mod - r_mod`` with
``x_mod = log(x_df(dividend date)) / maturity`` — note the discount is taken
at the **dividend** date but divided by the **maturity** time — and whose
volatility is ``sqrt(blackVariance(dividend date, strike) / maturity)``. The
correlation between legs ``i`` and ``j < i`` is ``v_j / sqrt(v_i v_j)``,
degenerating to ``QL_EPSILON`` when ``v_j`` vanishes. The basket is then an
``AverageBasketPayoff`` wrapping a **Put** struck at the *spot*, with unit
(not ``1/n``) weights, and the Choi engine is configured
``(lambda=10, maxNrIntegrationSteps=2000, calcFwdDelta=False,
controlVariate=True)``. The put branch finally converts back through
put-call parity on the dividend-adjusted forward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend, FixedDividend
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.basket_option import AverageBasketPayoff, BasketOption
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.constants import QL_EPSILON
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.choi_basket_engine import ChoiBasketEngine
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_dividend_european_engine import (
    AnalyticDividendEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.date import Date


class CashDividendModel(IntEnum):
    """Discrete-dividend treatment.

    # C++ parity: ``enum CashDividendModel { Spot, Escrowed };`` nested in
    # ``CashDividendEuropeanEngine``. Integer values follow C++ declaration
    # order: Spot = 0, Escrowed = 1.
    """

    Spot = 0
    Escrowed = 1


class CashDividendEuropeanEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """(Semi-)analytic European engine for spot and escrowed cash dividends.

    # C++ parity: ``CashDividendEuropeanEngine(process, dividends,
    # cashDividendModel = Spot)``.
    """

    #: Exposed so callers can write ``CashDividendEuropeanEngine.Escrowed``,
    #: mirroring the C++ nested enum.
    Spot = CashDividendModel.Spot
    Escrowed = CashDividendModel.Escrowed

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        dividends: Sequence[Dividend] = (),
        cash_dividend_model: CashDividendModel = CashDividendModel.Spot,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._dividends: list[Dividend] = list(dividends)
        self._cash_dividend_model: CashDividendModel = cash_dividend_model
        process.register_with(self)

    def cash_dividend_model(self) -> CashDividendModel:
        return self._cash_dividend_model

    def dividends(self) -> list[Dividend]:
        return list(self._dividends)

    # --- inner-engine plumbing -------------------------------------------

    def _price_with(
        self,
        engine: GenericEngine[OptionArguments, OneAssetOptionResults],
        payoff: PlainVanillaPayoff,
        exercise: Exercise,
    ) -> float:
        """Run ``engine`` on (payoff, exercise) and return its NPV.

        C++ builds a throwaway ``VanillaOption`` and calls ``NPV()``; that
        does nothing but copy the payoff and exercise into the engine's
        arguments, validate, and calculate. Driving the engine directly
        keeps the same semantics without an instrument round-trip (the same
        shape ``ForwardVanillaEngine._setup`` already uses in this port).
        """
        engine.reset()
        args = engine.get_arguments()
        args.payoff = payoff
        args.exercise = exercise
        args.validate()
        engine.calculate()
        value = engine.get_results().value
        assert value is not None
        return value

    def calculate(self) -> None:
        """# C++ parity: ``CashDividendEuropeanEngine::calculate``."""
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        exercise = args.exercise
        qassert.require(
            exercise.type() == Exercise.Type.European, "not an European option"
        )
        # C++ casts to PlainVanillaPayoff (not merely StrikedTypePayoff), so a
        # cash-or-nothing payoff is rejected here despite the variable's type.
        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        strike = payoff.strike()
        process = self._process
        r_ts = process.risk_free_rate()

        settlement_date = r_ts.reference_date()
        maturity_date = exercise.last_date()

        dividends = [
            d
            for d in self._dividends
            if settlement_date <= d.date() <= maturity_date and d.amount() > 0.0
        ]
        dividends.sort(key=lambda d: d.date())

        # (1) Escrowed, or a lone dividend on the settlement date: delegate and
        #     copy ONLY the value. Note C++ hands the delegate the UNFILTERED
        #     schedule, which then applies its own (inclusive) window.
        if self._cash_dividend_model == CashDividendModel.Escrowed or (
            len(dividends) == 1 and dividends[-1].date() == settlement_date
        ):
            results.value = self._price_with(
                AnalyticDividendEuropeanEngine(process, self._dividends),
                payoff,
                exercise,
            )
            return

        underlyings: list[Dividend] = list(dividends)
        if underlyings and underlyings[-1].date() == maturity_date:
            # A dividend paid at maturity is indistinguishable from a strike
            # adjustment, so it is folded into the strike rather than becoming
            # a separate basket leg.
            underlyings[-1] = FixedDividend(
                underlyings[-1].amount() + strike, maturity_date
            )
        else:
            underlyings.append(FixedDividend(strike, maturity_date))

        # (2)/(3) One underlying left: a plain European at that amount.
        if len(underlyings) == 1:
            results.value = self._price_with(
                AnalyticEuropeanEngine(process),
                PlainVanillaPayoff(payoff.option_type(), underlyings[-1].amount()),
                exercise,
            )
            return

        # (4) The Healy basket representation.
        results.value = self._basket_value(
            payoff, exercise, dividends, underlyings, maturity_date
        )

    # --- the Healy basket branch -----------------------------------------

    def _leg_process(
        self,
        amount: float,
        dividend_date: Date,
        strike: float,
        maturity: float,
        settlement_date: Date,
    ) -> GeneralizedBlackScholesProcess:
        """One synthetic lognormal leg of the Healy basket.

        # C++ parity: cashdividendeuropeanengine.cpp:120-146. Note ``r_mod`` and
        # ``q_mod`` take the log-discount at the DIVIDEND date but divide by the
        # MATURITY time, and the leg's risk-free curve is flat zero, so the
        # whole carry lives in the synthetic dividend rate ``q_mod - r_mod``.
        """
        process = self._process
        r_ts = process.risk_free_rate()
        q_ts = process.dividend_yield()
        vol_ts = process.black_volatility()

        r_mod = math.log(r_ts.discount(dividend_date)) / maturity
        q_mod = math.log(q_ts.discount(dividend_date)) / maturity

        return GeneralizedBlackScholesProcess(
            x0=SimpleQuote(amount),
            dividend_ts=FlatForward.from_rate(
                reference_date=settlement_date,
                forward_rate=q_mod - r_mod,
                day_counter=r_ts.day_counter(),
            ),
            risk_free_ts=FlatForward.from_rate(
                reference_date=settlement_date,
                forward_rate=0.0,
                day_counter=r_ts.day_counter(),
            ),
            black_vol_ts=BlackConstantVol(
                reference_date=vol_ts.reference_date(),
                calendar=vol_ts.calendar(),
                volatility=SimpleQuote(
                    math.sqrt(vol_ts.black_variance(dividend_date, strike) / maturity)
                ),
                day_counter=vol_ts.day_counter(),
            ),
        )

    def _basket_value(
        self,
        payoff: PlainVanillaPayoff,
        exercise: Exercise,
        dividends: list[Dividend],
        underlyings: list[Dividend],
        maturity_date: Date,
    ) -> float:
        """# C++ parity: cashdividendeuropeanengine.cpp:118-190."""
        process = self._process
        r_ts = process.risk_free_rate()
        q_ts = process.dividend_yield()
        vol_ts = process.black_volatility()

        settlement_date = r_ts.reference_date()
        maturity = process.time(maturity_date)
        strike = payoff.strike()
        n = len(underlyings)

        processes = [
            self._leg_process(
                u.amount(), u.date(), strike, maturity, settlement_date
            )
            for u in underlyings
        ]

        v = [vol_ts.black_variance(u.date(), strike) for u in underlyings]

        rho = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            rho[i][i] = 1.0
            for j in range(i):
                # C++ writes v[j] / sqrt(v[i]*v[j]) rather than the
                # sqrt(v[j]/v[i]) it simplifies to; kept literal because the two
                # round differently.
                if v[j] > QL_EPSILON:
                    rho[i][j] = rho[j][i] = v[j] / math.sqrt(v[i] * v[j])
                else:
                    rho[i][j] = rho[j][i] = QL_EPSILON

        # A PUT struck at the spot, over unit-weighted legs: the basket
        # `sum_i D_i` is the terminal payment stream, and the exchange against
        # the spot is what turns it back into the option.
        basket_option = BasketOption(
            AverageBasketPayoff(
                PlainVanillaPayoff(OptionType.Put, process.x0()),
                [1.0] * n,
            ),
            EuropeanExercise(maturity_date),
        )
        basket_option.set_pricing_engine(
            ChoiBasketEngine(
                processes,
                rho,
                10.0,
                2000,
                calc_fwd_delta=False,
                control_variate=True,
            )
        )
        del exercise  # the basket carries its own European exercise

        basket_npv = basket_option.npv()
        q_discount_maturity = q_ts.discount(maturity_date)

        if payoff.option_type() == OptionType.Call:
            return basket_npv * q_discount_maturity

        # Put: convert through parity on the dividend-adjusted forward. Note the
        # sum runs over the FILTERED dividends, not over `underlyings` (which
        # carries the extra strike leg).
        div_discounted = 0.0
        for div in dividends:
            div_discounted += (
                div.amount() * r_ts.discount(div.date()) / q_ts.discount(div.date())
            )
        fwd = (
            (process.x0() - div_discounted)
            * q_discount_maturity
            / r_ts.discount(maturity_date)
        )
        return basket_npv * q_discount_maturity - (fwd - payoff.strike()) * r_ts.discount(
            maturity_date
        )


__all__ = ["CashDividendEuropeanEngine", "CashDividendModel"]
