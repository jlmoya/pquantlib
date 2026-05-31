"""VannaVolgaBarrierEngine — Vanna-Volga FX single-barrier engine.

# C++ parity: ql/experimental/barrieroption/vannavolgabarrierengine.{hpp,cpp}
#             (v1.42.1).

Prices a single-asset FX barrier option by the Vanna-Volga method
(Castagna-Mercurio). The Black-Scholes barrier price (at the ATM vol) is
corrected by a survival-probability-weighted hedge made of three vanilla
options (ATM, 25-delta put, 25-delta call), whose vega/vanna/volga
sensitivities reconcile the model and market smile prices.

This closes the Phase 11 W4-C VannaVolga carve-out.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.barrieroption.vanna_volga_interpolation import (
    VannaVolgaInterpolation,
)
from pquantlib.instruments.barrier_option import (
    BarrierOption,
    BarrierOptionArguments,
    BarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.barrier.analytic_barrier_engine import (
    AnalyticBarrierEngine,
)
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.black_delta_calculator import (
    BlackDeltaCalculator,
)
from pquantlib.processes.black_scholes_merton_process import (
    BlackScholesMertonProcess,
)
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.delta_vol_quote import (
    DeltaVolQuote,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.compounding import Compounding


class VannaVolgaBarrierEngine(
    GenericEngine[BarrierOptionArguments, OneAssetOptionResults]
):
    """Vanna-Volga single-barrier FX engine.

    # C++ parity: ``VannaVolgaBarrierEngine``.

    Args:
        atm_vol: ATM ``DeltaVolQuote``.
        vol25_put: 25-delta put ``DeltaVolQuote`` (delta must be -0.25).
        vol25_call: 25-delta call ``DeltaVolQuote`` (delta must be 0.25).
        spot_fx: spot FX rate quote.
        domestic_ts: domestic (risk-free) yield curve.
        foreign_ts: foreign yield curve.
        adapt_van_delta: enable Vanilla-delta adaptation.
        bs_price_with_smile: BS vanilla price under the smile (used when
            ``adapt_van_delta`` is True).
    """

    def __init__(
        self,
        atm_vol: DeltaVolQuote,
        vol25_put: DeltaVolQuote,
        vol25_call: DeltaVolQuote,
        spot_fx: Quote,
        domestic_ts: YieldTermStructure,
        foreign_ts: YieldTermStructure,
        adapt_van_delta: bool = False,
        bs_price_with_smile: float = 0.0,
    ) -> None:
        super().__init__(BarrierOptionArguments(), OneAssetOptionResults())
        self._atm_vol = atm_vol
        self._vol25_put = vol25_put
        self._vol25_call = vol25_call
        self._t = atm_vol.maturity()
        self._spot_fx = spot_fx
        self._domestic_ts = domestic_ts
        self._foreign_ts = foreign_ts
        self._adapt_van_delta = adapt_van_delta
        self._bs_price_with_smile = bs_price_with_smile

        qassert.require(
            vol25_put.delta() == -0.25, "25 delta put is required by vanna volga method"
        )
        qassert.require(
            vol25_call.delta() == 0.25, "25 delta call is required by vanna volga method"
        )
        qassert.require(
            vol25_put.maturity() == vol25_call.maturity()
            and vol25_put.maturity() == atm_vol.maturity(),
            "Maturity of 3 vols are not the same",
        )

        for obs in (atm_vol, vol25_put, vol25_call, spot_fx, domestic_ts, foreign_ts):
            obs.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915 (faithful C++ port; one long method)
        args = self._arguments
        results = self._results
        bt = args.barrier_type
        qassert.require(
            bt in (BarrierType.UpIn, BarrierType.UpOut, BarrierType.DownIn, BarrierType.DownOut),
            "Invalid barrier type",
        )
        assert args.barrier is not None
        assert args.rebate is not None
        assert args.exercise is not None

        t = self._t
        sqrt_t = math.sqrt(t)
        d_disc = self._domestic_ts.discount(t)
        f_disc = self._foreign_ts.discount(t)

        sigma_shift_vega = 0.0001
        sigma_shift_volga = 0.0001
        spot_shift_delta = 0.0001 * self._spot_fx.value()
        sigma_shift_vanna = 0.0001

        # Shiftable quotes used for the finite-difference Greeks.
        x0_quote = SimpleQuote(self._spot_fx.value())
        atm_vol_quote = SimpleQuote(self._atm_vol.value())

        today = ObservableSettings().evaluation_date_or_today()
        black_vol_ts = BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=Actual365Fixed(),
            volatility=atm_vol_quote,
        )
        stoch_process = BlackScholesMertonProcess(
            x0=x0_quote,
            dividend_ts=self._foreign_ts,
            risk_free_ts=self._domestic_ts,
            black_vol_ts=black_vol_ts,
        )
        engine_bs = AnalyticBarrierEngine(stoch_process)

        # Strikes from delta + ATM strike.
        bdc_atm = BlackDeltaCalculator(
            OptionType.Call, self._atm_vol.delta_type(), x0_quote.value(),
            d_disc, f_disc, self._atm_vol.value() * sqrt_t,
        )
        atm_strike = bdc_atm.atm_strike(self._atm_vol.atm_type())

        call25_vol = self._vol25_call.value()
        put25_vol = self._vol25_put.value()
        bdc_put25 = BlackDeltaCalculator(
            OptionType.Put, self._vol25_put.delta_type(), x0_quote.value(),
            d_disc, f_disc, put25_vol * sqrt_t,
        )
        put25_strike = bdc_put25.strike_from_delta(-0.25)
        bdc_call25 = BlackDeltaCalculator(
            OptionType.Call, self._vol25_call.delta_type(), x0_quote.value(),
            d_disc, f_disc, call25_vol * sqrt_t,
        )
        call25_strike = bdc_call25.strike_from_delta(0.25)

        # Vanna-Volga interpolated smile for the payoff strike.
        strikes = [put25_strike, atm_strike, call25_strike]
        vols = [put25_vol, self._atm_vol.value(), call25_vol]
        vanna_volga = VannaVolgaInterpolation(
            strikes, vols, x0_quote.value(), d_disc, f_disc, t
        )
        payoff = args.payoff
        assert isinstance(payoff, StrikedTypePayoff)
        strike_vol = vanna_volga(payoff.strike())

        forward = x0_quote.value() * f_disc / d_disc
        vanilla_option = black_formula(
            payoff.option_type(), payoff.strike(), forward, strike_vol * sqrt_t, d_disc
        )
        results.additional_results["Forward"] = forward
        results.additional_results["StrikeVol"] = strike_vol

        spot = x0_quote.value()
        barrier = args.barrier
        smile_van = self._bs_price_with_smile if self._adapt_van_delta else vanilla_option

        # Already-knocked branches.
        if spot >= barrier and bt == BarrierType.UpOut:
            results.value = 0.0
            results.additional_results["VanillaPrice"] = smile_van
            results.additional_results["BarrierInPrice"] = smile_van
            results.additional_results["BarrierOutPrice"] = 0.0
            return
        if spot >= barrier and bt == BarrierType.UpIn:
            results.value = smile_van
            results.additional_results["VanillaPrice"] = smile_van
            results.additional_results["BarrierInPrice"] = smile_van
            results.additional_results["BarrierOutPrice"] = 0.0
            return
        if spot <= barrier and bt == BarrierType.DownOut:
            results.value = 0.0
            results.additional_results["VanillaPrice"] = smile_van
            results.additional_results["BarrierInPrice"] = smile_van
            results.additional_results["BarrierOutPrice"] = 0.0
            return
        if spot <= barrier and bt == BarrierType.DownIn:
            results.value = smile_van
            results.additional_results["VanillaPrice"] = smile_van
            results.additional_results["BarrierInPrice"] = smile_van
            results.additional_results["BarrierOutPrice"] = 0.0
            return

        # Only the knock-out price is computed; in = vanilla - out.
        if bt == BarrierType.UpOut:
            barrier_type = bt
        elif bt == BarrierType.UpIn:
            barrier_type = BarrierType.UpOut
        elif bt == BarrierType.DownOut:
            barrier_type = bt
        else:
            barrier_type = BarrierType.DownOut

        barrier_option = BarrierOption(
            barrier_type, barrier, args.rebate, payoff, args.exercise
        )
        barrier_option.set_pricing_engine(engine_bs)

        # ``set_value`` on a registered quote invalidates the lazy cache via
        # the observer chain (quote -> process -> engine -> instrument), so
        # the next ``npv()`` recomputes — no explicit ``recalculate()`` needed
        # as in C++.
        price_bs = barrier_option.npv()
        price25_call_bs = black_formula(
            OptionType.Call, call25_strike, forward, self._atm_vol.value() * sqrt_t, d_disc
        )
        price25_put_bs = black_formula(
            OptionType.Put, put25_strike, forward, self._atm_vol.value() * sqrt_t, d_disc
        )
        price25_call_mkt = black_formula(
            OptionType.Call, call25_strike, forward, call25_vol * sqrt_t, d_disc
        )
        price25_put_mkt = black_formula(
            OptionType.Put, put25_strike, forward, put25_vol * sqrt_t, d_disc
        )

        # Analytical BS vega/vanna/volga for the three vanillas.
        norm = NormalDistribution()
        atm_v = atm_vol_quote.value()

        def greeks(strike: float) -> tuple[float, float, float]:
            d1 = (math.log(forward / strike) + 0.5 * atm_v**2 * t) / (atm_v * sqrt_t)
            vega = spot * norm(d1) * sqrt_t * f_disc
            vanna = vega / spot * (1.0 - d1 / (atm_v * sqrt_t))
            volga = vega * d1 * (d1 - atm_v * sqrt_t) / atm_v
            return vega, vanna, volga

        vega_atm, vanna_atm, volga_atm = greeks(atm_strike)
        vega_c, vanna_c, volga_c = greeks(call25_strike)
        vega_p, vanna_p, volga_p = greeks(put25_strike)

        # Finite-difference BS Greeks of the barrier. ``barrier_option.update()``
        # forces a recompute (mirrors C++ ``recalculate()``): a bare
        # ``set_value`` to a value equal to the current one is a no-op that
        # would NOT invalidate the lazy cache, so we invalidate explicitly.
        def reprice() -> float:
            barrier_option.update()
            return barrier_option.npv()

        # FD BS vega.
        atm_vol_quote.set_value(atm_v + sigma_shift_vega)
        vega_bar_bs = (reprice() - price_bs) / sigma_shift_vega
        atm_vol_quote.set_value(atm_v - sigma_shift_vega)  # set back

        # FD BS volga.
        atm_vol_quote.set_value(atm_v + sigma_shift_volga)
        price_bs2 = reprice()
        atm_vol_quote.set_value(atm_v + sigma_shift_volga + sigma_shift_vega)
        vega_bar_bs2 = (reprice() - price_bs2) / sigma_shift_vega
        volga_bar_bs = (vega_bar_bs2 - vega_bar_bs) / sigma_shift_volga
        atm_vol_quote.set_value(atm_v)  # set back

        # FD BS delta + vanna (cross-derivative).
        x0_quote.set_value(spot + spot_shift_delta)
        price_bs_delta1 = reprice()
        x0_quote.set_value(spot - spot_shift_delta)
        price_bs_delta2 = reprice()
        x0_quote.set_value(spot)  # set back
        delta_bar1 = (price_bs_delta1 - price_bs_delta2) / (2.0 * spot_shift_delta)

        atm_vol_quote.set_value(atm_v + sigma_shift_vanna)
        x0_quote.set_value(spot + spot_shift_delta)
        price_bs_delta1 = reprice()
        x0_quote.set_value(spot - spot_shift_delta)
        price_bs_delta2 = reprice()
        x0_quote.set_value(spot)  # set back
        delta_bar2 = (price_bs_delta1 - price_bs_delta2) / (2.0 * spot_shift_delta)

        vanna_bar_bs = (delta_bar2 - delta_bar1) / sigma_shift_vanna
        atm_vol_quote.set_value(atm_v - sigma_shift_vanna)  # set back

        # Solve A q = b for the hedge weights.
        a_mat = np.array(
            [
                [vega_atm, vega_c, vega_p],
                [vanna_atm, vanna_c, vanna_p],
                [volga_atm, volga_c, volga_p],
            ],
            dtype=float,
        )
        b_vec = np.array([vega_bar_bs, vanna_bar_bs, volga_bar_bs], dtype=float)
        q = np.linalg.solve(a_mat, b_vec)

        # Survival (no-touch) probability.
        cnd = CumulativeNormalDistribution()
        atm_value = self._atm_vol.value()
        mu = (
            self._domestic_ts.zero_rate(t, Compounding.Continuous).rate()
            - self._foreign_ts.zero_rate(t, Compounding.Continuous).rate()
            - atm_value**2 / 2.0
        )
        h2 = (math.log(barrier / spot) + mu * t) / (atm_value * sqrt_t)
        h2_prime = (math.log(spot / barrier) + mu * t) / (atm_value * sqrt_t)
        if bt in (BarrierType.UpIn, BarrierType.UpOut):
            prob_touch = cnd(h2_prime) + (barrier / spot) ** (2.0 * mu / atm_value**2) * cnd(-h2)
        else:
            prob_touch = cnd(-h2_prime) + (barrier / spot) ** (2.0 * mu / atm_value**2) * cnd(h2)
        p_survival = 1.0 - prob_touch

        lambda_ = p_survival
        adjust = q[1] * (price25_call_mkt - price25_call_bs) + q[2] * (
            price25_put_mkt - price25_put_bs
        )
        out_price = price_bs + lambda_ * adjust

        if self._adapt_van_delta:
            out_price += lambda_ * (self._bs_price_with_smile - vanilla_option)
            out_price = max(0.0, min(self._bs_price_with_smile, out_price))
            in_price = self._bs_price_with_smile - out_price
        else:
            out_price = max(0.0, min(vanilla_option, out_price))
            in_price = vanilla_option - out_price

        if bt in (BarrierType.DownOut, BarrierType.UpOut):
            results.value = out_price
        else:
            results.value = in_price
        results.additional_results["VanillaPrice"] = vanilla_option
        results.additional_results["BarrierInPrice"] = in_price
        results.additional_results["BarrierOutPrice"] = out_price
        results.additional_results["lambda"] = lambda_


__all__ = ["VannaVolgaBarrierEngine"]
