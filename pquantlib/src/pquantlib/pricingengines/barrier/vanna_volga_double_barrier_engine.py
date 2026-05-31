"""VannaVolgaDoubleBarrierEngine — Vanna-Volga FX double-barrier engine.

# C++ parity: ql/experimental/barrieroption/vannavolgadoublebarrierengine.hpp
#             (v1.42.1).

Double-barrier analogue of :class:`VannaVolgaBarrierEngine`. The C++ class
is templated on the underlying double-barrier engine
(``VannaVolgaDoubleBarrierEngine<DoubleBarrierEngine>``); the Python port
renders that template parameter as an ``engine_factory`` callable that
builds a BS double-barrier engine from ``(process, series)``. It defaults
to :class:`AnalyticDoubleBarrierEngine`.

The survival probability uses the double-no-touch series expansion.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.barrieroption.vanna_volga_interpolation import (
    VannaVolgaInterpolation,
)
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.barrier.analytic_double_barrier_engine import (
    AnalyticDoubleBarrierEngine,
)
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.black_delta_calculator import (
    BlackDeltaCalculator,
)
from pquantlib.processes.black_scholes_merton_process import (
    BlackScholesMertonProcess,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
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


def _default_engine_factory(
    process: GeneralizedBlackScholesProcess, series: int
) -> PricingEngine:
    return AnalyticDoubleBarrierEngine(process, series)


class VannaVolgaDoubleBarrierEngine(
    GenericEngine[DoubleBarrierOptionArguments, OneAssetOptionResults]
):
    """Vanna-Volga double-barrier FX engine.

    # C++ parity: ``VannaVolgaDoubleBarrierEngine<DoubleBarrierEngine>``.

    Args:
        atm_vol / vol25_put / vol25_call: the three FX vol quotes.
        spot_fx: spot FX rate quote.
        domestic_ts / foreign_ts: domestic / foreign yield curves.
        adapt_van_delta: enable Vanilla-delta adaptation.
        bs_price_with_smile: BS vanilla price under the smile.
        series: terms in the double-no-touch series (default 5).
        engine_factory: builds the BS double-barrier engine from
            ``(process, series)`` (renders the C++ template param).
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
        series: int = 5,
        engine_factory: Callable[
            [GeneralizedBlackScholesProcess, int], PricingEngine
        ] = _default_engine_factory,
    ) -> None:
        super().__init__(DoubleBarrierOptionArguments(), OneAssetOptionResults())
        self._atm_vol = atm_vol
        self._vol25_put = vol25_put
        self._vol25_call = vol25_call
        self._t = atm_vol.maturity()
        self._spot_fx = spot_fx
        self._domestic_ts = domestic_ts
        self._foreign_ts = foreign_ts
        self._adapt_van_delta = adapt_van_delta
        self._bs_price_with_smile = bs_price_with_smile
        self._series = series
        self._engine_factory = engine_factory

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
            bt in (DoubleBarrierType.KnockIn, DoubleBarrierType.KnockOut),
            "Only same type barrier supported",
        )
        assert args.barrier_lo is not None
        assert args.barrier_hi is not None
        assert args.rebate is not None
        assert args.exercise is not None

        t = self._t
        sqrt_t = math.sqrt(t)
        d_disc = self._domestic_ts.discount(t)
        f_disc = self._foreign_ts.discount(t)

        sigma_shift_vega = 0.001
        sigma_shift_volga = 0.0001
        spot_shift_delta = 0.0001 * self._spot_fx.value()
        sigma_shift_vanna = 0.0001

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
        engine_bs = self._engine_factory(stoch_process, self._series)

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

        # C++ uses (spot, foreignTS, foreignTS, T) here (a known quirk that
        # makes the forward in the smile interpolation == spot). Preserved.
        strikes = [put25_strike, atm_strike, call25_strike]
        vols = [put25_vol, self._atm_vol.value(), call25_vol]
        vanna_volga = VannaVolgaInterpolation(
            strikes, vols, x0_quote.value(), f_disc, f_disc, t
        )
        payoff = args.payoff
        assert isinstance(payoff, StrikedTypePayoff)
        strike_vol = vanna_volga(payoff.strike())

        forward = x0_quote.value() * f_disc / d_disc
        vanilla_option = black_formula(
            payoff.option_type(), payoff.strike(), forward, strike_vol * sqrt_t, d_disc
        )

        spot = x0_quote.value()
        bhi = args.barrier_hi
        blo = args.barrier_lo
        smile_van = self._bs_price_with_smile if self._adapt_van_delta else vanilla_option

        if (spot > bhi or spot < blo) and bt == DoubleBarrierType.KnockOut:
            results.value = 0.0
            results.additional_results["VanillaPrice"] = smile_van
            results.additional_results["BarrierInPrice"] = smile_van
            results.additional_results["BarrierOutPrice"] = 0.0
            return
        if (spot > bhi or spot < blo) and bt == DoubleBarrierType.KnockIn:
            results.value = smile_van
            results.additional_results["VanillaPrice"] = smile_van
            results.additional_results["BarrierInPrice"] = smile_van
            results.additional_results["BarrierOutPrice"] = 0.0
            return

        barrier_option = DoubleBarrierOption(
            DoubleBarrierType.KnockOut, blo, bhi, args.rebate, payoff, args.exercise
        )
        barrier_option.set_pricing_engine(engine_bs)
        price_bs = barrier_option.npv()

        price_atm_call_bs = black_formula(
            OptionType.Call, atm_strike, forward, self._atm_vol.value() * sqrt_t, d_disc
        )
        price25_call_bs = black_formula(
            OptionType.Call, call25_strike, forward, self._atm_vol.value() * sqrt_t, d_disc
        )
        price25_put_bs = black_formula(
            OptionType.Put, put25_strike, forward, self._atm_vol.value() * sqrt_t, d_disc
        )
        price_atm_call_mkt = black_formula(
            OptionType.Call, atm_strike, forward, self._atm_vol.value() * sqrt_t, d_disc
        )
        price25_call_mkt = black_formula(
            OptionType.Call, call25_strike, forward, call25_vol * sqrt_t, d_disc
        )
        price25_put_mkt = black_formula(
            OptionType.Put, put25_strike, forward, put25_vol * sqrt_t, d_disc
        )

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

        # FD BS Greeks of the barrier. ``barrier_option.update()`` forces a
        # recompute (mirrors C++ ``recalculate()``) so an unchanged
        # ``set_value`` cannot leave a stale lazy cache in place.
        def reprice() -> float:
            barrier_option.update()
            return barrier_option.npv()

        # FD BS vega.
        atm_vol_quote.set_value(atm_v + sigma_shift_vega)
        vega_bar_bs = (reprice() - price_bs) / sigma_shift_vega
        atm_vol_quote.set_value(atm_v - sigma_shift_vega)

        # FD BS volga.
        atm_vol_quote.set_value(atm_v + sigma_shift_volga)
        price_bs2 = reprice()
        atm_vol_quote.set_value(atm_v + sigma_shift_volga + sigma_shift_vega)
        vega_bar_bs2 = (reprice() - price_bs2) / sigma_shift_vega
        volga_bar_bs = (vega_bar_bs2 - vega_bar_bs) / sigma_shift_volga
        atm_vol_quote.set_value(atm_v)

        # FD BS delta + vanna.
        x0_quote.set_value(spot + spot_shift_delta)
        price_bs_delta1 = reprice()
        x0_quote.set_value(spot - spot_shift_delta)
        price_bs_delta2 = reprice()
        x0_quote.set_value(spot)
        delta_bar1 = (price_bs_delta1 - price_bs_delta2) / (2.0 * spot_shift_delta)

        atm_vol_quote.set_value(atm_v + sigma_shift_vanna)
        x0_quote.set_value(spot + spot_shift_delta)
        price_bs_delta1 = reprice()
        x0_quote.set_value(spot - spot_shift_delta)
        price_bs_delta2 = reprice()
        x0_quote.set_value(spot)
        delta_bar2 = (price_bs_delta1 - price_bs_delta2) / (2.0 * spot_shift_delta)

        vanna_bar_bs = (delta_bar2 - delta_bar1) / sigma_shift_vanna
        atm_vol_quote.set_value(atm_v - sigma_shift_vanna)

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

        # Double-no-touch survival probability (series expansion).
        atm_value = self._atm_vol.value()
        theta_tilt_minus = (
            (
                self._domestic_ts.zero_rate(t, Compounding.Continuous).rate()
                - self._foreign_ts.zero_rate(t, Compounding.Continuous).rate()
            )
            / atm_value
            - atm_value / 2.0
        ) * sqrt_t
        h = 1.0 / atm_value * math.log(bhi / spot) / sqrt_t
        ell = 1.0 / atm_value * math.log(blo / spot) / sqrt_t
        cnd = CumulativeNormalDistribution()

        double_no_touch = 0.0
        for j in range(-self._series, self._series):
            e_minus = 2 * j * (h - ell) - theta_tilt_minus
            double_no_touch += np.exp(-2.0 * j * theta_tilt_minus * (h - ell)) * (
                cnd(h + e_minus) - cnd(ell + e_minus)
            ) - np.exp(
                -2.0 * j * theta_tilt_minus * (h - ell) + 2.0 * theta_tilt_minus * h
            ) * (cnd(h - 2.0 * h + e_minus) - cnd(ell - 2.0 * h + e_minus))

        p_survival = double_no_touch
        lambda_ = p_survival
        adjust = (
            q[0] * (price_atm_call_mkt - price_atm_call_bs)
            + q[1] * (price25_call_mkt - price25_call_bs)
            + q[2] * (price25_put_mkt - price25_put_bs)
        )
        out_price = price_bs + lambda_ * adjust

        if self._adapt_van_delta:
            out_price += lambda_ * (self._bs_price_with_smile - vanilla_option)
            out_price = max(0.0, min(self._bs_price_with_smile, out_price))
            in_price = self._bs_price_with_smile - out_price
        else:
            out_price = max(0.0, min(vanilla_option, out_price))
            in_price = vanilla_option - out_price

        if bt == DoubleBarrierType.KnockOut:
            results.value = out_price
        else:
            results.value = in_price
        results.additional_results["VanillaPrice"] = vanilla_option
        results.additional_results["BarrierInPrice"] = in_price
        results.additional_results["BarrierOutPrice"] = out_price
        results.additional_results["lambda"] = lambda_


__all__ = ["VannaVolgaDoubleBarrierEngine"]
