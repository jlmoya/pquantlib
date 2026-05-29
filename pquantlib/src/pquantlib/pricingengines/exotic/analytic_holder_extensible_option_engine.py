"""AnalyticHolderExtensibleOptionEngine — Haug 2007 closed-form.

# C++ parity:
# ql/pricingengines/exotic/analyticholderextensibleoptionengine.{hpp,cpp}
# (v1.42.1).

Haug 2007 ``Option Pricing Formulas`` closed-form for holder-extensible
options under Black-Scholes. The formula has the structure:

  result = BSM(S, X1, t1)
         +/- S * exp((b-r)*T2) * M2(...)
         -/+ X2 * exp(-r*T2) * M2(...)
         -/+ S * exp((b-r)*t1) * N2(...)
         +/- X1 * exp(-r*t1) * N2(...)
         - A * exp(-r*t1) * N2(...)

where ``M2`` and ``N2`` reduce to bivariate-normal CDFs over
rectangular regions, and the critical spots ``I1`` / ``I2`` solve
Newton-Raphson equations involving the BSM call/put value at expiry
``t1``.

The engine accepts only ``PlainVanillaPayoff`` (mirrors the C++
``dynamic_pointer_cast<PlainVanillaPayoff>``).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.holder_extensible_option import (
    HolderExtensibleOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistributionDr78,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_scholes_calculator import BlackScholesCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

_NEWTON_EPS = 0.001
_NEWTON_MAX_ITERATIONS = 1000
_NEWTON_DIVERGE_BOUND = 1e15


class AnalyticHolderExtensibleOptionEngine(
    GenericEngine[HolderExtensibleOptionArguments, OneAssetOptionResults]
):
    """Haug 2007 closed-form for holder-extensible options.

    # C++ parity: ``AnalyticHolderExtensibleOptionEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            HolderExtensibleOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    # --- helpers --------------------------------------------------------

    def _strike(self) -> float:
        payoff = self._arguments.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        return payoff.strike()

    def _option_type(self) -> OptionType:
        payoff = self._arguments.payoff
        assert isinstance(payoff, PlainVanillaPayoff)
        return payoff.option_type()

    def _first_expiry_date(self) -> Date:
        ex = self._arguments.exercise
        assert ex is not None
        return ex.last_date()

    def _first_expiry_time(self) -> float:
        return self._process.time(self._first_expiry_date())

    def _second_expiry_time(self) -> float:
        return self._process.time(self._arguments.second_expiry_date)

    def _volatility(self) -> float:
        return self._process.black_volatility().black_vol(
            self._first_expiry_date(), self._strike(), extrapolate=True
        )

    def _risk_free_rate(self) -> float:
        return self._process.risk_free_rate().zero_rate(
            self._first_expiry_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
        ).rate()

    def _dividend_yield(self) -> float:
        return self._process.dividend_yield().zero_rate(
            self._first_expiry_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
        ).rate()

    def _dividend_discount(self, t: float) -> float:
        return self._process.dividend_yield().discount(t)

    def _risk_free_discount(self, t: float) -> float:
        return self._process.risk_free_rate().discount(t)

    def _bs_calculator(self, spot: float, option_type: OptionType) -> BlackScholesCalculator:
        """BSM call/put value calculator over the *extension* tenor (T2 - t1)."""
        x2 = self._arguments.second_strike
        assert x2 is not None
        t = self._second_expiry_time() - self._first_expiry_time()
        vol = self._volatility() * math.sqrt(t)
        growth = self._dividend_discount(t)
        discount = self._risk_free_discount(t)
        payoff = PlainVanillaPayoff(option_type, x2)
        return BlackScholesCalculator(payoff, spot, growth, vol, discount)

    def _i1_call(self) -> float:
        """Critical-spot solver for the call-extension I1 leg.

        # C++ parity: ``AnalyticHolderExtensibleOptionEngine::I1Call`` —
        # ``yi = ci - A`` (da/ds = 0).
        """
        a = self._arguments.premium
        assert a is not None
        if a == 0.0:
            return 0.0
        sv = self._process.x0()
        bs = self._bs_calculator(sv, OptionType.Call)
        ci = bs.value()
        dc = bs.delta()
        yi = ci - a
        di = dc - 0.0
        for _ in range(_NEWTON_MAX_ITERATIONS):
            if abs(yi) <= _NEWTON_EPS:
                return sv
            sv = sv - yi / di
            if sv > _NEWTON_DIVERGE_BOUND:
                return math.inf
            bs = self._bs_calculator(sv, OptionType.Call)
            ci = bs.value()
            dc = bs.delta()
            yi = ci - a
            di = dc - 0.0
        return math.inf

    def _i2_call(self) -> float:
        """Critical-spot solver for the call-extension I2 leg.

        # C++ parity: ``AnalyticHolderExtensibleOptionEngine::I2Call`` —
        # ``yi = ci - A - Sv + X1`` (da/ds = 1).

        Returns +inf when ``A < X1 - X2*exp(-r*(T2-t1))``.
        """
        x1 = self._strike()
        x2 = self._arguments.second_strike
        a = self._arguments.premium
        assert x2 is not None
        assert a is not None
        t2 = self._second_expiry_time()
        t1 = self._first_expiry_time()
        r = self._risk_free_rate()
        val = x1 - x2 * math.exp(-r * (t2 - t1))
        if a < val:
            return math.inf
        sv = self._process.x0()
        bs = self._bs_calculator(sv, OptionType.Call)
        ci = bs.value()
        dc = bs.delta()
        yi = ci - a - sv + x1
        di = dc - 1.0
        for _ in range(_NEWTON_MAX_ITERATIONS):
            if abs(yi) <= _NEWTON_EPS:
                return sv
            sv = sv - yi / di
            if sv > _NEWTON_DIVERGE_BOUND:
                return math.inf
            bs = self._bs_calculator(sv, OptionType.Call)
            ci = bs.value()
            dc = bs.delta()
            yi = ci - a - sv + x1
            di = dc - 1.0
        return math.inf

    def _i1_put(self) -> float:
        """Put I1 — ``yi = pi - A + Sv - X1`` (da/ds = 1).

        # C++ parity: ``AnalyticHolderExtensibleOptionEngine::I1Put``.

        Note: the C++ algorithm has no convergence guard; when the
        Haug formula admits no real critical spot, the iteration
        runs Sv to +infinity. We detect that case explicitly and
        return ``+inf``, mirroring the C++ infinity-arithmetic
        behaviour.
        """
        x1 = self._strike()
        a = self._arguments.premium
        assert a is not None
        sv = self._process.x0()
        bs = self._bs_calculator(sv, OptionType.Put)
        pi = bs.value()
        dc = bs.delta()
        yi = pi - a + sv - x1
        di = dc - 1.0
        for _ in range(_NEWTON_MAX_ITERATIONS):
            if abs(yi) <= _NEWTON_EPS:
                return sv
            sv = sv - yi / di
            if sv > _NEWTON_DIVERGE_BOUND:
                return math.inf
            bs = self._bs_calculator(sv, OptionType.Put)
            pi = bs.value()
            dc = bs.delta()
            yi = pi - a + sv - x1
            di = dc - 1.0
        return math.inf

    def _i2_put(self) -> float:
        """Put I2 — ``yi = pi - A`` (da/ds = 0).

        Returns +inf when premium is 0.
        # C++ parity: ``AnalyticHolderExtensibleOptionEngine::I2Put``.
        """
        a = self._arguments.premium
        assert a is not None
        if a == 0.0:
            return math.inf
        sv = self._process.x0()
        bs = self._bs_calculator(sv, OptionType.Put)
        pi = bs.value()
        dc = bs.delta()
        yi = pi - a
        di = dc - 0.0
        for _ in range(_NEWTON_MAX_ITERATIONS):
            if abs(yi) <= _NEWTON_EPS:
                return sv
            if abs(di) < 1e-15:
                return math.inf
            sv = sv - yi / di
            if sv > _NEWTON_DIVERGE_BOUND or sv <= 0:
                return math.inf
            bs = self._bs_calculator(sv, OptionType.Put)
            pi = bs.value()
            dc = bs.delta()
            yi = pi - a
            di = dc - 0.0
        return math.inf

    @staticmethod
    def _m2(a: float, b: float, c: float, d: float, rho: float) -> float:
        """Rectangular bivariate-normal probability.

        # C++ parity: ``M2(a, b, c, d, rho) =
        # N2(b, d) - N2(a, d) - N2(b, c) + N2(a, c)`` with correlation
        # ``rho`` on each bivariate term.
        """
        cml = BivariateCumulativeNormalDistributionDr78(rho)
        return cml(b, d) - cml(a, d) - cml(b, c) + cml(a, c)

    @staticmethod
    def _n2(a: float, b: float) -> float:
        """1D-normal rectangular probability ``N(b) - N(a)``.

        # C++ parity: ``N2(a, b) = NormDist(b) - NormDist(a)``.
        """
        nd = CumulativeNormalDistribution()
        return nd(b) - nd(a)

    @staticmethod
    def _safe_log_div(spot: float, ix: float) -> float:
        """``log(spot/ix)``, returning -inf when ix is ``+inf``.

        # C++ parity: the Haug formula admits ``I1 = +inf`` (no
        # critical extension spot). C++ ``std::log(spot / inf)``
        # returns ``-inf`` directly; Python's ``math.log(0)`` raises.
        # Mirror the C++ behaviour explicitly.
        """
        if math.isinf(ix):
            return -math.inf
        return math.log(spot / ix)

    def _y1(self, option_type: OptionType) -> float:
        spot = self._process.x0()
        i2 = self._i2_call() if option_type == OptionType.Call else self._i2_put()
        b = self._risk_free_rate() - self._dividend_yield()
        vol = self._volatility()
        t1 = self._first_expiry_time()
        return (
            self._safe_log_div(spot, i2) + (b + vol * vol / 2.0) * t1
        ) / (vol * math.sqrt(t1))

    def _y2(self, option_type: OptionType) -> float:
        spot = self._process.x0()
        i1 = self._i1_call() if option_type == OptionType.Call else self._i1_put()
        b = self._risk_free_rate() - self._dividend_yield()
        vol = self._volatility()
        t1 = self._first_expiry_time()
        return (
            self._safe_log_div(spot, i1) + (b + vol * vol / 2.0) * t1
        ) / (vol * math.sqrt(t1))

    def _z1(self) -> float:
        spot = self._process.x0()
        x2 = self._arguments.second_strike
        assert x2 is not None
        b = self._risk_free_rate() - self._dividend_yield()
        vol = self._volatility()
        t2 = self._second_expiry_time()
        return (math.log(spot / x2) + (b + vol * vol / 2.0) * t2) / (vol * math.sqrt(t2))

    def _z2(self) -> float:
        spot = self._process.x0()
        x1 = self._strike()
        b = self._risk_free_rate() - self._dividend_yield()
        vol = self._volatility()
        t1 = self._first_expiry_time()
        return (math.log(spot / x1) + (b + vol * vol / 2.0) * t1) / (vol * math.sqrt(t1))

    # --- main ----------------------------------------------------------

    def calculate(self) -> None:
        """Compute the holder-extensible option NPV.

        # C++ parity: ``AnalyticHolderExtensibleOptionEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        s = self._process.x0()
        r = self._risk_free_rate()
        b = r - self._dividend_yield()
        x1 = self._strike()
        x2 = args.second_strike
        assert x2 is not None
        t2 = self._second_expiry_time()
        t1 = self._first_expiry_time()
        a = args.premium
        assert a is not None

        z1 = self._z1()
        z2 = self._z2()
        rho = math.sqrt(t1 / t2)

        payoff = args.payoff
        assert isinstance(payoff, PlainVanillaPayoff)
        vol = self._volatility()

        growth = self._dividend_discount(t1)
        discount = self._risk_free_discount(t1)
        minus_inf = -math.inf

        y1 = self._y1(payoff.option_type())
        y2 = self._y2(payoff.option_type())

        if payoff.option_type() == OptionType.Call:
            vanilla_call_payoff = PlainVanillaPayoff(OptionType.Call, x1)
            bsm_value = BlackScholesCalculator(
                vanilla_call_payoff, s, growth, vol * math.sqrt(t1), discount
            ).value()
            result = (
                bsm_value
                + s * math.exp((b - r) * t2) * self._m2(y1, y2, minus_inf, z1, rho)
                - x2 * math.exp(-r * t2) * self._m2(
                    y1 - vol * math.sqrt(t1),
                    y2 - vol * math.sqrt(t1),
                    minus_inf,
                    z1 - vol * math.sqrt(t2),
                    rho,
                )
                - s * math.exp((b - r) * t1) * self._n2(y1, z2)
                + x1 * math.exp(-r * t1) * self._n2(
                    y1 - vol * math.sqrt(t1), z2 - vol * math.sqrt(t1)
                )
                - a * math.exp(-r * t1) * self._n2(
                    y1 - vol * math.sqrt(t1), y2 - vol * math.sqrt(t1)
                )
            )
        else:
            vanilla_put_payoff = PlainVanillaPayoff(OptionType.Put, x1)
            bsm_value = BlackScholesCalculator(
                vanilla_put_payoff, s, growth, vol * math.sqrt(t1), discount
            ).value()
            result = (
                bsm_value
                - s * math.exp((b - r) * t2) * self._m2(y1, y2, minus_inf, -z1, rho)
                + x2 * math.exp(-r * t2) * self._m2(
                    y1 - vol * math.sqrt(t1),
                    y2 - vol * math.sqrt(t1),
                    minus_inf,
                    -z1 + vol * math.sqrt(t2),
                    rho,
                )
                + s * math.exp((b - r) * t1) * self._n2(z2, y2)
                - x1 * math.exp(-r * t1) * self._n2(
                    z2 - vol * math.sqrt(t1), y2 - vol * math.sqrt(t1)
                )
                - a * math.exp(-r * t1) * self._n2(
                    y1 - vol * math.sqrt(t1), y2 - vol * math.sqrt(t1)
                )
            )

        results.reset()
        results.value = result

    def update(self) -> None:
        self.notify_observers()


__all__ = ["AnalyticHolderExtensibleOptionEngine"]
