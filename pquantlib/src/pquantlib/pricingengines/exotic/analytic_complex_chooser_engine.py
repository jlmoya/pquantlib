"""AnalyticComplexChooserEngine — Rubinstein complex-chooser closed form.

# C++ parity:
# ql/pricingengines/exotic/analyticcomplexchooserengine.{hpp,cpp} (v1.43).

Rubinstein (1991) complex chooser, as reproduced in Haug, "Option
Pricing Formulas".  With ``T`` the time to the choosing date, ``Tc`` /
``Tp`` the *remaining* times from the choosing date to the call / put
maturities, and ``I`` the critical spot at which the call and put are
worth the same::

    value =  S e^{(b-r)Tc} M(d1,  y1;  rho1) - Xc e^{-r Tc} M(d2,  y1 - v sqrt(Tc);  rho1)
           - S e^{(b-r)Tp} M(-d1, -y2; rho2) + Xp e^{-r Tp} M(-d2, -y2 + v sqrt(Tp); rho2)

``I`` is found by Newton-Raphson on ``call(S) - put(S)`` with an
absolute tolerance of 1e-3 (C++ ``epsilon``).

The engine fills ``value`` only; every Greek accessor on the instrument
raises, as in C++.
"""

from __future__ import annotations

import math

from pquantlib.instruments.complex_chooser_option import (
    ComplexChooserOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistributionDr78,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_scholes_calculator import BlackScholesCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

# C++ ``Real epsilon = 0.001`` in ``criticalValue``.
_NEWTON_EPSILON = 0.001


class AnalyticComplexChooserEngine(GenericEngine[ComplexChooserOptionArguments, OneAssetOptionResults]):
    """Closed-form engine for :class:`ComplexChooserOption`.

    # C++ parity: ``AnalyticComplexChooserEngine``.

    Args:
        process: GBSM process of the single underlying.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(ComplexChooserOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    # --- helpers (C++ private members) ------------------------------------

    def _strike(self, option_type: OptionType) -> float:
        """# C++ parity: ``AnalyticComplexChooserEngine::strike``."""
        strike_call = self._arguments.strike_call
        strike_put = self._arguments.strike_put
        assert strike_call is not None
        assert strike_put is not None
        return strike_call if option_type == OptionType.Call else strike_put

    def _choosing_time(self) -> float:
        """# C++ parity: ``choosingTime``."""
        return self._process.time(self._arguments.choosing_date)

    def _put_maturity(self) -> float:
        """# C++ parity: ``putMaturity``."""
        exercise_put = self._arguments.exercise_put
        assert exercise_put is not None
        return self._process.time(exercise_put.last_date())

    def _call_maturity(self) -> float:
        """# C++ parity: ``callMaturity``."""
        exercise_call = self._arguments.exercise_call
        assert exercise_call is not None
        return self._process.time(exercise_call.last_date())

    def _volatility(self, t: float) -> float:
        """# C++ parity: ``volatility`` — always read at the CALL strike."""
        strike_call = self._arguments.strike_call
        assert strike_call is not None
        return self._process.black_volatility().black_vol_at_time(t, strike_call)

    def _dividend_yield(self, t: float) -> float:
        """# C++ parity: ``dividendYield``."""
        return (
            self._process.dividend_yield().zero_rate(t, Compounding.Continuous, Frequency.NoFrequency).rate()
        )

    def _dividend_discount(self, t: float) -> float:
        """# C++ parity: ``dividendDiscount``."""
        return self._process.dividend_yield().discount(t)

    def _risk_free_rate(self, t: float) -> float:
        """# C++ parity: ``riskFreeRate``."""
        return (
            self._process.risk_free_rate().zero_rate(t, Compounding.Continuous, Frequency.NoFrequency).rate()
        )

    def _risk_free_discount(self, t: float) -> float:
        """# C++ parity: ``riskFreeDiscount``."""
        return self._process.risk_free_rate().discount(t)

    def _bs_calculator(self, spot: float, option_type: OptionType) -> BlackScholesCalculator:
        """# C++ parity: ``AnalyticComplexChooserEngine::bsCalculator``.

        Note the C++ time argument: ``maturity - 2*T``, not ``maturity - T``
        (the comment in the C++ source says "TC-T" but the code subtracts
        ``2*T``). Ported verbatim — this is the ground truth the reference
        JSON was generated from. It also means a maturity of exactly twice
        the choosing time degenerates to ``t = 0`` and raises.
        """
        choosing = self._choosing_time()
        if option_type == OptionType.Call:
            t = self._call_maturity() - 2.0 * choosing
            vanilla_payoff = PlainVanillaPayoff(OptionType.Call, self._strike(OptionType.Call))
        else:
            t = self._put_maturity() - 2.0 * choosing
            vanilla_payoff = PlainVanillaPayoff(OptionType.Put, self._strike(OptionType.Put))
        # C++ comment: "QuantLib requires sigma * sqrt(t) rather than just
        # sigma/volatility".
        vol = self._volatility(t) * math.sqrt(t)
        growth = self._dividend_discount(t)
        discount = self._risk_free_discount(t)
        return BlackScholesCalculator(vanilla_payoff, spot, growth, vol, discount)

    def _critical_value(self) -> float:
        """# C++ parity: ``criticalValue`` — Newton-Raphson on call - put."""
        sv = self._process.x0()

        bs = self._bs_calculator(sv, OptionType.Call)
        ci = bs.value()
        dc = bs.delta()

        bs = self._bs_calculator(sv, OptionType.Put)
        pi = bs.value()
        dp = bs.delta()

        yi = ci - pi
        di = dc - dp

        while abs(yi) > _NEWTON_EPSILON:
            sv = sv - yi / di

            bs = self._bs_calculator(sv, OptionType.Call)
            ci = bs.value()
            dc = bs.delta()

            bs = self._bs_calculator(sv, OptionType.Put)
            pi = bs.value()
            dp = bs.delta()

            yi = ci - pi
            di = dc - dp

        return sv

    # --- engine -----------------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticComplexChooserEngine::calculate``."""
        args = self._arguments
        results = self._results

        s = self._process.x0()
        xc = args.strike_call
        xp = args.strike_put
        assert xc is not None
        assert xp is not None

        choosing = self._choosing_time()
        tc = self._call_maturity() - choosing
        tp = self._put_maturity() - choosing

        i = self._critical_value()

        b = self._risk_free_rate(choosing) - self._dividend_yield(choosing)
        v = self._volatility(choosing)
        d1 = (math.log(s / i) + (b + math.pow(v, 2) / 2.0) * choosing) / (v * math.sqrt(choosing))
        d2 = d1 - v * math.sqrt(choosing)

        b = self._risk_free_rate(choosing + tc) - self._dividend_yield(choosing + tc)
        v = self._volatility(tc)
        y1 = (math.log(s / xc) + (b + math.pow(v, 2) / 2.0) * tc) / (v * math.sqrt(tc))

        b = self._risk_free_rate(choosing + tp) - self._dividend_yield(choosing + tp)
        v = self._volatility(tp)
        y2 = (math.log(s / xp) + (b + math.pow(v, 2) / 2.0) * tp) / (v * math.sqrt(tp))

        rho1 = math.sqrt(choosing / tc)
        rho2 = math.sqrt(choosing / tp)

        b = self._risk_free_rate(choosing + tc) - self._dividend_yield(choosing + tc)
        r = self._risk_free_rate(choosing + tc)
        # NOTE: ``v`` here is still ``volatility(tp)`` from the y2 block —
        # C++ does not refresh it before the ``y1 - v*sqrt(tc)`` term. Kept
        # verbatim; with a flat vol surface the two are numerically equal.
        value = s * math.exp((b - r) * tc) * BivariateCumulativeNormalDistributionDr78(rho1)(
            d1, y1
        ) - xc * math.exp(-r * tc) * BivariateCumulativeNormalDistributionDr78(rho1)(
            d2, y1 - v * math.sqrt(tc)
        )

        b = self._risk_free_rate(choosing + tp) - self._dividend_yield(choosing + tp)
        r = self._risk_free_rate(choosing + tp)
        value -= s * math.exp((b - r) * tp) * BivariateCumulativeNormalDistributionDr78(rho2)(-d1, -y2)
        value += (
            xp
            * math.exp(-r * tp)
            * BivariateCumulativeNormalDistributionDr78(rho2)(-d2, -y2 + v * math.sqrt(tp))
        )

        results.value = value


__all__ = ["AnalyticComplexChooserEngine"]
