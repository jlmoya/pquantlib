"""AnalyticContinuousFixedLookbackEngine — Conze-Viswanathan fixed-strike lookback.

# C++ parity:
# ql/pricingengines/lookback/analyticcontinuousfixedlookback.{hpp,cpp}
# (v1.43) — ``class AnalyticContinuousFixedLookbackEngine :
# public ContinuousFixedLookbackOption::engine``.

Closed form from Haug, *Option Pricing Formulas*, pp. 63-64. The
fixed-strike lookback pays at maturity

* Call: ``max(max_{0<=t<=T} S_t - K, 0)``
* Put:  ``max(K - min_{0<=t<=T} S_t, 0)``

where the running extremum observed so far is ``minmax``.

Branch selection (C++ ``calculate``):

* Call with ``strike <= minmax`` -> ``A(+1) + C(+1)``; otherwise ``B(+1)``.
* Put with ``strike >= minmax``  -> ``A(-1) + C(-1)``; otherwise ``B(-1)``.

Equality goes to the ``A + C`` side for both types. ``A`` and ``B`` are the
same expression with ``ss = S/minmax`` resp. ``ss = S/strike``, and ``C``
is the discounted intrinsic ``eta * rDF * (minmax - strike)``.

Three details the cross-validation pins:

* A call accepts ``strike == 0`` (``>= 0.0``); a put does not (``> 0.0``).
* ``lambda = 2 (r - q) / vol^2`` appears in a **denominator**. When
  ``r == q`` it is exactly zero and the C++ formula divides by zero,
  producing NaN. This port reproduces that rather than special-casing it —
  a caller who silently got a "nice" number instead would not learn that
  the closed form does not cover the driftless case.
* Every market quantity is read at ``residual_time = process.time(expiry)``,
  i.e. on the **risk-free** curve's day counter, including the volatility
  and the dividend yield.

Only ``value`` is assigned; every greek stays unset.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.lookback_option import (
    ContinuousFixedLookbackOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


def _cpp_divide(numerator: float, denominator: float) -> float:
    """IEEE-754 division with C++ semantics.

    C++ evaluates ``x / 0.0`` to +-infinity and ``0.0 / 0.0`` to NaN without
    raising; Python raises ``ZeroDivisionError`` instead. The ``lambda``
    denominator below is exactly zero whenever ``r == q``, and C++ answers NaN
    there — a port that raised, or that quietly special-cased the driftless
    case into a finite number, would not agree with the reference and would
    hide the fact that this closed form does not cover it.
    """
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return math.nan
    return math.copysign(math.inf, numerator) * math.copysign(1.0, denominator)


class AnalyticContinuousFixedLookbackEngine(
    GenericEngine[ContinuousFixedLookbackOptionArguments, OneAssetOptionResults]
):
    """Haug closed-form engine for continuous fixed-strike lookbacks.

    # C++ parity: ``AnalyticContinuousFixedLookbackEngine(process)``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            ContinuousFixedLookbackOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)
        self._f: CumulativeNormalDistribution = CumulativeNormalDistribution()

    # --- helper accessors (one-for-one with the C++ private methods) -----

    def _underlying(self) -> float:
        return self._process.x0()

    def _strike(self) -> float:
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "Non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        return payoff.strike()

    def _residual_time(self) -> float:
        ex = self._arguments.exercise
        assert ex is not None
        return self._process.time(ex.last_date())

    def _volatility(self) -> float:
        # C++ reads blackVol(residualTime(), strike()) -- a TIME, computed on
        # the risk-free curve's day counter, not on the vol surface's own.
        return self._process.black_volatility().black_vol_at_time(
            self._residual_time(), self._strike()
        )

    def _std_deviation(self) -> float:
        return self._volatility() * math.sqrt(self._residual_time())

    def _risk_free_rate(self) -> float:
        return (
            self._process.risk_free_rate()
            .zero_rate(
                self._residual_time(), Compounding.Continuous, Frequency.NoFrequency
            )
            .rate()
        )

    def _risk_free_discount(self) -> float:
        return self._process.risk_free_rate().discount(self._residual_time())

    def _dividend_yield(self) -> float:
        return (
            self._process.dividend_yield()
            .zero_rate(
                self._residual_time(), Compounding.Continuous, Frequency.NoFrequency
            )
            .rate()
        )

    def _dividend_discount(self) -> float:
        return self._process.dividend_yield().discount(self._residual_time())

    def _minmax(self) -> float:
        minmax = self._arguments.minmax
        assert minmax is not None
        return minmax

    # --- the A / B / C terms --------------------------------------------

    def _term(self, eta: float, denominator: float) -> float:
        """Shared body of C++'s ``A(eta)`` and ``B(eta)``.

        C++ writes the two out separately; they are identical except that
        ``A`` uses ``minmax`` where ``B`` uses ``strike``, both as the ratio
        denominator *and* as the second (discounted-strike) term. Kept as one
        helper with that quantity as the parameter.
        """
        vol = self._volatility()
        underlying = self._underlying()
        std_dev = self._std_deviation()
        rf_discount = self._risk_free_discount()
        div_discount = self._dividend_discount()

        lam = 2.0 * (self._risk_free_rate() - self._dividend_yield()) / (vol * vol)
        ss = underlying / denominator
        d1 = math.log(ss) / std_dev + 0.5 * (lam + 1.0) * std_dev
        n1 = self._f(eta * d1)
        n2 = self._f(eta * (d1 - std_dev))
        n3 = self._f(eta * (d1 - lam * std_dev))
        # C++ computes N4 = f_(eta*d1), i.e. literally the same value as N1.
        n4 = self._f(eta * d1)
        pow_ss = ss ** (-lam)
        # `lam` is zero exactly when r == q; C++ divides by it unguarded, so
        # _cpp_divide reproduces the resulting NaN / infinity rather than hiding
        # it. The division is kept innermost, exactly where C++ has it, so the
        # rounding of the surrounding products is unchanged.
        return eta * (
            underlying * div_discount * n1
            - denominator * rf_discount * n2
            - underlying
            * rf_discount
            * _cpp_divide(pow_ss * n3 - div_discount * n4 / rf_discount, lam)
        )

    def _term_a(self, eta: float) -> float:
        """# C++ parity: ``AnalyticContinuousFixedLookbackEngine::A``."""
        return self._term(eta, self._minmax())

    def _term_b(self, eta: float) -> float:
        """# C++ parity: ``AnalyticContinuousFixedLookbackEngine::B``."""
        return self._term(eta, self._strike())

    def _term_c(self, eta: float) -> float:
        """# C++ parity: ``AnalyticContinuousFixedLookbackEngine::C``."""
        return eta * (self._risk_free_discount() * (self._minmax() - self._strike()))

    # --- main entry point ------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticContinuousFixedLookbackEngine::calculate``."""
        args = self._arguments
        results = self._results

        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "Non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        qassert.require(self._process.x0() > 0.0, "negative or null underlying")

        strike = payoff.strike()

        if payoff.option_type() == OptionType.Call:
            qassert.require(strike >= 0.0, "Strike must be positive or null")
            if strike <= self._minmax():
                results.value = self._term_a(1.0) + self._term_c(1.0)
            else:
                results.value = self._term_b(1.0)
        elif payoff.option_type() == OptionType.Put:
            qassert.require(strike > 0.0, "Strike must be positive")
            if strike >= self._minmax():
                results.value = self._term_a(-1.0) + self._term_c(-1.0)
            else:
                results.value = self._term_b(-1.0)
        else:
            qassert.fail("Unknown type")


__all__ = ["AnalyticContinuousFixedLookbackEngine"]
