"""AnalyticContinuousPartialFixedLookbackEngine — Heynen-Kat (1994).

# C++ parity:
# ql/pricingengines/lookback/analyticcontinuouspartialfixedlookback.{hpp,cpp}
# (v1.43).

Closed form for a European partial-time fixed-strike lookback under
Black-Scholes dynamics. Reference: Haug, "Option Pricing Formulas",
2nd ed., p.148.

The lookback window STARTS at ``lookback_period_start`` (after
inception) and runs to expiry; the call pays the excess of the maximum
observed over that window above the fixed strike, the put the excess of
the strike above the minimum. The instrument carries no running
extremum at all.

``A(eta)`` keeps one shape throughout, but three of its bivariate
correlations depend on whether the window starts before expiry
(``lookbackPeriodStartTime() != residualTime()``); when it does not,
``e1``/``e2`` vanish and the correlations degenerate to ``-1``, ``0``
and ``0``.

The engine fills ``results.value`` only — no greeks, exactly as C++.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.lookback_option import (
    ContinuousPartialFixedLookbackOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
)
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


class AnalyticContinuousPartialFixedLookbackEngine(
    GenericEngine[ContinuousPartialFixedLookbackOptionArguments, OneAssetOptionResults]
):
    """Heynen-Kat partial-time fixed-strike lookback engine.

    # C++ parity: ``AnalyticContinuousPartialFixedLookbackEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(ContinuousPartialFixedLookbackOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)
        self._cnd: CumulativeNormalDistribution = CumulativeNormalDistribution()

    # --- helpers (mirror C++) -------------------------------------------

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
        # C++ reads the vol AT THE STRIKE, not at spot
        # (analyticcontinuouspartialfixedlookback.cpp:76-78), and does not
        # pass an extrapolation flag.
        return self._process.black_volatility().black_vol_at_time(self._residual_time(), self._strike())

    def _std_deviation(self) -> float:
        return self._volatility() * math.sqrt(self._residual_time())

    def _risk_free_rate(self) -> float:
        return (
            self._process.risk_free_rate()
            .zero_rate(
                self._residual_time(),
                Compounding.Continuous,
                Frequency.NoFrequency,
            )
            .rate()
        )

    def _risk_free_discount(self) -> float:
        return self._process.risk_free_rate().discount(self._residual_time())

    def _dividend_yield(self) -> float:
        return (
            self._process.dividend_yield()
            .zero_rate(
                self._residual_time(),
                Compounding.Continuous,
                Frequency.NoFrequency,
            )
            .rate()
        )

    def _dividend_discount(self) -> float:
        return self._process.dividend_yield().discount(self._residual_time())

    def _lookback_period_start_time(self) -> float:
        return self._process.time(self._arguments.lookback_period_start)

    # --- closed form ------------------------------------------------------

    def _term_A(self, eta: float) -> float:  # noqa: N802
        """Haug p.148 ``A(eta)``.

        # C++ parity: ``AnalyticContinuousPartialFixedLookbackEngine::A``
        # (analyticcontinuouspartialfixedlookback.cpp:107-159), ported
        # term for term including the branch structure.
        """
        t = self._residual_time()
        t_lb = self._lookback_period_start_time()
        # Exact float comparison, as in C++.
        different_start_of_lookback = t_lb != t

        carry = self._risk_free_rate() - self._dividend_yield()
        vol = self._volatility()
        x = 2.0 * carry / (vol * vol)
        strike = self._strike()
        s = self._underlying() / strike
        ls = math.log(s)
        sd = self._std_deviation()
        d1 = ls / sd + 0.5 * (x + 1.0) * sd
        d2 = d1 - sd

        # C++ declares these on one line: ``Real e1 = 0, e2 = 0;``.
        e1 = e2 = 0.0
        if different_start_of_lookback:
            e1 = (carry + vol * vol / 2) * (t - t_lb) / (vol * math.sqrt(t - t_lb))
            e2 = e1 - vol * math.sqrt(t - t_lb)

        f1 = (ls + (carry + vol * vol / 2) * t_lb) / (vol * math.sqrt(t_lb))
        f2 = f1 - vol * math.sqrt(t_lb)

        n1 = self._cnd(eta * d1)
        n2 = self._cnd(eta * d2)

        cnbn1 = BivariateCumulativeNormalDistribution(-1.0)
        cnbn2 = BivariateCumulativeNormalDistribution(0.0)
        cnbn3 = BivariateCumulativeNormalDistribution(0.0)
        if different_start_of_lookback:
            cnbn1 = BivariateCumulativeNormalDistribution(-math.sqrt(t_lb / t))
            cnbn2 = BivariateCumulativeNormalDistribution(math.sqrt(1 - t_lb / t))
            cnbn3 = BivariateCumulativeNormalDistribution(-math.sqrt(1 - t_lb / t))

        n3 = cnbn1(
            eta * (d1 - x * sd),
            eta * (-f1 + 2.0 * carry * math.sqrt(t_lb) / vol),
        )
        n4 = cnbn2(eta * e1, eta * d1)
        n5 = cnbn3(-eta * e1, eta * d1)
        n6 = cnbn1(eta * f2, -eta * d2)
        n7 = self._cnd(eta * f1)
        n8 = self._cnd(-eta * e2)

        pow_s = s ** (-x)
        carry_discount = math.exp(-carry * (t - t_lb))

        spot = self._underlying()
        div_disc = self._dividend_discount()
        rf_disc = self._risk_free_discount()

        return eta * (
            spot * div_disc * n1
            - strike * rf_disc * n2
            + spot * rf_disc / x * (-pow_s * n3 + div_disc / rf_disc * n4)
            - spot * div_disc * n5
            - strike * rf_disc * n6
            + carry_discount * div_disc * (1 - 0.5 * vol * vol / carry) * spot * n7 * n8
        )

    # --- main entry point -------------------------------------------------

    def calculate(self) -> None:
        """Compute the partial-time fixed-strike lookback NPV.

        # C++ parity:
        # ``AnalyticContinuousPartialFixedLookbackEngine::calculate``.
        # Note the asymmetric strike guard: a CALL admits a zero strike,
        # a PUT does not.
        """
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "Non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        qassert.require(self._process.x0() > 0.0, "negative or null underlying")

        if payoff.option_type() == OptionType.Call:
            qassert.require(payoff.strike() >= 0.0, "Strike must be positive or null")
            self._results.value = self._term_A(1.0)
        else:
            qassert.require(payoff.strike() > 0.0, "Strike must be positive")
            self._results.value = self._term_A(-1.0)


__all__ = ["AnalyticContinuousPartialFixedLookbackEngine"]
