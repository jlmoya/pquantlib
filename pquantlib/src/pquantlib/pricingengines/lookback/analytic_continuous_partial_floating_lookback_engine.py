"""AnalyticContinuousPartialFloatingLookbackEngine — Heynen-Kat (1994).

# C++ parity:
# ql/pricingengines/lookback/analyticcontinuouspartialfloatinglookback.{hpp,cpp}
# (v1.43).

Closed form for a European partial-time floating-strike lookback under
Black-Scholes dynamics. Reference: Haug, "Option Pricing Formulas",
2nd ed., p.146.

The lookback window runs from inception to ``lookback_period_end``; the
strike struck at expiry is ``lambda_`` times the extremum realized over
that window (``lambda_ >= 1`` for calls, ``<= 1`` for puts, so the
scaling always works against the holder).

``A(eta)`` has two shapes, selected by whether the lookback window
reaches expiry (``lookbackPeriodEndTime() == residualTime()``):

* window ends BEFORE expiry — the full seven-term expression, which
  needs three bivariate normal CDFs at correlations
  ``+sqrt(t_lb/T)``, ``-sqrt(1 - t_lb/T)`` and ``-sqrt(t_lb/T)``;
* window runs TO expiry — the three-term "simpler calculation", whose
  single bivariate CDF degenerates to correlation ``+1``.

With ``lambda_ == 1`` the second shape reduces to the plain
Conze-Viswanathan floating lookback of
:class:`~pquantlib.pricingengines.lookback.analytic_continuous_floating_lookback_engine.AnalyticContinuousFloatingLookbackEngine`.

The engine fills ``results.value`` only — no greeks, exactly as C++.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.lookback_option import (
    ContinuousPartialFloatingLookbackOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import FloatingTypePayoff, OptionType
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticContinuousPartialFloatingLookbackEngine(
    GenericEngine[ContinuousPartialFloatingLookbackOptionArguments, OneAssetOptionResults]
):
    """Heynen-Kat partial-time floating-strike lookback engine.

    # C++ parity: ``AnalyticContinuousPartialFloatingLookbackEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(ContinuousPartialFloatingLookbackOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)
        self._cnd: CumulativeNormalDistribution = CumulativeNormalDistribution()

    # --- helpers (mirror C++) -------------------------------------------

    def _underlying(self) -> float:
        return self._process.x0()

    def _residual_time(self) -> float:
        ex = self._arguments.exercise
        assert ex is not None
        return self._process.time(ex.last_date())

    def _volatility(self) -> float:
        # C++ reads the vol AT THE RUNNING EXTREMUM, not at spot
        # (analyticcontinuouspartialfloatinglookback.cpp:62-64), and does
        # not pass an extrapolation flag.
        return self._process.black_volatility().black_vol_at_time(self._residual_time(), self._minmax())

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

    def _minmax(self) -> float:
        assert self._arguments.minmax is not None
        return self._arguments.minmax

    def _lambda(self) -> float:
        assert self._arguments.lambda_ is not None
        return self._arguments.lambda_

    def _lookback_period_end_time(self) -> float:
        return self._process.time(self._arguments.lookback_period_end)

    # --- closed form ------------------------------------------------------

    def _term_A(self, eta: float) -> float:  # noqa: N802
        """Haug p.146 ``A(eta)``.

        # C++ parity: ``AnalyticContinuousPartialFloatingLookbackEngine::A``
        # (analyticcontinuouspartialfloatinglookback.cpp:105-176), ported
        # term for term including the branch structure.
        """
        t = self._residual_time()
        t_lb = self._lookback_period_end_time()
        # Exact float comparison, as in C++: both sides come from the same
        # ``process.time(date)`` call, so they are bit-identical when the
        # lookback window ends on the exercise date.
        full_lookback_period = t_lb == t

        carry = self._risk_free_rate() - self._dividend_yield()
        vol = self._volatility()
        x = 2.0 * carry / (vol * vol)
        s = self._underlying() / self._minmax()

        ls = math.log(s)
        sd = self._std_deviation()
        d1 = ls / sd + 0.5 * (x + 1.0) * sd
        d2 = d1 - sd

        # C++ declares these on one line each: ``Real e1 = 0, e2 = 0;`` and
        # ``Real n4 = 0, n5 = 0, n6 = 0, n7 = 0;``.
        e1 = e2 = 0.0
        if not full_lookback_period:
            e1 = (carry + vol * vol / 2) * (t - t_lb) / (vol * math.sqrt(t - t_lb))
            e2 = e1 - vol * math.sqrt(t - t_lb)

        f1 = (ls + (carry + vol * vol / 2) * t_lb) / (vol * math.sqrt(t_lb))
        f2 = f1 - vol * math.sqrt(t_lb)

        lam = self._lambda()
        l1 = math.log(lam) / vol
        g1 = l1 / math.sqrt(t)

        n1 = self._cnd(eta * (d1 - g1))
        n2 = self._cnd(eta * (d2 - g1))

        cnbn1 = BivariateCumulativeNormalDistribution(1.0)
        cnbn2 = BivariateCumulativeNormalDistribution(0.0)
        cnbn3 = BivariateCumulativeNormalDistribution(-1.0)
        if not full_lookback_period:
            cnbn1 = BivariateCumulativeNormalDistribution(math.sqrt(t_lb / t))
            cnbn2 = BivariateCumulativeNormalDistribution(-math.sqrt(1 - t_lb / t))
            cnbn3 = BivariateCumulativeNormalDistribution(-math.sqrt(t_lb / t))

        n3 = cnbn1(
            eta * (-f1 + 2.0 * carry * math.sqrt(t_lb) / vol),
            eta * (-d1 + x * sd - g1),
        )
        n4 = n5 = n6 = n7 = 0.0
        if not full_lookback_period:
            g2 = l1 / math.sqrt(t - t_lb)
            n4 = cnbn2(-eta * (d1 + g1), eta * (e1 + g2))
            n5 = cnbn2(-eta * (d1 - g1), eta * (e1 - g2))
            n6 = cnbn3(eta * -f2, eta * (d2 - g1))
            n7 = self._cnd(eta * (e2 - g2))
        else:
            n4 = self._cnd(-eta * (d1 + g1))

        n8 = self._cnd(-eta * f1)
        pow_s = s ** (-x)
        pow_l = lam**x

        spot = self._underlying()
        div_disc = self._dividend_discount()
        rf_disc = self._risk_free_discount()
        minmax = self._minmax()

        if not full_lookback_period:
            return eta * (
                spot * div_disc * n1
                - lam * minmax * rf_disc * n2
                + spot * rf_disc * lam / x * (pow_s * n3 - div_disc / rf_disc * pow_l * n4)
                + spot * div_disc * n5
                + rf_disc * lam * minmax * n6
                - math.exp(-carry * (t - t_lb))
                * div_disc
                * (1 + 0.5 * vol * vol / carry)
                * lam
                * spot
                * n7
                * n8
            )
        # Simpler calculation: the window runs to expiry.
        return eta * (
            spot * div_disc * n1
            - lam * minmax * rf_disc * n2
            + spot * rf_disc * lam / x * (pow_s * n3 - div_disc / rf_disc * pow_l * n4)
        )

    # --- main entry point -------------------------------------------------

    def calculate(self) -> None:
        """Compute the partial-time floating-strike lookback NPV.

        # C++ parity:
        # ``AnalyticContinuousPartialFloatingLookbackEngine::calculate``.
        """
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, FloatingTypePayoff), "Non-floating payoff given")
        assert isinstance(payoff, FloatingTypePayoff)
        qassert.require(self._process.x0() > 0.0, "negative or null underlying")

        if payoff.option_type() == OptionType.Call:
            self._results.value = self._term_A(1.0)
        else:
            self._results.value = self._term_A(-1.0)


__all__ = ["AnalyticContinuousPartialFloatingLookbackEngine"]
