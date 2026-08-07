"""SuoWangDoubleBarrierEngine — Suo/Wang analytic double-barrier engine.

# C++ parity: ql/experimental/barrieroption/suowangdoublebarrierengine.{hpp,cpp}
# (v1.43).

Closed-form double-barrier pricing following Wulin Suo and Yong Wang,
"Barrier Option Pricing". Like the Ikeda-Kunitomo engine it sums an image
series indexed by ``n in [-series, series)``; unlike it, the summation runs
over a *half-open* range (the C++ loop is ``for (int n = -series_; n <
series_; n++)``), so ``series = 5`` means ten terms, not eleven.

Only ``KnockIn`` and ``KnockOut`` are supported.

Publishes four additional results: ``vanilla``, ``barrierOut``, ``barrierIn``
and ``rebateIn``.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class SuoWangDoubleBarrierEngine(
    GenericEngine[DoubleBarrierOptionArguments, OneAssetOptionResults]
):
    """Suo/Wang analytic double-barrier engine.

    # C++ parity: ``class SuoWangDoubleBarrierEngine``
    # (suowangdoublebarrierengine.hpp:42-62 + .cpp:28-168).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        series: int = 5,
    ) -> None:
        super().__init__(DoubleBarrierOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._series: int = series
        self._f: CumulativeNormalDistribution = CumulativeNormalDistribution()
        process.register_with(self)

    # --- helper accessors (mirror the C++ private helpers) -----------------

    def _exercise(self) -> Exercise:
        ex = self._arguments.exercise
        assert ex is not None
        return ex

    def _strike(self) -> float:
        """# C++ parity: ``strike()`` (suowangdoublebarrierengine.cpp:133-138)."""
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        return payoff.strike()

    def _residual_time(self) -> float:
        """# C++ parity: ``residualTime()`` (.cpp:140-142)."""
        return self._process.time(self._exercise().last_date())

    def _volatility(self) -> float:
        """# C++ parity: ``volatility()`` (.cpp:144-146)."""
        return self._process.black_volatility().black_vol_at_time(
            self._residual_time(), self._strike()
        )

    def _risk_free_rate(self) -> float:
        """# C++ parity: ``riskFreeRate()`` (.cpp:148-151)."""
        return self._process.risk_free_rate().zero_rate(
            self._residual_time(), Compounding.Continuous, Frequency.NoFrequency
        ).rate()

    def _risk_free_discount(self) -> float:
        """# C++ parity: ``riskFreeDiscount()`` (.cpp:153-155)."""
        return self._process.risk_free_rate().discount(self._residual_time())

    def _dividend_yield(self) -> float:
        """# C++ parity: ``dividendYield()`` (.cpp:157-160)."""
        return self._process.dividend_yield().zero_rate(
            self._residual_time(), Compounding.Continuous, Frequency.NoFrequency
        ).rate()

    def _dividend_discount(self) -> float:
        """# C++ parity: ``dividendDiscount()`` (.cpp:162-164)."""
        return self._process.dividend_yield().discount(self._residual_time())

    @staticmethod
    def _d(x: float, lambda_: float, sigma: float, t: float) -> float:
        """# C++ parity: ``SuoWangDoubleBarrierEngine::D`` (.cpp:166-168)."""
        return (math.log(x) + lambda_ * t) / (sigma * math.sqrt(t))

    def _triggered(self, underlying: float) -> bool:
        """# C++ parity: ``DoubleBarrierOption::engine::triggered``."""
        barrier_lo = self._arguments.barrier_lo
        barrier_hi = self._arguments.barrier_hi
        assert barrier_lo is not None
        assert barrier_hi is not None
        return underlying <= barrier_lo or underlying >= barrier_hi

    # --- engine -----------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915 — one-to-one with the C++ body
        """# C++ parity: ``SuoWangDoubleBarrierEngine::calculate`` (.cpp:34-130)."""
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        qassert.require(payoff.strike() > 0.0, "strike must be positive")

        k = payoff.strike()
        s = self._process.x0()
        qassert.require(s > 0.0, "negative or null underlying given")
        qassert.require(not self._triggered(s), "barrier touched")

        barrier_type = self._arguments.barrier_type
        qassert.require(
            barrier_type in (DoubleBarrierType.KnockOut, DoubleBarrierType.KnockIn),
            "only KnockIn and KnockOut options supported",
        )

        barrier_lo = self._arguments.barrier_lo
        barrier_hi = self._arguments.barrier_hi
        rebate = self._arguments.rebate
        assert barrier_lo is not None
        assert barrier_hi is not None
        assert rebate is not None

        low = barrier_lo
        high = barrier_hi
        k_up = min(high, k)
        k_down = max(low, k)
        t = self._residual_time()
        dd = self._risk_free_discount()
        df = self._dividend_discount()
        vol = self._volatility()
        mu = self._risk_free_rate() - self._dividend_yield() - vol * vol / 2.0
        sgn = 1.0 if mu > 0 else (-1.0 if mu < 0 else 0.0)
        # rebate
        r_l = rebate
        r_h = rebate

        # european option
        european_option = EuropeanOption(payoff, self._exercise())
        european_option.set_pricing_engine(AnalyticEuropeanEngine(self._process))
        european = european_option.npv()

        f = self._f
        d = self._d
        sqrt_t = math.sqrt(t)
        lh = low / high
        vol_sq = vol * vol
        lam = vol_sq + mu

        barrier_out = 0.0
        rebate_in = 0.0
        for n in range(-self._series, self._series):
            # # C++ parity note: the loop is ``n < series_``, i.e. the top
            # # index is excluded — the series is asymmetric by one term.
            lh2n = lh ** (2.0 * n)
            lh2nm1 = lh ** (2.0 * n - 1.0)
            d1 = d(s / high * lh2n, lam, vol, t)
            d2 = d1 - vol * sqrt_t
            g1 = d(high / s * lh2nm1, lam, vol, t)
            g2 = g1 - vol * sqrt_t
            h1 = d(s / high * lh2nm1, lam, vol, t)
            h2 = h1 - vol * sqrt_t
            k1 = d(low / s * lh2nm1, lam, vol, t)
            k2 = k1 - vol * sqrt_t
            d1_down = d(s / k_down * lh2n, lam, vol, t)
            d2_down = d1_down - vol * sqrt_t
            d1_up = d(s / k_up * lh2n, lam, vol, t)
            d2_up = d1_up - vol * sqrt_t
            k1_down = d((high * high) / (k_down * s) * lh2n, lam, vol, t)
            k2_down = k1_down - vol * sqrt_t
            k1_up = d((high * high) / (k_up * s) * lh2n, lam, vol, t)
            k2_up = k1_up - vol * sqrt_t

            hs_pow = (high / s) ** (2.0 * mu / vol_sq)
            lh_pow = lh ** (2.0 * n * mu / vol_sq)

            if payoff.option_type() == OptionType.Call:
                barrier_out += lh_pow * (
                    df * s * lh2n * (f(d1_down) - f(d1))
                    - dd * k * (f(d2_down) - f(d2))
                    - df * lh2n * high * high / s * hs_pow * (f(k1_down) - f(k1))
                    + dd * k * hs_pow * (f(k2_down) - f(k2))
                )
            elif payoff.option_type() == OptionType.Put:
                barrier_out += lh_pow * (
                    dd * k * (f(h2) - f(d2_up))
                    - df * s * lh2n * (f(h1) - f(d1_up))
                    - dd * k * hs_pow * (f(g2) - f(k2_up))
                    + df * lh2n * high * high / s * hs_pow * (f(g1) - f(k1_up))
                )
            else:  # pragma: no cover - OptionType has only Call/Put
                # # C++ parity: ``QL_FAIL("option type not recognized")``
                # # (suowangdoublebarrierengine.cpp:110).
                raise LibraryException("option type not recognized")

            v1 = d(high / s * (high / low) ** (2.0 * n), -mu, vol, t)
            v2 = d(high / s * (high / low) ** (2.0 * n), mu, vol, t)
            v3 = d(s / low * (high / low) ** (2.0 * n), -mu, vol, t)
            v4 = d(s / low * (high / low) ** (2.0 * n), mu, vol, t)
            rebate_in += dd * r_h * sgn * (
                lh ** (2.0 * n * mu / vol_sq) * f(sgn * v1) - hs_pow * f(-sgn * v2)
            ) + dd * r_l * sgn * (
                (low / s) ** (2.0 * mu / vol_sq) * f(-sgn * v3)
                - (high / low) ** (2.0 * n * mu / vol_sq) * f(sgn * v4)
            )

        # # C++ parity note: `rebateIn` is computed and published but never
        # # added to `value` — the "rebate paid at maturity" comment above the
        # # assignment (.cpp:121) describes an unimplemented branch. Verbatim.
        if barrier_type == DoubleBarrierType.KnockOut:
            self._results.value = barrier_out
        else:
            self._results.value = european - barrier_out
        self._results.additional_results["vanilla"] = european
        self._results.additional_results["barrierOut"] = barrier_out
        self._results.additional_results["barrierIn"] = european - barrier_out
        self._results.additional_results["rebateIn"] = rebate_in


__all__ = ["SuoWangDoubleBarrierEngine"]
