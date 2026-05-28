"""AnalyticBarrierEngine — Reiner-Rubinstein closed-form barrier pricing.

# C++ parity:
# ql/pricingengines/barrier/analyticbarrierengine.{hpp,cpp} (v1.42.1).

Closed-form pricing for single-asset barrier options under Black-Scholes
dynamics, following Rich-Chesney / Reiner-Rubinstein:

  Reference: Haug "Option Pricing Formulas" pp.69 ff.

The engine assembles the option value from six helper terms A, B, C,
D, E, F. The combinatorics depend on:

  * option_type (Call/Put)
  * barrier_type (DownIn/UpIn/DownOut/UpOut)
  * strike vs barrier (a different combo applies if strike >= barrier)

The eight strike-vs-barrier branches mirror C++ exactly. Both
``rebate`` (paid at expiry on KO if knocked out, else at expiry on KI
if NEVER knocked in) and the underlying value are folded in through
the E/F helper terms.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.barrier_option import (
    BarrierOptionArguments,
    BarrierType,
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

_PHI: Final[CumulativeNormalDistribution] = CumulativeNormalDistribution()


class AnalyticBarrierEngine(
    GenericEngine[BarrierOptionArguments, OneAssetOptionResults]
):
    """Reiner-Rubinstein closed-form analytic barrier engine.

    # C++ parity: ``AnalyticBarrierEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(BarrierOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    # --- helper accessors (mirror C++) ----------------------------------

    def _underlying(self) -> float:
        return self._process.x0()

    def _strike(self) -> float:
        payoff = self._arguments.payoff
        assert isinstance(payoff, PlainVanillaPayoff)
        return payoff.strike()

    def _exercise(self) -> Exercise:
        ex = self._arguments.exercise
        assert ex is not None
        return ex

    def _volatility(self) -> float:
        return self._process.black_volatility().black_vol(
            self._exercise().last_date(), self._strike(), extrapolate=True
        )

    def _std_deviation(self) -> float:
        return math.sqrt(
            self._process.black_volatility().black_variance(
                self._exercise().last_date(), self._strike(), extrapolate=True
            )
        )

    def _barrier(self) -> float:
        assert self._arguments.barrier is not None
        return self._arguments.barrier

    def _rebate(self) -> float:
        assert self._arguments.rebate is not None
        return self._arguments.rebate

    def _risk_free_rate(self) -> float:
        return self._process.risk_free_rate().zero_rate(
            self._exercise().last_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
        ).rate()

    def _risk_free_discount(self) -> float:
        return self._process.risk_free_rate().discount(
            self._exercise().last_date()
        )

    def _dividend_yield(self) -> float:
        return self._process.dividend_yield().zero_rate(
            self._exercise().last_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
        ).rate()

    def _dividend_discount(self) -> float:
        return self._process.dividend_yield().discount(
            self._exercise().last_date()
        )

    def _mu(self) -> float:
        vol = self._volatility()
        return (self._risk_free_rate() - self._dividend_yield()) / (vol * vol) - 0.5

    def _mu_sigma(self) -> float:
        return (1.0 + self._mu()) * self._std_deviation()

    def _triggered(self, underlying: float) -> bool:
        """Check if the spot is on the wrong side of the barrier."""
        bt = self._arguments.barrier_type
        b = self._barrier()
        if bt in (BarrierType.DownIn, BarrierType.DownOut):
            return underlying <= b
        return underlying >= b

    # --- closed-form helper terms A..F (Haug p. 69) ---------------------

    def _term_A(self, phi: float) -> float:  # noqa: N802
        """A(phi) — vanilla-like term."""
        sd = self._std_deviation()
        x1 = math.log(self._underlying() / self._strike()) / sd + self._mu_sigma()
        n1 = _PHI(phi * x1)
        n2 = _PHI(phi * (x1 - sd))
        return phi * (
            self._underlying() * self._dividend_discount() * n1
            - self._strike() * self._risk_free_discount() * n2
        )

    def _term_B(self, phi: float) -> float:  # noqa: N802
        """B(phi) — vanilla-like term with barrier as strike."""
        sd = self._std_deviation()
        x2 = math.log(self._underlying() / self._barrier()) / sd + self._mu_sigma()
        n1 = _PHI(phi * x2)
        n2 = _PHI(phi * (x2 - sd))
        return phi * (
            self._underlying() * self._dividend_discount() * n1
            - self._strike() * self._risk_free_discount() * n2
        )

    def _term_C(self, eta: float, phi: float) -> float:  # noqa: N802
        """C(eta, phi) — image term @ barrier-reflected strike."""
        sd = self._std_deviation()
        hs = self._barrier() / self._underlying()
        pow_hs0 = hs ** (2.0 * self._mu())
        pow_hs1 = pow_hs0 * hs * hs
        y1 = (
            math.log(self._barrier() * hs / self._strike()) / sd + self._mu_sigma()
        )
        n1 = _PHI(eta * y1)
        n2 = _PHI(eta * (y1 - sd))
        # C++: when N == 0 the corresponding pow_HS might be Inf → product NaN.
        # Limit is 0.
        term1 = 0.0 if n1 == 0.0 else pow_hs1 * n1
        term2 = 0.0 if n2 == 0.0 else pow_hs0 * n2
        return phi * (
            self._underlying() * self._dividend_discount() * term1
            - self._strike() * self._risk_free_discount() * term2
        )

    def _term_D(self, eta: float, phi: float) -> float:  # noqa: N802
        """D(eta, phi) — image term @ barrier."""
        sd = self._std_deviation()
        hs = self._barrier() / self._underlying()
        pow_hs0 = hs ** (2.0 * self._mu())
        pow_hs1 = pow_hs0 * hs * hs
        y2 = math.log(self._barrier() / self._underlying()) / sd + self._mu_sigma()
        n1 = _PHI(eta * y2)
        n2 = _PHI(eta * (y2 - sd))
        term1 = 0.0 if n1 == 0.0 else pow_hs1 * n1
        term2 = 0.0 if n2 == 0.0 else pow_hs0 * n2
        return phi * (
            self._underlying() * self._dividend_discount() * term1
            - self._strike() * self._risk_free_discount() * term2
        )

    def _term_E(self, eta: float) -> float:  # noqa: N802
        """E(eta) — knock-in rebate (paid if NEVER touched)."""
        rebate = self._rebate()
        if rebate <= 0.0:
            return 0.0
        sd = self._std_deviation()
        pow_hs0 = (self._barrier() / self._underlying()) ** (2.0 * self._mu())
        x2 = math.log(self._underlying() / self._barrier()) / sd + self._mu_sigma()
        y2 = math.log(self._barrier() / self._underlying()) / sd + self._mu_sigma()
        n1 = _PHI(eta * (x2 - sd))
        n2 = _PHI(eta * (y2 - sd))
        term2 = 0.0 if n2 == 0.0 else pow_hs0 * n2
        return rebate * self._risk_free_discount() * (n1 - term2)

    def _term_F(self, eta: float) -> float:  # noqa: N802
        """F(eta) — knock-out rebate (paid on first touch)."""
        rebate = self._rebate()
        if rebate <= 0.0:
            return 0.0
        m = self._mu()
        vol = self._volatility()
        lam = math.sqrt(m * m + 2.0 * self._risk_free_rate() / (vol * vol))
        hs = self._barrier() / self._underlying()
        pow_hs_plus = hs ** (m + lam)
        pow_hs_minus = hs ** (m - lam)
        sigma_sqrt_t = self._std_deviation()
        z = math.log(self._barrier() / self._underlying()) / sigma_sqrt_t + lam * sigma_sqrt_t
        n1 = _PHI(eta * z)
        n2 = _PHI(eta * (z - 2.0 * lam * sigma_sqrt_t))
        term1 = 0.0 if n1 == 0.0 else pow_hs_plus * n1
        term2 = 0.0 if n2 == 0.0 else pow_hs_minus * n2
        return rebate * (term1 + term2)

    # --- main entry point -----------------------------------------------

    def calculate(self) -> None:
        """Branch on (option_type, barrier_type, strike vs barrier).

        # C++ parity: ``AnalyticBarrierEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        qassert.require(payoff.strike() > 0.0, "strike must be positive")

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "only european style option are supported",
        )

        spot = self._process.x0()
        qassert.require(spot > 0.0, "negative or null underlying given")
        qassert.require(not self._triggered(spot), "barrier touched")

        strike = payoff.strike()
        barrier = self._barrier()
        bt = args.barrier_type

        # Aliases for compactness, matching the C++ table:
        a = self._term_A
        b = self._term_B
        c = self._term_C
        d = self._term_D
        e = self._term_E
        f = self._term_F

        value: float
        if payoff.option_type() == OptionType.Call:
            if bt == BarrierType.DownIn:
                value = c(1, 1) + e(1) if strike >= barrier else a(1) - b(1) + d(1, 1) + e(1)
            elif bt == BarrierType.UpIn:
                value = a(1) + e(-1) if strike >= barrier else b(1) - c(-1, 1) + d(-1, 1) + e(-1)
            elif bt == BarrierType.DownOut:
                value = a(1) - c(1, 1) + f(1) if strike >= barrier else b(1) - d(1, 1) + f(1)
            else:  # UpOut
                value = (
                    f(-1)
                    if strike >= barrier
                    else a(1) - b(1) + c(-1, 1) - d(-1, 1) + f(-1)
                )
        elif bt == BarrierType.DownIn:  # Put
            value = (
                b(-1) - c(1, -1) + d(1, -1) + e(1)
                if strike >= barrier
                else a(-1) + e(1)
            )
        elif bt == BarrierType.UpIn:  # Put
            value = (
                a(-1) - b(-1) + d(-1, -1) + e(-1)
                if strike >= barrier
                else c(-1, -1) + e(-1)
            )
        elif bt == BarrierType.DownOut:  # Put
            value = (
                a(-1) - b(-1) + c(1, -1) - d(1, -1) + f(1)
                if strike >= barrier
                else f(1)
            )
        else:  # Put / UpOut
            value = (
                b(-1) - d(-1, -1) + f(-1)
                if strike >= barrier
                else a(-1) - c(-1, -1) + f(-1)
            )
        results.value = value


__all__ = ["AnalyticBarrierEngine"]
