# ruff: noqa: N803, N806
# Reason: math-symbol var names (S, X, T, U, L, mu, lambda1, lambda2,
# d1..d4, e1..e4, sigma, sqrtT) match Hart-Ross 1994 / Haug "Complete
# Guide" 2nd ed p. 165 notation 1:1 with the C++ source; replacing
# them with snake_case would break the parity with
# ``analyticsoftbarrierengine.cpp``.
"""AnalyticSoftBarrierEngine — Hart-Ross 1994 closed form.

# C++ parity:
# ql/pricingengines/barrier/analyticsoftbarrierengine.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for soft-barrier European options under
Black-Scholes dynamics.

Reference: Haug "The Complete Guide to Option Pricing Formulas" 2nd
ed, p. 165 (formula originally from Hart & Ross 1994).

A soft barrier is a *band* of barriers: the option is knocked in/out
proportionally to where the underlying spot lies within the band
``[L, U]``. The closed form integrates a continuum of standard
barrier options over that band, yielding the eight-term ``w(S, X, T,
U, L)`` formula.

Special cases handled:

* Already-knocked-in (e.g. UpIn with S >= U): price = vanilla
  equivalent (BlackCalculator on the BSM-forward).
* Already-knocked-out: price = 0.
* Degenerate U == L (single hard barrier): delegates to the standard
  ``AnalyticBarrierEngine``.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.experimental.barrieroption.soft_barrier_option import (
    SoftBarrierOptionArguments,
)
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.barrier.analytic_barrier_engine import (
    AnalyticBarrierEngine,
)
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticSoftBarrierEngine(
    GenericEngine[SoftBarrierOptionArguments, OneAssetOptionResults]
):
    """Hart-Ross 1994 closed-form soft-barrier engine.

    # C++ parity: ``AnalyticSoftBarrierEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            SoftBarrierOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._cnd: CumulativeNormalDistribution = (
            CumulativeNormalDistribution()
        )
        process.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticSoftBarrierEngine::calculate``
        (.cpp:43-103).
        """
        args = self._arguments
        results = self._results

        # Market data
        S = self._underlying()
        X = self._strike()
        r = self._risk_free_rate()
        q = self._dividend_yield()
        sigma = self._volatility()

        # Barrier band
        assert args.barrier_hi is not None
        assert args.barrier_lo is not None
        assert args.barrier_type is not None
        U = args.barrier_hi
        L = args.barrier_lo
        barrier_type = args.barrier_type

        # Stability tweak for r/q to avoid the mu = 0.5 singularity at
        # b == 0. C++ guards r = q + 1e-6 (.cpp:58-61).
        epsilon = 1e-6
        if abs(r - q) < 1e-10:
            r = q + epsilon

        T = self._residual_time()
        payoff = args.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        option_type = payoff.option_type()
        eta = 1 if option_type == OptionType.Call else -1
        b = r - q  # cost of carry

        self._validate_inputs(
            S=S,
            X=X,
            r=r,
            q=q,
            T=T,
            U=U,
            L=L,
            option_type=option_type,
            barrier_type=barrier_type,
            sigma=sigma,
        )

        is_knocked_in = (
            barrier_type == BarrierType.DownIn and S <= L
        ) or (barrier_type == BarrierType.UpIn and S >= U)
        is_knocked_out = (
            barrier_type == BarrierType.DownOut and S <= L
        ) or (barrier_type == BarrierType.UpOut and S >= U)
        # Haug formula breaks at U == L; delegate to standard barrier
        # engine instead (.cpp:77).
        is_single_barrier = abs(U - L) < 1e-4

        # edge case 1: already knocked in -> vanilla
        if is_knocked_in:
            results.value = self._vanilla_equivalent()
            return
        # edge case 2: already knocked out -> 0
        if is_knocked_out:
            results.value = 0.0
            return
        # edge case 3: hard-barrier degenerate -> standard barrier engine
        if is_single_barrier:
            results.value = self._standard_barrier_equivalent()
            return

        # Soft-barrier closed form.
        w = self._knock_in_value(
            S=S,
            X=X,
            sigma=sigma,
            T=T,
            U=U,
            L=L,
            b=b,
            option_type=option_type,
            eta=eta,
        )
        if barrier_type in (BarrierType.DownIn, BarrierType.UpIn):
            results.value = w
        else:
            results.value = self._vanilla_equivalent() - w

    # ------------------------------------------------------------------
    # Hart-Ross knock-in closed form
    # ------------------------------------------------------------------
    def _knock_in_value(
        self,
        *,
        S: float,
        X: float,
        sigma: float,
        T: float,
        U: float,
        L: float,
        b: float,
        option_type: OptionType,
        eta: int,
    ) -> float:
        """Hart-Ross ``w`` term.

        # C++ parity: ``knockInValue`` (.cpp:107-161).
        """
        _ = option_type  # used via eta passed in
        mu = (b + 0.5 * sigma * sigma) / (sigma * sigma)
        sqrtT = math.sqrt(T)
        lambda1 = math.exp(
            -0.5 * sigma * sigma * T * (mu + 0.5) * (mu - 0.5)
        )
        lambda2 = math.exp(
            -0.5 * sigma * sigma * T * (mu - 0.5) * (mu - 1.5)
        )
        SX = S * X
        logU2_SX = math.log((U * U) / SX)
        logL2_SX = math.log((L * L) / SX)

        d1 = logU2_SX / (sigma * sqrtT) + mu * sigma * sqrtT
        d2 = d1 - (mu + 0.5) * sigma * sqrtT
        d3 = logU2_SX / (sigma * sqrtT) + (mu - 1) * sigma * sqrtT
        d4 = d3 - (mu - 0.5) * sigma * sqrtT

        e1 = logL2_SX / (sigma * sqrtT) + mu * sigma * sqrtT
        e2 = e1 - (mu + 0.5) * sigma * sqrtT
        e3 = logL2_SX / (sigma * sqrtT) + (mu - 1) * sigma * sqrtT
        e4 = e3 - (mu - 0.5) * sigma * sqrtT

        N = self._cnd
        Nd1 = N(eta * d1)
        Nd2 = N(eta * d2)
        Nd3 = N(eta * d3)
        Nd4 = N(eta * d4)
        Ne1 = N(eta * e1)
        Ne2 = N(eta * e2)
        Ne3 = N(eta * e3)
        Ne4 = N(eta * e4)

        r = self._risk_free_rate()

        # term 1
        term1 = (
            eta
            * S
            * math.exp((b - r) * T)
            * (S ** (-2.0 * mu))
            * (SX ** (mu + 0.5))
            / (2.0 * (mu + 0.5))
        )
        term1 *= (
            ((U * U / SX) ** (mu + 0.5)) * Nd1
            - lambda1 * Nd2
            - ((L * L / SX) ** (mu + 0.5)) * Ne1
            + lambda1 * Ne2
        )

        # term 2
        term2 = (
            eta
            * X
            * math.exp(-r * T)
            * (S ** (-2.0 * (mu - 1)))
            * (SX ** (mu - 0.5))
            / (2.0 * (mu - 0.5))
        )
        term2 *= (
            ((U * U / SX) ** (mu - 0.5)) * Nd3
            - lambda2 * Nd4
            - ((L * L / SX) ** (mu - 0.5)) * Ne3
            + lambda2 * Ne4
        )

        return (1.0 / (U - L)) * (term1 - term2)

    # ------------------------------------------------------------------
    # Validation / equivalents
    # ------------------------------------------------------------------
    def _validate_inputs(
        self,
        *,
        S: float,
        X: float,
        r: float,
        q: float,
        T: float,
        U: float,
        L: float,
        option_type: OptionType,
        barrier_type: BarrierType,
        sigma: float,
    ) -> None:
        """# C++ parity: ``validateInputs`` (.cpp:165-189)."""
        qassert.require(S > 0.0, "Spot price must be > 0")
        qassert.require(X > 0.0, "Strike price must be > 0")
        qassert.require(T > 0.0, "Option must have time to maturity > 0")
        qassert.require(sigma > 0.0, "Volatility must be > 0")
        qassert.require(
            option_type in (OptionType.Call, OptionType.Put),
            "Invalid option type",
        )
        qassert.require(
            -0.05 <= r <= 1.0, "Interest rate must be between -5% and 100%"
        )
        qassert.require(
            -0.1 <= q <= 1.0, "Dividend yield must be between -10% and 100%"
        )
        qassert.require(
            barrier_type
            in (
                BarrierType.DownIn,
                BarrierType.DownOut,
                BarrierType.UpIn,
                BarrierType.UpOut,
            ),
            "Invalid barrier type",
        )
        qassert.require(U > 0.0 and L > 0.0, "Barrier levels must be positive")
        qassert.require(
            U >= L, "Upper barrier must be greater than or equal to lower barrier"
        )

    def _vanilla_equivalent(self) -> float:
        """Vanilla NPV via BlackCalculator on the BSM forward.

        # C++ parity: ``vanillaEquivalent`` (.cpp:240-247).
        """
        args = self._arguments
        assert args.payoff is not None
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        forward = (
            self._underlying()
            * self._dividend_discount()
            / self._risk_free_discount()
        )
        bc = BlackCalculator(
            args.payoff, forward, self._std_deviation(), self._risk_free_discount()
        )
        return max(bc.value(), 0.0)

    def _standard_barrier_equivalent(self) -> float:
        """Hard-barrier degenerate case (U == L).

        # C++ parity: ``standardBarrierEquivalent`` (.cpp:250-281).
        Uses the standard ``BarrierOption`` + ``AnalyticBarrierEngine``
        with ``barrier=U`` and ``rebate=0``.
        """
        args = self._arguments
        assert args.payoff is not None
        assert args.barrier_hi is not None
        assert args.barrier_type is not None
        assert args.exercise is not None
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)

        opt = BarrierOption(
            barrier_type=args.barrier_type,
            barrier=args.barrier_hi,
            rebate=0.0,
            payoff=args.payoff,
            exercise=args.exercise,
        )
        opt.set_pricing_engine(AnalyticBarrierEngine(self._process))
        return max(opt.npv(), 0.0)

    # ------------------------------------------------------------------
    # Process accessors
    # ------------------------------------------------------------------
    def _underlying(self) -> float:
        return self._process.state_variable().value()

    def _strike(self) -> float:
        args = self._arguments
        assert args.payoff is not None
        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff), "non-plain payoff"
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        return args.payoff.strike()

    def _residual_time(self) -> float:
        args = self._arguments
        assert args.exercise is not None
        return self._process.time(args.exercise.last_date())

    def _volatility(self) -> float:
        return self._process.black_volatility().black_vol(
            self._arguments.exercise.last_date()  # type: ignore[union-attr]
            if self._arguments.exercise is not None
            else None,
            self._strike(),
            extrapolate=True,
        )

    def _std_deviation(self) -> float:
        return self._volatility() * math.sqrt(self._residual_time())

    def _risk_free_rate(self) -> float:
        T = self._residual_time()
        return (
            self._process.risk_free_rate()
            .zero_rate(
                T,
                compounding=Compounding.Continuous,
                frequency=Frequency.NoFrequency,
            )
            .rate()
        )

    def _risk_free_discount(self) -> float:
        return self._process.risk_free_rate().discount(
            self._residual_time(), extrapolate=True
        )

    def _dividend_yield(self) -> float:
        T = self._residual_time()
        return (
            self._process.dividend_yield()
            .zero_rate(
                T,
                compounding=Compounding.Continuous,
                frequency=Frequency.NoFrequency,
            )
            .rate()
        )

    def _dividend_discount(self) -> float:
        return self._process.dividend_yield().discount(
            self._residual_time(), extrapolate=True
        )


__all__ = ["AnalyticSoftBarrierEngine"]
