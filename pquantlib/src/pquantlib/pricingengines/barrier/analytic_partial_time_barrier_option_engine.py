# ruff: noqa: N802, N803, N806
# Reason: math-symbol var names (S, X, T, B, e1..e4, g1..g4, mu_, rho_,
# HSMu, HSMu1, X1) match Heynen-Kat 1994 / Haug "Complete Guide" 2nd ed
# notation 1:1 with the C++ source; replacing them with snake_case would
# break the parity with ``analyticpartialtimebarrieroptionengine.cpp``.
"""AnalyticPartialTimeBarrierOptionEngine — Heynen-Kat closed form.

# C++ parity:
# ql/pricingengines/barrier/analyticpartialtimebarrieroptionengine.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for partial-time barrier options under
Black-Scholes dynamics.

Reference: Haug "The Complete Guide to Option Pricing Formulas" 2nd
ed, p. 165 (formulas attributed to Heynen-Kat 1994). The formulas use
the bivariate cumulative normal ``M(a, b, rho)`` evaluated via the
Drezner-1978 approximation.

The engine currently does not implement knock-in partial-time end
barriers (this matches the C++ engine's QL_FAIL coverage).

For puts, the engine maps to a call via the FX-style put-call
reflection (X_call = S^2/X_put, B_call = S^2/B_put, with a flipped
barrier type and a scale factor X_put/S).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.barrieroption.partial_time_barrier_option import (
    PartialBarrierRange,
    PartialTimeBarrierOptionArguments,
)
from pquantlib.instruments.barrier_option import BarrierType
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
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


class AnalyticPartialTimeBarrierOptionEngine(
    GenericEngine[
        PartialTimeBarrierOptionArguments,
        OneAssetOptionResults,
    ]
):
    """Heynen-Kat 1994 closed-form partial-time barrier engine.

    # C++ parity: ``AnalyticPartialTimeBarrierOptionEngine``.

    Currently handles the same matrix of cases as the C++ engine:
    DownOut/UpOut x (Start, EndB1, EndB2) and DownIn/UpIn x Start.
    The four (DownIn, UpIn) x (EndB1, EndB2) combinations raise
    ``LibraryException`` (mirror of QL_FAIL).
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            PartialTimeBarrierOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        # Per-call FX put/call symmetry flag. Set by ``calculate`` before
        # ``_calculate_call`` dispatch.
        self._swap_rates: bool = False
        process.register_with(self)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def calculate(self) -> None:
        """# C++ parity: ``calculate()`` (.cpp:101-138).

        Handles the put-call reflection prior to delegating to
        ``_calculate_call``.
        """
        args = self._arguments
        results = self._results

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        qassert.require(payoff.strike() > 0.0, "strike must be positive")

        spot = self._underlying()
        qassert.require(spot > 0.0, "negative or null underlying given")

        assert args.barrier is not None
        assert args.barrier_type is not None
        assert args.barrier_range is not None

        if payoff.option_type() == OptionType.Put:
            # C++ parity: ``getSymmetricBarrierType`` + spot^2/strike
            # reflection — FX-style put/call symmetry (.cpp:118-135).
            # C++ also swaps r <-> q on the equivalent call process:
            #   GeneralizedBlackScholesProcess(x0, riskFreeRate,
            #                                   dividendYield, blackVol)
            # which feeds the symmetric-barrier formula a callProcess
            # where dividendTS = original r and riskFreeTS = original q.
            spot_sq = spot * spot
            call_strike = spot_sq / payoff.strike()
            call_payoff = PlainVanillaPayoff(OptionType.Call, call_strike)

            reflected_barrier_type = _flip_barrier_type(args.barrier_type)
            reflected_barrier = spot_sq / args.barrier
            scale = payoff.strike() / spot

            value = scale * self._calculate_call(
                payoff=call_payoff,
                barrier_type=reflected_barrier_type,
                barrier_range=args.barrier_range,
                barrier=reflected_barrier,
                swap_rates=True,
            )
        else:
            value = self._calculate_call(
                payoff=payoff,
                barrier_type=args.barrier_type,
                barrier_range=args.barrier_range,
                barrier=args.barrier,
                swap_rates=False,
            )

        results.value = value

    # ------------------------------------------------------------------
    # Branch table on (barrier_type, barrier_range)
    # ------------------------------------------------------------------
    def _calculate_call(
        self,
        *,
        payoff: PlainVanillaPayoff,
        barrier_type: BarrierType,
        barrier_range: PartialBarrierRange,
        barrier: float,
        swap_rates: bool,
    ) -> float:
        """# C++ parity: ``calculate(arguments, payoff, process)`` switch
        table (.cpp:34-99). Branches on (barrier_type, barrier_range)
        to dispatch among CA / CIA / CoB1 / CoB2.

        ``swap_rates`` toggles the put/call symmetry r<->q swap (.cpp:128-133):
        when True, the equivalent call is priced under a process where
        the original risk-free curve plays the dividend role and vice
        versa. Helper methods reading rates / discount honor this flag.
        """
        self._swap_rates: bool = swap_rates
        r = self._zero_rate(self._risk_free_rate_dc, is_risk_free=True)
        q = self._zero_rate(self._dividend_dc, is_risk_free=False)
        strike = payoff.strike()

        if barrier_type == BarrierType.DownOut:
            if barrier_range == PartialBarrierRange.Start:
                return self._CA(eta=1, barrier=barrier, strike=strike, r=r, q=q)
            if barrier_range == PartialBarrierRange.EndB1:
                return self._CoB1(barrier=barrier, strike=strike, r=r, q=q)
            return self._CoB2(
                barrier_type=BarrierType.DownOut,
                barrier=barrier,
                strike=strike,
                r=r,
                q=q,
            )

        if barrier_type == BarrierType.DownIn:
            if barrier_range == PartialBarrierRange.Start:
                return self._CIA(
                    eta=1, barrier=barrier, strike=strike, r=r, q=q
                )
            raise LibraryException(
                "Down-and-in partial-time end barrier is not implemented"
            )

        if barrier_type == BarrierType.UpOut:
            if barrier_range == PartialBarrierRange.Start:
                return self._CA(
                    eta=-1, barrier=barrier, strike=strike, r=r, q=q
                )
            if barrier_range == PartialBarrierRange.EndB1:
                return self._CoB1(barrier=barrier, strike=strike, r=r, q=q)
            return self._CoB2(
                barrier_type=BarrierType.UpOut,
                barrier=barrier,
                strike=strike,
                r=r,
                q=q,
            )

        if barrier_type == BarrierType.UpIn:
            if barrier_range == PartialBarrierRange.Start:
                return self._CIA(
                    eta=-1, barrier=barrier, strike=strike, r=r, q=q
                )
            raise LibraryException(
                "Up-and-in partial-time end barrier is not implemented"
            )

        raise LibraryException(f"unknown barrier type: {barrier_type}")

    # ------------------------------------------------------------------
    # Core closed-form pieces
    # ------------------------------------------------------------------
    def _CA(
        self,
        *,
        eta: int,
        barrier: float,
        strike: float,
        r: float,
        q: float,
    ) -> float:
        """Partial-Time-Start OUT call (eta=+1 down, eta=-1 up).

        # C++ parity: ``CA(eta, barrier, strike, r, q)`` (.cpp:242-264).
        """
        b = r - q
        T = self._residual_time()
        S = self._underlying()
        mu_ = self._mu(strike=strike, b=b)
        rho_ = self._rho()
        e1_ = self._e1(barrier, strike, b)
        e2_ = self._e2(barrier, strike, b)
        e3_ = self._e3(barrier, strike, b)
        e4_ = self._e4(barrier, strike, b)
        HSMu = self._HS(S, barrier, 2 * mu_)
        HSMu1 = self._HS(S, barrier, 2 * (mu_ + 1))

        result = S * math.exp((b - r) * T)
        result *= self._M(
            self._d1(strike, b), eta * e1_, eta * rho_
        ) - HSMu1 * self._M(
            self._f1(barrier, strike, b), eta * e3_, eta * rho_
        )
        result -= strike * math.exp(-r * T) * (
            self._M(self._d2(strike, b), eta * e2_, eta * rho_)
            - HSMu * self._M(
                self._f2(barrier, strike, b), eta * e4_, eta * rho_
            )
        )
        return result

    def _CIA(
        self,
        *,
        eta: int,
        barrier: float,
        strike: float,
        r: float,
        q: float,
    ) -> float:
        """Partial-Time-Start IN call: vanilla NPV - CA.

        # C++ parity: ``CIA(eta, barrier, strike, r, q)`` (.cpp:227-240).

        Uses an AnalyticEuropeanEngine bound to the (original) process.

        The C++ engine never enters this path under the put/call
        symmetry branch because UpIn/DownIn don't have a non-Start
        partial-range case (.cpp:60-68) and the symmetry-flipped
        barrier types preserve KI<->KI / KO<->KO classes — i.e. a Put
        UpIn maps to a Call DownIn (still a KI route through CIA).
        That CIA call would need the rate-swapped GBSM process; we
        raise here pending a follow-up to wrap the process. The Put
        UpOut→Call DownOut path used by the W4-C probe goes through
        CA (KO route) and never enters CIA.
        """
        args = self._arguments
        assert args.payoff is not None
        assert args.exercise is not None
        if self._swap_rates:
            raise LibraryException(
                "put put/call symmetry path through CIA "
                "(rate-swapped process) is not implemented; "
                "the put case only exercises CA-tier code paths"
            )
        eu_exercise = (
            args.exercise
            if isinstance(args.exercise, EuropeanExercise)
            else EuropeanExercise(args.exercise.last_date())
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        eu_opt = VanillaOption(args.payoff, eu_exercise)
        eu_opt.set_pricing_engine(AnalyticEuropeanEngine(self._process))
        return eu_opt.npv() - self._CA(
            eta=eta, barrier=barrier, strike=strike, r=r, q=q
        )

    def _CoB1(
        self,
        *,
        barrier: float,
        strike: float,
        r: float,
        q: float,
    ) -> float:
        """Partial-Time-EndB1 closed form (DownOut / UpOut).

        # C++ parity: ``CoB1(barrier, strike, r, q)`` (.cpp:188-223).
        """
        b = r - q
        T = self._residual_time()
        S = self._underlying()
        mu_ = self._mu(strike=strike, b=b)
        g1_ = self._g1(barrier, strike, b)
        g2_ = self._g2(barrier, strike, b)
        g3_ = self._g3(barrier, strike, b)
        g4_ = self._g4(barrier, strike, b)
        e1_ = self._e1(barrier, strike, b)
        e2_ = self._e2(barrier, strike, b)
        e3_ = self._e3(barrier, strike, b)
        e4_ = self._e4(barrier, strike, b)
        rho_ = self._rho()
        HSMu = self._HS(S, barrier, 2 * mu_)
        HSMu1 = self._HS(S, barrier, 2 * (mu_ + 1))
        X1 = strike * math.exp(-r * T)

        if strike > barrier:
            result = S * math.exp((b - r) * T)
            result *= self._M(
                self._d1(strike, b), e1_, rho_
            ) - HSMu1 * self._M(self._f1(barrier, strike, b), -e3_, -rho_)
            result -= X1 * (
                self._M(self._d2(strike, b), e2_, rho_)
                - HSMu * self._M(self._f2(barrier, strike, b), -e4_, -rho_)
            )
            return result

        S1 = S * math.exp((b - r) * T)
        result = S1
        result *= self._M(-g1_, -e1_, rho_) - HSMu1 * self._M(
            -g3_, e3_, -rho_
        )
        result -= X1 * (
            self._M(-g2_, -e2_, rho_) - HSMu * self._M(-g4_, e4_, -rho_)
        )
        result -= S1 * (
            self._M(-self._d1(strike, b), -e1_, rho_)
            - HSMu1 * self._M(-self._f1(barrier, strike, b), e3_, -rho_)
        )
        result += X1 * (
            self._M(-self._d2(strike, b), -e2_, rho_)
            - HSMu * self._M(-self._f2(barrier, strike, b), e4_, -rho_)
        )
        result += S1 * (
            self._M(g1_, e1_, rho_) - HSMu1 * self._M(g3_, -e3_, -rho_)
        )
        result -= X1 * (
            self._M(g2_, e2_, rho_) - HSMu * self._M(g4_, -e4_, -rho_)
        )
        return result

    def _CoB2(
        self,
        *,
        barrier_type: BarrierType,
        barrier: float,
        strike: float,
        r: float,
        q: float,
    ) -> float:
        """Partial-Time-EndB2 closed form (DownOut / UpOut, strike<barrier).

        # C++ parity: ``CoB2(barrierType, barrier, strike, r, q)``
        # (.cpp:140-186). C++ QL_FAIL's if strike >= barrier.
        """
        b = r - q
        T = self._residual_time()
        S = self._underlying()
        mu_ = self._mu(strike=strike, b=b)
        g1_ = self._g1(barrier, strike, b)
        g2_ = self._g2(barrier, strike, b)
        g3_ = self._g3(barrier, strike, b)
        g4_ = self._g4(barrier, strike, b)
        e1_ = self._e1(barrier, strike, b)
        e2_ = self._e2(barrier, strike, b)
        e3_ = self._e3(barrier, strike, b)
        e4_ = self._e4(barrier, strike, b)
        rho_ = self._rho()
        HSMu = self._HS(S, barrier, 2 * mu_)
        HSMu1 = self._HS(S, barrier, 2 * (mu_ + 1))
        X1 = strike * math.exp(-r * T)

        if strike >= barrier:
            raise LibraryException(
                "case of strike>barrier is not implemented for OutEnd B2 type"
            )

        if barrier_type == BarrierType.DownOut:
            result = S * math.exp((b - r) * T)
            result *= self._M(g1_, e1_, rho_) - HSMu1 * self._M(
                g3_, -e3_, -rho_
            )
            result -= X1 * (
                self._M(g2_, e2_, rho_) - HSMu * self._M(g4_, -e4_, -rho_)
            )
            return result

        if barrier_type == BarrierType.UpOut:
            result = S * math.exp((b - r) * T)
            result *= self._M(-g1_, -e1_, rho_) - HSMu1 * self._M(
                -g3_, e3_, -rho_
            )
            result -= X1 * (
                self._M(-g2_, -e2_, rho_) - HSMu * self._M(-g4_, e4_, -rho_)
            )
            result -= (
                S
                * math.exp((b - r) * T)
                * (
                    self._M(-self._d1(strike, b), -e1_, rho_)
                    - HSMu1
                    * self._M(e3_, -self._f1(barrier, strike, b), -rho_)
                )
            )
            result += X1 * (
                self._M(-self._d2(strike, b), -e2_, rho_)
                - HSMu * self._M(e4_, -self._f2(barrier, strike, b), -rho_)
            )
            return result

        raise LibraryException("invalid barrier type")

    # ------------------------------------------------------------------
    # Helper functions (d_i, e_i, f_i, g_i, mu, rho, M)
    # ------------------------------------------------------------------

    def _underlying(self) -> float:
        return self._process.state_variable().value()

    def _residual_time(self) -> float:
        args = self._arguments
        assert args.exercise is not None
        return self._process.time(args.exercise.last_date())

    def _cover_event_time(self) -> float:
        args = self._arguments
        assert args.cover_event_date is not None
        return self._process.time(args.cover_event_date)

    def _volatility(self, t: float, strike: float) -> float:
        return self._process.black_volatility().black_vol_at_time(
            t, strike, extrapolate=True
        )

    @property
    def _risk_free_rate_dc(self) -> object:
        return self._process.risk_free_rate().day_counter()

    @property
    def _dividend_dc(self) -> object:
        return self._process.dividend_yield().day_counter()

    def _zero_rate(self, dc: object, *, is_risk_free: bool) -> float:
        """# C++ parity: ``riskFreeRate->zeroRate(residualTime, Continuous,
        # NoFrequency)`` and similarly for dividend yield (.cpp:39-42).

        ``self._swap_rates`` toggles the FX put/call symmetry r<->q swap
        described in the calculate() docstring.
        """
        T = self._residual_time()
        # Pick the original curve, then swap if symmetry transform asked.
        use_risk_free = is_risk_free ^ self._swap_rates
        curve = (
            self._process.risk_free_rate()
            if use_risk_free
            else self._process.dividend_yield()
        )
        # C++ uses zeroRate(time, Continuous, NoFrequency), not zeroRate
        # at a date. Use the (time, dc, compounding, freq) overload.
        zr = curve.zero_rate(
            T,
            compounding=Compounding.Continuous,
            frequency=Frequency.NoFrequency,
        )
        return zr.rate()

    def _mu(self, *, strike: float, b: float) -> float:
        """# C++ parity: ``mu(strike, b)`` (.cpp:304-307)."""
        vol = self._volatility(self._cover_event_time(), strike)
        return (b - (vol * vol) / 2.0) / (vol * vol)

    def _rho(self) -> float:
        """# C++ parity: ``rho()`` (.cpp:300-302)."""
        return math.sqrt(self._cover_event_time() / self._residual_time())

    def _M(self, a: float, b: float, rho: float) -> float:
        """Bivariate cumulative normal at (a, b) with correlation rho.

        # C++ parity: ``M(a, b, rho)`` =
        #   ``BivariateCumulativeNormalDistributionDr78(rho)(a, b)`` (.cpp:295-298).
        # The Python port uses ``BivariateCumulativeNormalDistribution``
        # (Drezner-1978 high-precision) which matches.
        """
        # Use BivariateCumulativeNormalDistribution which is the Drezner
        # variant ported under that name.
        return BivariateCumulativeNormalDistribution(rho)(a, b)

    def _d1(self, strike: float, b: float) -> float:
        """# C++ parity: ``d1(strike, b)`` (.cpp:309-313)."""
        T2 = self._residual_time()
        vol = self._volatility(T2, strike)
        return (
            math.log(self._underlying() / strike) + (b + vol * vol / 2.0) * T2
        ) / (math.sqrt(T2) * vol)

    def _d2(self, strike: float, b: float) -> float:
        """# C++ parity: ``d2(strike, b)`` (.cpp:315-319)."""
        T2 = self._residual_time()
        vol = self._volatility(T2, strike)
        return self._d1(strike, b) - vol * math.sqrt(T2)

    def _e1(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``e1(barrier, strike, b)`` (.cpp:321-325)."""
        T1 = self._cover_event_time()
        vol = self._volatility(T1, strike)
        return (
            math.log(self._underlying() / barrier)
            + (b + vol * vol / 2.0) * T1
        ) / (math.sqrt(T1) * vol)

    def _e2(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``e2(barrier, strike, b)`` (.cpp:327-331)."""
        T1 = self._cover_event_time()
        vol = self._volatility(T1, strike)
        return self._e1(barrier, strike, b) - vol * math.sqrt(T1)

    def _e3(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``e3(barrier, strike, b)`` (.cpp:333-337)."""
        T1 = self._cover_event_time()
        vol = self._volatility(T1, strike)
        return self._e1(barrier, strike, b) + (
            2.0 * math.log(barrier / self._underlying()) / (vol * math.sqrt(T1))
        )

    def _e4(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``e4(barrier, strike, b)`` (.cpp:339-342)."""
        t = self._cover_event_time()
        return self._e3(barrier, strike, b) - self._volatility(
            t, strike
        ) * math.sqrt(t)

    def _g1(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``g1(barrier, strike, b)`` (.cpp:344-348)."""
        T2 = self._residual_time()
        vol = self._volatility(T2, strike)
        return (
            math.log(self._underlying() / barrier)
            + (b + vol * vol / 2.0) * T2
        ) / (math.sqrt(T2) * vol)

    def _g2(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``g2(barrier, strike, b)`` (.cpp:350-354)."""
        T2 = self._residual_time()
        vol = self._volatility(T2, strike)
        return self._g1(barrier, strike, b) - vol * math.sqrt(T2)

    def _g3(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``g3(barrier, strike, b)`` (.cpp:356-360)."""
        T2 = self._residual_time()
        vol = self._volatility(T2, strike)
        return self._g1(barrier, strike, b) + (
            2.0 * math.log(barrier / self._underlying()) / (vol * math.sqrt(T2))
        )

    def _g4(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``g4(barrier, strike, b)`` (.cpp:362-366)."""
        T2 = self._residual_time()
        vol = self._volatility(T2, strike)
        return self._g3(barrier, strike, b) - vol * math.sqrt(T2)

    def _f1(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``f1(barrier, strike, b)`` (.cpp:282-288)."""
        S = self._underlying()
        T = self._residual_time()
        sigma = self._volatility(T, strike)
        return (
            math.log(S / strike)
            + 2 * math.log(barrier / S)
            + (b + (sigma * sigma / 2.0)) * T
        ) / (sigma * math.sqrt(T))

    def _f2(self, barrier: float, strike: float, b: float) -> float:
        """# C++ parity: ``f2(barrier, strike, b)`` (.cpp:290-293)."""
        T = self._residual_time()
        return self._f1(barrier, strike, b) - self._volatility(
            T, strike
        ) * math.sqrt(T)

    @staticmethod
    def _HS(S: float, H: float, power: float) -> float:
        """# C++ parity: ``HS(S, H, power) = pow(H/S, power)`` (.cpp:368-370)."""
        return (H / S) ** power


def _flip_barrier_type(bt: BarrierType) -> BarrierType:
    """# C++ parity: ``getSymmetricBarrierType`` lambda (.cpp:111-116)."""
    if bt == BarrierType.UpIn:
        return BarrierType.DownIn
    if bt == BarrierType.DownIn:
        return BarrierType.UpIn
    if bt == BarrierType.UpOut:
        return BarrierType.DownOut
    return BarrierType.UpOut


__all__ = ["AnalyticPartialTimeBarrierOptionEngine"]
