"""BlackDeltaCalculator — FX Black-Scholes delta / strike-from-delta.

# C++ parity: ql/pricingengines/blackdeltacalculator.{hpp,cpp} (v1.42.1).

FX options are quoted by delta, and a strike can be expressed in both
numeraires; this helper packages the operations needed to convert
between strike and delta under the various FX delta conventions
(spot, forward, premium-adjusted spot/forward) and to compute the
ATM strike for the standard ATM conventions.

The class works with the *standard deviation* (``vol * sqrt(T)``)
rather than the volatility, matching C++.
"""

from __future__ import annotations

import math
import sys

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.payoffs import OptionType
from pquantlib.termstructures.volatility.equity_fx.delta_vol_quote import (
    AtmType,
    DeltaType,
)

_QL_EPSILON: float = sys.float_info.epsilon


class BlackDeltaCalculator:
    """Black delta calculator for FX conventions.

    # C++ parity: ``BlackDeltaCalculator``.

    Args:
        option_type: Call or Put.
        delta_type: FX delta convention.
        spot: spot FX rate (> 0).
        d_discount: domestic discount factor (> 0).
        f_discount: foreign discount factor (> 0).
        std_dev: standard deviation ``volatility * sqrt(T)`` (>= 0).
    """

    __slots__ = (
        "_d_discount",
        "_delta_type",
        "_f_discount",
        "_f_exp_neg",
        "_f_exp_pos",
        "_forward",
        "_option_type",
        "_phi",
        "_spot",
        "_std_dev",
    )

    def __init__(
        self,
        option_type: OptionType,
        delta_type: DeltaType,
        spot: float,
        d_discount: float,
        f_discount: float,
        std_dev: float,
    ) -> None:
        qassert.require(spot > 0.0, f"positive spot value required: {spot} not allowed")
        qassert.require(
            d_discount > 0.0,
            f"positive domestic discount factor required: {d_discount} not allowed",
        )
        qassert.require(
            f_discount > 0.0,
            f"positive foreign discount factor required: {f_discount} not allowed",
        )
        qassert.require(
            std_dev >= 0.0,
            f"non-negative standard deviation required: {std_dev} not allowed",
        )

        self._delta_type: DeltaType = delta_type
        self._option_type: OptionType = option_type
        self._d_discount: float = d_discount
        self._f_discount: float = f_discount
        self._std_dev: float = std_dev
        self._spot: float = spot
        self._forward: float = spot * f_discount / d_discount
        self._phi: int = int(option_type)

        self._f_exp_pos: float = self._forward * math.exp(0.5 * std_dev * std_dev)
        self._f_exp_neg: float = self._forward * math.exp(-0.5 * std_dev * std_dev)

    # --- public API -----------------------------------------------------

    def delta_from_strike(self, strike: float) -> float:
        """Option delta for ``strike`` under the configured convention."""
        qassert.require(strike >= 0.0, f"positive strike value required: {strike} not allowed")

        if self._delta_type == DeltaType.Spot:
            return self._phi * self._f_discount * self._cum_d1(strike)
        if self._delta_type == DeltaType.Fwd:
            return self._phi * self._cum_d1(strike)
        if self._delta_type == DeltaType.PaSpot:
            return self._phi * self._f_discount * self._cum_d2(strike) * strike / self._forward
        if self._delta_type == DeltaType.PaFwd:
            return self._phi * self._cum_d2(strike) * strike / self._forward
        qassert.fail("invalid delta type")
        raise AssertionError  # unreachable

    def strike_from_delta(self, delta: float) -> float:
        """Strike yielding ``delta`` under the configured convention."""
        return self._strike_from_delta(delta, self._delta_type)

    def atm_strike(self, atm_type: AtmType) -> float:
        """ATM strike for the requested ATM convention."""
        if atm_type == AtmType.AtmSpot:
            return self._spot
        if atm_type == AtmType.AtmDeltaNeutral:
            if self._delta_type in (DeltaType.Spot, DeltaType.Fwd):
                return self._f_exp_pos
            return self._f_exp_neg
        if atm_type == AtmType.AtmFwd:
            return self._forward
        if atm_type in (AtmType.AtmGammaMax, AtmType.AtmVegaMax):
            return self._f_exp_pos
        if atm_type == AtmType.AtmPutCall50:
            qassert.require(
                self._delta_type == DeltaType.Fwd,
                "|PutDelta|=CallDelta=0.50 only possible for forward delta.",
            )
            return self._f_exp_pos
        qassert.fail("invalid atm type")
        raise AssertionError  # unreachable

    def set_delta_type(self, delta_type: DeltaType) -> None:
        self._delta_type = delta_type

    def set_option_type(self, option_type: OptionType) -> None:
        self._option_type = option_type
        self._phi = int(option_type)

    # --- internal: strike-from-delta dispatch ---------------------------

    def _strike_from_delta(self, delta: float, delta_type: DeltaType) -> float:
        inv_norm = InverseCumulativeNormal()
        qassert.require(delta * self._phi >= 0.0, "Option type and delta are incoherent.")

        if delta_type == DeltaType.Spot:
            qassert.require(math.fabs(delta) <= self._f_discount, "Spot delta out of range.")
            arg = (
                -self._phi * inv_norm(self._phi * delta / self._f_discount) * self._std_dev
                + 0.5 * self._std_dev * self._std_dev
            )
            return self._forward * math.exp(arg)

        if delta_type == DeltaType.Fwd:
            qassert.require(math.fabs(delta) <= 1.0, "Forward delta out of range.")
            arg = (
                -self._phi * inv_norm(self._phi * delta) * self._std_dev
                + 0.5 * self._std_dev * self._std_dev
            )
            return self._forward * math.exp(arg)

        if delta_type in (DeltaType.PaSpot, DeltaType.PaFwd):
            # Premium-adjusted call delta is not monotonic in strike, so we
            # solve numerically (Brent, bracketed). See C++ comment block.
            def f(strike: float) -> float:
                return self.delta_from_strike(strike) - delta

            solver = Brent()
            solver.set_max_evaluations(1000)
            accuracy = 1.0e-10

            # Non-premium-adjusted strike is always to the right.
            if delta_type == DeltaType.PaSpot:
                right_limit = self._strike_from_delta(delta, DeltaType.Spot)
            else:
                right_limit = self._strike_from_delta(delta, DeltaType.Fwd)

            if self._phi < 0:  # put
                return solver.solve(f, accuracy, right_limit, 0.0, self._spot * 100.0)

            # Call: bracket left of the delta-maximum.
            def g(strike: float) -> float:
                return self._cum_d2(strike) * self._std_dev - self._n_d2(strike)

            left_limit = solver.solve(g, accuracy, right_limit * 0.5, 0.0, right_limit)
            guess = left_limit + (right_limit - left_limit) * 0.5
            return solver.solve(f, accuracy, guess, left_limit, right_limit)

        qassert.fail("invalid delta type")
        raise AssertionError  # unreachable

    # --- internal: N(d1), N(d2), n(d1), n(d2) ---------------------------

    def _cum_d1(self, strike: float) -> float:
        cum_d1_pos = 1.0  # N(d1)
        cum_d1_neg = 0.0  # N(-d1)
        f = CumulativeNormalDistribution()

        if self._std_dev >= _QL_EPSILON:
            if strike > 0:
                d1 = math.log(self._forward / strike) / self._std_dev + 0.5 * self._std_dev
                return f(self._phi * d1)
        elif self._forward < strike:
            cum_d1_pos = 0.0
            cum_d1_neg = 1.0
        elif self._forward == strike:
            d1 = 0.5 * self._std_dev
            return f(self._phi * d1)

        return cum_d1_pos if self._phi > 0 else cum_d1_neg

    def _n_d1(self, strike: float) -> float:
        n_d1 = 0.0  # n(d1)
        if self._std_dev >= _QL_EPSILON and strike > 0:
            d1 = math.log(self._forward / strike) / self._std_dev + 0.5 * self._std_dev
            n_d1 = CumulativeNormalDistribution().derivative(d1)
        return n_d1

    def _cum_d2(self, strike: float) -> float:
        cum_d2_pos = 1.0  # N(d2)
        cum_d2_neg = 0.0  # N(-d2)
        f = CumulativeNormalDistribution()

        if self._std_dev >= _QL_EPSILON:
            if strike > 0:
                d2 = math.log(self._forward / strike) / self._std_dev - 0.5 * self._std_dev
                return f(self._phi * d2)
        elif self._forward < strike:
            cum_d2_pos = 0.0
            cum_d2_neg = 1.0
        elif self._forward == strike:
            d2 = -0.5 * self._std_dev
            return f(self._phi * d2)

        return cum_d2_pos if self._phi > 0 else cum_d2_neg

    def _n_d2(self, strike: float) -> float:
        n_d2 = 0.0  # n(d2)
        if self._std_dev >= _QL_EPSILON and strike > 0:
            d2 = math.log(self._forward / strike) / self._std_dev - 0.5 * self._std_dev
            n_d2 = CumulativeNormalDistribution().derivative(d2)
        return n_d2


__all__ = ["BlackDeltaCalculator"]
