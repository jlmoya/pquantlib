"""BlackCalculator — Black 1976 Greeks calculator.

# C++ parity: ql/pricingengines/blackcalculator.{hpp,cpp} (v1.42.1).

The C++ class precomputes the d1 / d2 / N(d1) / N(d2) / n(d1) / n(d2)
intermediates from (forward, stdDev, strike, discount), then exposes
value() + Greeks (delta, gamma, vega, theta, rho, dividendRho,
strikeSensitivity, vanna, volga, ...).

C++ uses a Visitor pattern (``BlackCalculator::Calculator``) to
specialize the ``alpha`` / ``beta`` / ``x`` / ``DxDstrike`` quantities
for each ``StrikedTypePayoff`` subtype (PlainVanilla / CashOrNothing /
AssetOrNothing / Gap). The Python port replaces the Visitor with
``isinstance`` dispatch — direct, idiomatic, no double-dispatch
ceremony. The same four payoffs are supported; unknown subtypes raise
``LibraryException``.

Carve-outs:
- ``vanna`` and ``volga`` are ported (matched against C++).
- ``elasticityForward`` / ``gammaForward`` / ``strikeGamma`` are
  ported.
- The constructor that takes a ``(Option::Type, strike, ...)`` tuple
  (no payoff object) is exposed as a ``from_type_strike`` classmethod
  factory.
"""

from __future__ import annotations

import math
import sys
from typing import Final

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
    GapPayoff,
    OptionType,
    PlainVanillaPayoff,
    StrikedTypePayoff,
)

_PHI: Final[CumulativeNormalDistribution] = CumulativeNormalDistribution()
_QL_MAX_REAL: Final[float] = sys.float_info.max
_QL_MIN_REAL: Final[float] = sys.float_info.min
# C++ ``M_SQRT_2 * M_1_SQRTPI`` = sqrt(2/pi)/2 = the standard-normal
# pdf at the origin (= 1/sqrt(2*pi)).
_NORM_PDF_0: Final[float] = 1.0 / math.sqrt(2.0 * math.pi)


class BlackCalculator:
    """Black 1976 closed-form option calculator.

    # C++ parity: ``class BlackCalculator``. Holds precomputed
    # ``alpha`` / ``beta`` / ``DalphaDd1`` / ``DbetaDd2`` plus the
    # ``x`` / ``DxDstrike`` / ``DxDs`` discriminants set by the
    # payoff-specific dispatch in ``initialize``.
    """

    def __init__(
        self,
        payoff: StrikedTypePayoff,
        forward: float,
        std_dev: float,
        discount: float = 1.0,
    ) -> None:
        self._strike: float = payoff.strike()
        self._forward: float = forward
        self._std_dev: float = std_dev
        self._discount: float = discount
        self._variance: float = std_dev * std_dev
        self._initialize(payoff)

    @classmethod
    def from_type_strike(
        cls,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
        discount: float = 1.0,
    ) -> BlackCalculator:
        """Convenience: build from (type, strike) without a payoff object.

        # C++ parity: second BlackCalculator constructor (line 54 of
        # blackcalculator.cpp).
        """
        return cls(PlainVanillaPayoff(option_type, strike), forward, std_dev, discount)

    # --- private initialization -----------------------------------------

    def _initialize(self, payoff: StrikedTypePayoff) -> None:  # noqa: PLR0915
        """Compute d1/d2 + cumulative/density + alpha/beta dispatch.

        # C++ parity: ``BlackCalculator::initialize``. The branch logic
        # matches C++ line-by-line — too few statements to split into
        # helper methods without obscuring the parity.
        """
        qassert.require(
            self._strike >= 0.0, f"strike ({self._strike}) must be non-negative"
        )
        qassert.require(
            self._forward > 0.0, f"forward ({self._forward}) must be positive"
        )
        qassert.require(
            self._std_dev >= 0.0, f"stdDev ({self._std_dev}) must be non-negative"
        )
        qassert.require(
            self._discount > 0.0, f"discount ({self._discount}) must be positive"
        )

        if self._std_dev >= QL_EPSILON:
            if close(self._strike, 0.0):
                self._d1: float = _QL_MAX_REAL
                self._d2: float = _QL_MAX_REAL
                self._cum_d1: float = 1.0
                self._cum_d2: float = 1.0
                self._n_d1: float = 0.0
                self._n_d2: float = 0.0
            else:
                self._d1 = (
                    math.log(self._forward / self._strike) / self._std_dev
                    + 0.5 * self._std_dev
                )
                self._d2 = self._d1 - self._std_dev
                self._cum_d1 = _PHI(self._d1)
                self._cum_d2 = _PHI(self._d2)
                self._n_d1 = _PHI.derivative(self._d1)
                self._n_d2 = _PHI.derivative(self._d2)
        elif close(self._forward, self._strike):
            self._d1 = 0.0
            self._d2 = 0.0
            self._cum_d1 = 0.5
            self._cum_d2 = 0.5
            self._n_d1 = _NORM_PDF_0
            self._n_d2 = _NORM_PDF_0
        elif self._forward > self._strike:
            self._d1 = _QL_MAX_REAL
            self._d2 = _QL_MAX_REAL
            self._cum_d1 = 1.0
            self._cum_d2 = 1.0
            self._n_d1 = 0.0
            self._n_d2 = 0.0
        else:
            self._d1 = _QL_MIN_REAL
            self._d2 = _QL_MIN_REAL
            self._cum_d1 = 0.0
            self._cum_d2 = 0.0
            self._n_d1 = 0.0
            self._n_d2 = 0.0

        self._x: float = self._strike
        self._dx_dstrike: float = 1.0
        # Reserved for SuperShare (DxDs); not exercised by the four
        # ported payoffs but kept for symmetry with C++.
        self._dx_ds: float = 0.0

        # alpha / beta by Call / Put discriminant — PlainVanilla baseline.
        if payoff.option_type() == OptionType.Call:
            self._alpha: float = self._cum_d1
            self._dalpha_dd1: float = self._n_d1
            self._beta: float = -self._cum_d2
            self._dbeta_dd2: float = -self._n_d2
        else:  # Put
            self._alpha = -1.0 + self._cum_d1
            self._dalpha_dd1 = self._n_d1
            self._beta = 1.0 - self._cum_d2
            self._dbeta_dd2 = -self._n_d2

        # Payoff-subtype dispatch (replaces C++ Visitor).
        self._dispatch_payoff(payoff)

    def _dispatch_payoff(self, payoff: StrikedTypePayoff) -> None:
        """Specialize alpha/beta/x for CashOrNothing/AssetOrNothing/Gap.

        # C++ parity: BlackCalculator::Calculator::visit overloads. The
        # default (PlainVanillaPayoff) is a no-op; the other three
        # override alpha/beta/x/DxDstrike.
        """
        # PlainVanillaPayoff: no-op (baseline already set).
        if isinstance(payoff, PlainVanillaPayoff):
            return

        if isinstance(payoff, CashOrNothingPayoff):
            self._alpha = 0.0
            self._dalpha_dd1 = 0.0
            self._x = payoff.cash_payoff()
            self._dx_dstrike = 0.0
            if payoff.option_type() == OptionType.Call:
                self._beta = self._cum_d2
                self._dbeta_dd2 = self._n_d2
            else:
                self._beta = 1.0 - self._cum_d2
                self._dbeta_dd2 = -self._n_d2
            return

        if isinstance(payoff, AssetOrNothingPayoff):
            self._beta = 0.0
            self._dbeta_dd2 = 0.0
            if payoff.option_type() == OptionType.Call:
                self._alpha = self._cum_d1
                self._dalpha_dd1 = self._n_d1
            else:
                self._alpha = 1.0 - self._cum_d1
                self._dalpha_dd1 = -self._n_d1
            return

        if isinstance(payoff, GapPayoff):
            self._x = payoff.second_strike()
            self._dx_dstrike = 0.0
            return

        # Unknown subtype — match C++ Calculator::visit(Payoff&).
        qassert.require(False, f"unsupported payoff type: {payoff.name()}")

    # --- value + Greeks --------------------------------------------------

    def value(self) -> float:
        """Option value.

        # C++ parity: ``BlackCalculator::value``.
        """
        return self._discount * (self._forward * self._alpha + self._x * self._beta)

    def delta(self, spot: float) -> float:
        """Spot delta.

        # C++ parity: ``BlackCalculator::delta(Real spot)``.
        """
        qassert.require(spot > 0.0, f"positive spot value required: {spot} not allowed")

        if self._std_dev <= QL_EPSILON:
            dforward_ds = self._forward / spot
            if close(self._forward, self._strike):
                if self._alpha >= 0:  # Call
                    return self._discount * 0.5 * dforward_ds
                return self._discount * -0.5 * dforward_ds
            if self._forward > self._strike:
                if self._alpha >= 0:  # Call (ITM)
                    return self._discount * 1.0 * dforward_ds
                return 0.0  # Put OTM
            if self._alpha >= 0:  # Call OTM
                return 0.0
            return self._discount * -1.0 * dforward_ds

        dforward_ds = self._forward / spot
        temp = self._std_dev * spot
        dalpha_ds = self._dalpha_dd1 / temp
        dbeta_ds = self._dbeta_dd2 / temp
        return self._discount * (
            dalpha_ds * self._forward
            + self._alpha * dforward_ds
            + dbeta_ds * self._x
            + self._beta * self._dx_ds
        )

    def delta_forward(self) -> float:
        """Forward delta.

        # C++ parity: ``BlackCalculator::deltaForward``.
        """
        if self._std_dev <= QL_EPSILON:
            if close(self._forward, self._strike):
                if self._alpha >= 0:
                    return self._discount * 0.5
                return self._discount * -0.5
            if self._forward > self._strike:
                if self._alpha >= 0:
                    return self._discount * 1.0
                return 0.0
            if self._alpha >= 0:
                return 0.0
            return self._discount * -1.0

        temp = self._std_dev * self._forward
        dalpha_dforward = self._dalpha_dd1 / temp
        dbeta_dforward = self._dbeta_dd2 / temp
        return self._discount * (
            dalpha_dforward * self._forward
            + self._alpha
            + dbeta_dforward * self._x
        )

    def elasticity(self, spot: float) -> float:
        """Elasticity = delta * spot / value.

        # C++ parity: ``BlackCalculator::elasticity``.
        """
        val = self.value()
        d = self.delta(spot)
        if val > QL_EPSILON:
            return d / val * spot
        if abs(d) < QL_EPSILON:
            return 0.0
        if d > 0.0:
            return _QL_MAX_REAL
        return _QL_MIN_REAL

    def elasticity_forward(self) -> float:
        val = self.value()
        d = self.delta_forward()
        if val > QL_EPSILON:
            return d / val * self._forward
        if abs(d) < QL_EPSILON:
            return 0.0
        if d > 0.0:
            return _QL_MAX_REAL
        return _QL_MIN_REAL

    def gamma(self, spot: float) -> float:
        """Spot gamma.

        # C++ parity: ``BlackCalculator::gamma(Real spot)``.
        """
        qassert.require(spot > 0.0, f"positive spot value required: {spot} not allowed")

        if self._std_dev <= QL_EPSILON:
            return 0.0

        dforward_ds = self._forward / spot
        temp = self._std_dev * spot
        dalpha_ds = self._dalpha_dd1 / temp
        dbeta_ds = self._dbeta_dd2 / temp
        d2alpha_ds2 = -dalpha_ds / spot * (1.0 + self._d1 / self._std_dev)
        d2beta_ds2 = -dbeta_ds / spot * (1.0 + self._d2 / self._std_dev)
        return self._discount * (
            d2alpha_ds2 * self._forward
            + 2.0 * dalpha_ds * dforward_ds
            + d2beta_ds2 * self._x
            + 2.0 * dbeta_ds * self._dx_ds
        )

    def gamma_forward(self) -> float:
        if self._std_dev <= QL_EPSILON:
            return 0.0

        temp = self._std_dev * self._forward
        dalpha_dforward = self._dalpha_dd1 / temp
        dbeta_dforward = self._dbeta_dd2 / temp
        d2alpha_df2 = -dalpha_dforward / self._forward * (1.0 + self._d1 / self._std_dev)
        d2beta_df2 = -dbeta_dforward / self._forward * (1.0 + self._d2 / self._std_dev)
        return self._discount * (
            d2alpha_df2 * self._forward
            + 2.0 * dalpha_dforward
            + d2beta_df2 * self._x
        )

    def theta(self, spot: float, maturity: float) -> float:
        """Spot theta.

        # C++ parity: ``BlackCalculator::theta``.
        """
        qassert.require(maturity >= 0.0, f"maturity ({maturity}) must be non-negative")
        if close(maturity, 0.0):
            return 0.0
        return -(
            math.log(self._discount) * self.value()
            + math.log(self._forward / spot) * spot * self.delta(spot)
            + 0.5 * self._variance * spot * spot * self.gamma(spot)
        ) / maturity

    def theta_per_day(self, spot: float, maturity: float) -> float:
        """Theta divided by 365 for daily decay.

        # C++ parity: ``BlackCalculator::thetaPerDay``.
        """
        return self.theta(spot, maturity) / 365.0

    def vega(self, maturity: float) -> float:
        """Black vega.

        # C++ parity: ``BlackCalculator::vega``.
        """
        qassert.require(maturity >= 0.0, "negative maturity not allowed")

        if self._std_dev <= QL_EPSILON:
            return 0.0

        temp = math.log(self._strike / self._forward) / self._variance
        dalpha_dsigma = self._dalpha_dd1 * (temp + 0.5)
        dbeta_dsigma = self._dbeta_dd2 * (temp - 0.5)
        return (
            self._discount
            * math.sqrt(maturity)
            * (dalpha_dsigma * self._forward + dbeta_dsigma * self._x)
        )

    def rho(self, maturity: float) -> float:
        """Rho — sensitivity to the discount rate.

        # C++ parity: ``BlackCalculator::rho``.
        """
        qassert.require(maturity >= 0.0, "negative maturity not allowed")

        if self._std_dev <= QL_EPSILON:
            delta_fwd = self.delta_forward()
            return maturity * (delta_fwd * self._forward - self.value())

        dalpha_dr = self._dalpha_dd1 / self._std_dev
        dbeta_dr = self._dbeta_dd2 / self._std_dev
        temp = dalpha_dr * self._forward + self._alpha * self._forward + dbeta_dr * self._x
        return maturity * (self._discount * temp - self.value())

    def dividend_rho(self, maturity: float) -> float:
        """Dividend rho — sensitivity to the dividend rate.

        # C++ parity: ``BlackCalculator::dividendRho``.
        """
        qassert.require(maturity >= 0.0, "negative maturity not allowed")

        if self._std_dev <= QL_EPSILON:
            delta_fwd = self.delta_forward() / self._discount
            return -maturity * self._discount * delta_fwd * self._forward

        dalpha_dq = -self._dalpha_dd1 / self._std_dev
        dbeta_dq = -self._dbeta_dd2 / self._std_dev
        temp = dalpha_dq * self._forward - self._alpha * self._forward + dbeta_dq * self._x
        return maturity * self._discount * temp

    def strike_sensitivity(self) -> float:
        """Sensitivity to a change in the strike.

        # C++ parity: ``BlackCalculator::strikeSensitivity``.
        """
        if self._std_dev <= QL_EPSILON:
            if close(self._forward, self._strike):
                if self._alpha >= 0:
                    return -self._discount * 0.5
                return self._discount * 0.5
            if self._forward > self._strike:
                if self._alpha >= 0:
                    return -self._discount * 1.0
                return self._discount * 0.0
            if self._alpha >= 0:
                return -self._discount * 0.0
            return self._discount * 1.0

        temp = self._std_dev * self._strike
        dalpha_dstrike = -self._dalpha_dd1 / temp
        dbeta_dstrike = -self._dbeta_dd2 / temp
        return self._discount * (
            dalpha_dstrike * self._forward
            + dbeta_dstrike * self._x
            + self._beta * self._dx_dstrike
        )

    def strike_gamma(self) -> float:
        if self._std_dev <= QL_EPSILON:
            return 0.0

        temp = self._std_dev * self._strike
        dalpha_dstrike = -self._dalpha_dd1 / temp
        dbeta_dstrike = -self._dbeta_dd2 / temp
        d2alpha_d2strike = -dalpha_dstrike / self._strike * (1.0 - self._d1 / self._std_dev)
        d2beta_d2strike = -dbeta_dstrike / self._strike * (1.0 - self._d2 / self._std_dev)
        return self._discount * (
            d2alpha_d2strike * self._forward
            + d2beta_d2strike * self._x
            + 2.0 * dbeta_dstrike * self._dx_dstrike
        )

    def vanna(self, spot: float, maturity: float) -> float:
        """Sensitivity of vega to spot.

        # C++ parity: ``BlackCalculator::vanna``.
        """
        qassert.require(spot > 0.0, f"positive spot value required: {spot} not allowed")
        qassert.require(maturity >= 0.0, "negative maturity not allowed")
        if self._std_dev <= QL_EPSILON:
            return 0.0
        v = self.vega(maturity)
        return -self._d2 / (spot * self._std_dev) * v

    def volga(self, maturity: float) -> float:
        """Sensitivity of vega to volatility.

        # C++ parity: ``BlackCalculator::volga``.
        """
        qassert.require(maturity >= 0.0, "negative maturity not allowed")
        if self._std_dev <= QL_EPSILON:
            return 0.0
        v = self.vega(maturity)
        return v * self._d1 * self._d2 / self._std_dev

    def itm_cash_probability(self) -> float:
        """ITM probability in the bond martingale measure (N(d2)).

        # C++ parity: ``BlackCalculator::itmCashProbability``.
        """
        return self._cum_d2

    def itm_asset_probability(self) -> float:
        """ITM probability in the asset martingale measure (N(d1)).

        # C++ parity: ``BlackCalculator::itmAssetProbability``.
        """
        return self._cum_d1

    def alpha(self) -> float:
        """Internal alpha (exposed for testing).

        # C++ parity: ``BlackCalculator::alpha``.
        """
        return self._alpha

    def beta(self) -> float:
        """Internal beta (exposed for testing).

        # C++ parity: ``BlackCalculator::beta``.
        """
        return self._beta


__all__ = ["BlackCalculator"]
