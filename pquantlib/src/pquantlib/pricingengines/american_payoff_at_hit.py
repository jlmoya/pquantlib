"""AmericanPayoffAtHit — analytic American digital paying at the hitting time.

# C++ parity: ql/pricingengines/americanpayoffathit.{hpp,cpp} (v1.43) —
# ``class AmericanPayoffAtHit``.

Closed-form value of a one-touch (up-and-in / down-and-in) binary whose
payoff is delivered the moment the barrier ``strike`` is touched, rather
than at expiry.  The barrier is the payoff's own strike:

* ``Option::Call`` -> up-and-in (barrier above spot),
* ``Option::Put``  -> down-and-in (barrier below spot).

The pricer is a plain calculator over ``(spot, discount, dividend_discount,
variance, payoff)`` — no term structures, no observability — which is why
``AnalyticDigitalAmericanEngine`` can reuse it verbatim.

The value is ``K * (forward * alpha + X * beta)`` with

* ``mu     = ln(qDF / rDF) / variance - 0.5``,
* ``lambda = sqrt(mu^2 - 2 ln(rDF) / variance)``,
* ``forward = (H/S)^(mu + lambda)``, ``X = (H/S)^(mu - lambda)`` when the
  barrier has not been touched, and ``forward = X = 1`` when it has.

Two corners a port must not tidy away
-------------------------------------
1. The alpha/beta switch uses non-strict comparisons (``strike > spot`` for
   a Call, ``strike < spot`` for a Put) while ``in_the_money`` uses strict
   ones.  At ``strike == spot`` the two therefore disagree: the 0.5 branch
   fires *and* the ``(H/S)^k`` powers are evaluated.  They happen to be 1.0,
   so nothing breaks — but the shape must be preserved.
2. ``discount == 0.0 and dividend_discount == 0.0`` and ``discount == 0.0``
   are handled in the C++ source *after* ``QL_REQUIRE(discount > 0.0)`` has
   already fired, so both are unreachable.  They are documented here rather
   than reproduced as dead code.

Known C++ defect (reproduced, not fixed)
----------------------------------------
On the ``variance < QL_EPSILON`` path the C++ constructor never assigns
``D1_``/``D2_``, and ``gamma()`` reads them.  That is an indeterminate read.
This port sets them to NaN on that path so ``gamma()`` is visibly unusable
instead of silently plausible; the cross-validation test skips gamma there
(the probe marks those cases ``gamma_undefined``).
"""

from __future__ import annotations

import math
import sys
from typing import Final

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
    OptionType,
    StrikedTypePayoff,
)
from pquantlib.pricingengines._ieee_arithmetic import div as _div
from pquantlib.pricingengines._ieee_arithmetic import sqrt as _sqrt

# C++ ``QL_EPSILON`` is ``DBL_EPSILON``.
_QL_EPSILON: Final[float] = sys.float_info.epsilon




class AmericanPayoffAtHit:
    """Analytic formula for American-exercise payoff-at-hit binaries.

    # C++ parity: ``class AmericanPayoffAtHit`` (americanpayoffathit.hpp:33-56).
    """

    def __init__(  # noqa: PLR0915 - one-to-one with the C++ constructor
        self,
        spot: float,
        discount: float,
        dividend_discount: float,
        variance: float,
        payoff: StrikedTypePayoff,
    ) -> None:
        qassert.require(spot > 0.0, "positive spot value required")
        qassert.require(discount > 0.0, "positive discount required")
        qassert.require(dividend_discount > 0.0, "positive dividend discount required")
        qassert.require(variance >= 0.0, "negative variance not allowed")

        self._spot: float = spot
        self._discount: float = discount
        self._dividend_discount: float = dividend_discount
        self._variance: float = variance
        self._std_dev: float = math.sqrt(variance)

        option_type = payoff.option_type()
        self._strike: float = payoff.strike()
        self._log_h_s: float = math.log(self._strike / spot)

        if variance >= _QL_EPSILON:
            # C++ also branches on ``discount == 0`` here, but
            # QL_REQUIRE(discount > 0.0) above makes that unreachable.
            self._mu: float = _div(math.log(dividend_discount / discount), variance) - 0.5
            self._lambda: float = _sqrt(
                self._mu * self._mu - _div(2.0 * math.log(discount), variance)
            )
            self._d1: float = _div(self._log_h_s, self._std_dev) + self._lambda * self._std_dev
            self._d2: float = self._d1 - 2.0 * self._lambda * self._std_dev
            f = CumulativeNormalDistribution()
            self._cum_d1: float = f(self._d1)
            self._cum_d2: float = f(self._d2)
            n_d1 = f.derivative(self._d1)
            n_d2 = f.derivative(self._d2)
        else:
            # C++ comment: "not tested yet". D1_/D2_ are left indeterminate
            # there; NaN is the honest Python analogue (see the module note).
            self._mu = _div(math.log(dividend_discount / discount), variance) - 0.5
            self._lambda = _sqrt(self._mu * self._mu - _div(2.0 * math.log(discount), variance))
            self._d1 = math.nan
            self._d2 = math.nan
            breached = self._log_h_s > 0
            self._cum_d1 = 1.0 if breached else 0.0
            self._cum_d2 = 1.0 if breached else 0.0
            n_d1 = 0.0
            n_d2 = 0.0

        if option_type == OptionType.Call:
            # up-and-in cash-(at-hit)-or-nothing option
            if self._strike > spot:
                self._alpha: float = 1.0 - self._cum_d1  # N(-d1)
                self._dalpha_dd1: float = -n_d1  # -n(d1)
                self._beta: float = 1.0 - self._cum_d2  # N(-d2)
                self._dbeta_dd2: float = -n_d2  # -n(d2)
            else:
                self._alpha = 0.5
                self._dalpha_dd1 = 0.0
                self._beta = 0.5
                self._dbeta_dd2 = 0.0
        else:  # noqa: PLR5501 - mirrors the C++ ``switch (type)`` arm boundary
            # down-and-in cash-(at-hit)-or-nothing option
            if self._strike < spot:
                self._alpha = self._cum_d1  # N(d1)
                self._dalpha_dd1 = n_d1  # n(d1)
                self._beta = self._cum_d2  # N(d2)
                self._dbeta_dd2 = n_d2  # n(d2)
            else:
                self._alpha = 0.5
                self._dalpha_dd1 = 0.0
                self._beta = 0.5
                self._dbeta_dd2 = 0.0

        self._mu_plus_lambda: float = self._mu + self._lambda
        self._mu_minus_lambda: float = self._mu - self._lambda
        # NOTE strict comparisons here versus the non-strict ones above.
        self._in_the_money: bool = (option_type == OptionType.Call and self._strike < spot) or (
            option_type == OptionType.Put and self._strike > spot
        )

        if self._in_the_money:
            self._forward: float = 1.0
            self._x: float = 1.0
        else:
            self._forward = math.pow(self._strike / spot, self._mu_plus_lambda)
            self._x = math.pow(self._strike / spot, self._mu_minus_lambda)

        # K_ depends on the concrete binary payoff. C++ uses two successive
        # dynamic_pointer_casts; neither matches a PlainVanillaPayoff, which
        # would leave K_ indeterminate — this engine is only ever fed binaries.
        self._k: float = math.nan
        if isinstance(payoff, CashOrNothingPayoff):
            self._k = payoff.cash_payoff()
        if isinstance(payoff, AssetOrNothingPayoff):
            self._k = spot if self._in_the_money else payoff.strike()

    # -- results ----------------------------------------------------------

    def value(self) -> float:
        """Present value.

        # C++ parity: inline ``AmericanPayoffAtHit::value``
        # (americanpayoffathit.hpp:62-64).
        """
        return self._k * (self._forward * self._alpha + self._x * self._beta)

    def delta(self) -> float:
        """dV/dS.

        # C++ parity: ``AmericanPayoffAtHit::delta`` (americanpayoffathit.cpp).
        """
        temp_delta = -self._spot * self._std_dev
        dalpha_ds = _div(self._dalpha_dd1, temp_delta)
        dbeta_ds = _div(self._dbeta_dd2, temp_delta)

        if self._in_the_money:
            dforward_ds = 0.0
            dx_ds = 0.0
        else:
            dforward_ds = -self._mu_plus_lambda * self._forward / self._spot
            dx_ds = -self._mu_minus_lambda * self._x / self._spot

        return self._k * (
            dalpha_ds * self._forward
            + self._alpha * dforward_ds
            + dbeta_ds * self._x
            + self._beta * dx_ds
        )

    def gamma(self) -> float:
        """d2V/dS2.

        # C++ parity: ``AmericanPayoffAtHit::gamma``.

        Reads ``d1``/``d2``, which are NaN on the ``variance < QL_EPSILON``
        path (C++ leaves them indeterminate there — see the module note).
        """
        temp_delta = -self._spot * self._std_dev
        dalpha_ds = _div(self._dalpha_dd1, temp_delta)
        dbeta_ds = _div(self._dbeta_dd2, temp_delta)
        d2alpha_ds2 = -dalpha_ds / self._spot * (1 - _div(self._d1, self._std_dev))
        d2beta_ds2 = -dbeta_ds / self._spot * (1 - _div(self._d2, self._std_dev))

        if self._in_the_money:
            dforward_ds = 0.0
            dx_ds = 0.0
            d2forward_ds2 = 0.0
            d2x_ds2 = 0.0
        else:
            dforward_ds = -self._mu_plus_lambda * self._forward / self._spot
            dx_ds = -self._mu_minus_lambda * self._x / self._spot
            d2forward_ds2 = (
                self._mu_plus_lambda
                * self._forward
                / (self._spot * self._spot)
                * (1 + self._mu_plus_lambda)
            )
            d2x_ds2 = (
                self._mu_minus_lambda
                * self._x
                / (self._spot * self._spot)
                * (1 + self._mu_minus_lambda)
            )

        return self._k * (
            d2alpha_ds2 * self._forward
            + dalpha_ds * dforward_ds
            + dalpha_ds * dforward_ds
            + self._alpha * d2forward_ds2
            + d2beta_ds2 * self._x
            + dbeta_ds * dx_ds
            + dbeta_ds * dx_ds
            + self._beta * d2x_ds2
        )

    def rho(self, maturity: float) -> float:
        """dV/dr.

        # C++ parity: ``AmericanPayoffAtHit::rho(Time maturity)``. The C++
        # comment notes the body computes ``D.Dr / T`` and multiplies by
        # ``maturity`` at the end.
        """
        qassert.require(maturity >= 0.0, "negative maturity not allowed")

        dalpha_dr = _div(-self._dalpha_dd1, self._lambda * self._std_dev) * (1.0 + self._mu)
        dbeta_dr = _div(self._dbeta_dd2, self._lambda * self._std_dev) * (1.0 + self._mu)

        if self._in_the_money:
            dforward_dr = 0.0
            dx_dr = 0.0
        else:
            dforward_dr = _div(
                self._forward * (1.0 + _div(1.0 + self._mu, self._lambda)) * self._log_h_s,
                self._variance,
            )
            dx_dr = _div(
                self._x * (1.0 - _div(1.0 + self._mu, self._lambda)) * self._log_h_s,
                self._variance,
            )

        return (
            maturity
            * self._k
            * (
                dalpha_dr * self._forward
                + self._alpha * dforward_dr
                + dbeta_dr * self._x
                + self._beta * dx_dr
            )
        )


__all__ = ["AmericanPayoffAtHit"]
