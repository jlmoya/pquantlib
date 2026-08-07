"""AmericanPayoffAtExpiry — American digital paying at expiry, knock-in or -out.

# C++ parity: ql/pricingengines/americanpayoffatexpiry.{hpp,cpp} (v1.43) —
# ``class AmericanPayoffAtExpiry``.

Closed-form value of a binary that knocks in (or out) the first time the
barrier ``strike`` is touched, but pays only at expiry:

* ``Option::Call`` -> up-and-in / up-and-out,
* ``Option::Put``  -> down-and-in / down-and-out,

selected by the ``knock_in`` flag.  Like :class:`AmericanPayoffAtHit` this is
a bare calculator over ``(spot, discount, dividend_discount, variance,
payoff, knock_in)``; ``AnalyticDigitalAmericanEngine`` uses it for the
``payoff_at_expiry`` case.

Value is ``discount * K * (X * N(d1) + Y * N(d2))``.  The structure is a
sign pair ``(eta, phi)`` chosen by option type crossed with ``knock_in``,
then a second override when the barrier is *already* breached at t = 0:

=================  ==========  ===========================================
type / knock_in    breached?   cum_d1, cum_d2
=================  ==========  ===========================================
Call, knock-in     yes         0.5, 0.5   (touch is certain, pays for sure)
Call, knock-out    yes         0.0, 0.0   (already dead)
Put,  knock-in     yes         0.5, 0.5
Put,  knock-out    yes         0.0, 0.0
any                no          N(d1), N(d2)
=================  ==========  ===========================================

and finally ``Y`` is negated for the knock-out.  "Breached" is
``strike <= spot`` for a Call and ``strike >= spot`` for a Put — non-strict,
whereas ``in_the_money`` a few lines later is strict.  At ``strike == spot``
the two disagree; the shape is preserved deliberately.

An asset-or-nothing payoff sets ``K = forward`` **and shifts ``mu`` by 1** —
a port that forgets the shift still gets roughly the right number, which is
why the cross-validation covers both payoff kinds.

The ``if cum_d2 == 0.0: Y = 0.0`` guard (C++ comment: "check needed on some
extreme cases") short-circuits ``pow(strike/spot, 2*mu)`` for the breached
knock-out and degenerate-variance cases.
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

# C++ ``QL_EPSILON`` is ``DBL_EPSILON``.
_QL_EPSILON: Final[float] = sys.float_info.epsilon



class AmericanPayoffAtExpiry:
    """Analytic formula for American-exercise payoff-at-expiry binaries.

    # C++ parity: ``class AmericanPayoffAtExpiry``
    # (americanpayoffatexpiry.hpp:33-59).
    """

    def __init__(  # noqa: PLR0915 - one-to-one with the C++ constructor
        self,
        spot: float,
        discount: float,
        dividend_discount: float,
        variance: float,
        payoff: StrikedTypePayoff,
        knock_in: bool = True,
    ) -> None:
        qassert.require(spot > 0.0, "positive spot value required")
        qassert.require(discount > 0.0, "positive discount required")
        qassert.require(dividend_discount > 0.0, "positive dividend discount required")
        qassert.require(variance >= 0.0, "negative variance not allowed")

        self._spot: float = spot
        self._discount: float = discount
        self._dividend_discount: float = dividend_discount
        self._variance: float = variance
        self._knock_in: bool = knock_in
        self._std_dev: float = math.sqrt(variance)

        option_type = payoff.option_type()
        self._strike: float = payoff.strike()
        self._forward: float = spot * dividend_discount / discount

        self._mu: float = _div(math.log(dividend_discount / discount), variance) - 0.5

        # K_ depends on the concrete binary payoff (two successive
        # dynamic_pointer_casts in C++); asset-or-nothing also shifts mu_.
        self._k: float = math.nan
        if isinstance(payoff, CashOrNothingPayoff):
            self._k = payoff.cash_payoff()
        if isinstance(payoff, AssetOrNothingPayoff):
            self._k = self._forward
            self._mu += 1.0

        self._log_h_s: float = math.log(self._strike / spot)
        log_s_h = math.log(spot / self._strike)

        if option_type == OptionType.Call:
            # up-and-in (knock_in) / up-and-out cash-(at-expiry)-or-nothing
            eta = -1.0
            phi = 1.0 if knock_in else -1.0
        else:
            # down-and-in (knock_in) / down-and-out
            eta = 1.0
            phi = -1.0 if knock_in else 1.0

        if variance >= _QL_EPSILON:
            self._d1: float = phi * (_div(log_s_h, self._std_dev) + self._mu * self._std_dev)
            self._d2: float = eta * (_div(self._log_h_s, self._std_dev) + self._mu * self._std_dev)
            f = CumulativeNormalDistribution()
            self._cum_d1: float = f(self._d1)
            self._cum_d2: float = f(self._d2)
            self._n_d1: float = f.derivative(self._d1)
            self._n_d2: float = f.derivative(self._d2)
        else:
            # C++ leaves D1_/D2_ unassigned here; value() never reads them,
            # but NaN keeps the object honest if a caller ever does.
            self._d1 = math.nan
            self._d2 = math.nan
            self._cum_d1 = 1.0 if log_s_h * phi > 0 else 0.0
            self._cum_d2 = 1.0 if self._log_h_s * eta > 0 else 0.0
            self._n_d1 = 0.0
            self._n_d2 = 0.0

        # Barrier already breached at t = 0 -> the probabilities collapse.
        breached = (option_type == OptionType.Call and self._strike <= spot) or (
            option_type == OptionType.Put and self._strike >= spot
        )
        if breached:
            if knock_in:
                self._cum_d1 = 0.5
                self._cum_d2 = 0.5
            else:
                # already knocked out
                self._cum_d1 = 0.0
                self._cum_d2 = 0.0
            self._n_d1 = 0.0
            self._n_d2 = 0.0

        # NOTE strict comparisons here versus the non-strict ones above.
        self._in_the_money: bool = (option_type == OptionType.Call and self._strike < spot) or (
            option_type == OptionType.Put and self._strike > spot
        )
        if self._in_the_money:
            self._x: float = 1.0
            self._y: float = 1.0
        else:
            self._x = 1.0
            if self._cum_d2 == 0.0:
                self._y = 0.0  # check needed on some extreme cases
            else:
                self._y = math.pow(self._strike / spot, 2.0 * self._mu)
        if not knock_in:
            self._y *= -1.0

    def value(self) -> float:
        """Present value.

        # C++ parity: inline ``AmericanPayoffAtExpiry::value``
        # (americanpayoffatexpiry.hpp:65-67).
        """
        return self._discount * self._k * (self._x * self._cum_d1 + self._y * self._cum_d2)


__all__ = ["AmericanPayoffAtExpiry"]
