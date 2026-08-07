"""BachelierCalculator — normal-model (Bachelier 1900) Greeks calculator.

# C++ parity: ql/pricingengines/bacheliercalculator.{hpp,cpp} (v1.43),
#             ``class BachelierCalculator``.

The normal-model analogue of :class:`~pquantlib.pricingengines.black_calculator.BlackCalculator`.
The whole point of the model is that it is defined for **negative forwards and
negative strikes**, where every ``log()`` in the Black formula is undefined, so
the port must not reuse any of Black's logarithms. There is a single
discriminant::

    d = (F - K) / sigma            (sigma == stdDev, an absolute, not relative, vol)
    Call = (F - K) N(d) + sigma n(d)
    Put  = (K - F) N(-d) + sigma n(d)

Behaviour a port would plausibly get wrong, and which is cross-validated in
``pquantlib/tests/pricingengines/test_bachelier_calculator.py``:

* ``value()`` is **clamped at zero** — ``discount * max(result, 0)``
  (bacheliercalculator.cpp:188). No other greek is clamped.
* The option type is **not stored**; every branch recovers it from
  ``alpha_ >= 0``. For a plain Call ``alpha_ = N(d) >= 0`` and for a plain Put
  ``alpha_ = N(d) - 1 < 0``. But :class:`AssetOrNothingPayoff` **PUT** sets
  ``alpha_ = 1 - N(d) >= 0``, so ``value()``, ``delta_forward()``,
  ``itm_cash_probability()``, ``itm_asset_probability()``,
  ``strike_sensitivity()`` and ``dividend_rho()`` all take the *call* branch for
  an asset-or-nothing put. That is upstream behaviour and it is reproduced here.
* ``value()`` **ignores** ``alpha_`` / ``beta_`` / ``x_`` entirely — it
  recomputes ``intrinsic * cum_d + stdDev * n_d``. A CashOrNothing /
  AssetOrNothing / Gap payoff therefore returns the *plain vanilla* value; only
  the greeks that read ``alpha_``/``beta_`` differ.
* ``theta`` uses ``log(forward / spot)`` and is therefore NaN when the forward
  and the spot have opposite signs. C++ has no guard; neither does this.
* The zero-volatility branch (``stdDev < QL_EPSILON``) is asymmetric:
  ``value``/``theta`` test ``stdDev > QL_EPSILON`` while
  ``vega``/``gamma``/``gamma_forward``/``strike_gamma``/``vanna``/``volga``
  test ``stdDev <= QL_EPSILON``.

As in the ``BlackCalculator`` port, the C++ ``AcyclicVisitor``
(``BachelierCalculator::Calculator``) is replaced by ``isinstance`` dispatch:
same four payoff types, and any other ``StrikedTypePayoff`` raises exactly as
``Calculator::visit(Payoff&)`` does.
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
_QL_MIN_REAL: Final[float] = -sys.float_info.max
# C++ ``M_SQRT_2 * M_1_SQRTPI`` = (1/sqrt(2)) * (1/sqrt(pi)) = 1/sqrt(2*pi),
# the standard-normal density at the origin (mathconstants.hpp:86, :98).
_NORM_PDF_0: Final[float] = 1.0 / math.sqrt(2.0 * math.pi)


class BachelierCalculator:
    """Bachelier (normal-model) closed-form option calculator.

    # C++ parity: ``class BachelierCalculator`` (bacheliercalculator.hpp:33).
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
    ) -> BachelierCalculator:
        """Build from ``(type, strike)`` without a payoff object.

        # C++ parity: second constructor, bacheliercalculator.cpp:53-58 —
        # it wraps a ``PlainVanillaPayoff``, so it can never see an exotic
        # payoff and never trips the visitor's failure branch.
        """
        return cls(PlainVanillaPayoff(option_type, strike), forward, std_dev, discount)

    # --- private initialization -----------------------------------------

    def _initialize(self, payoff: StrikedTypePayoff) -> None:
        """C++ parity: ``BachelierCalculator::initialize`` (cpp:60-122)."""
        qassert.require(self._std_dev >= 0.0, f"stdDev ({self._std_dev}) must be non-negative")
        qassert.require(self._discount > 0.0, f"discount ({self._discount}) must be positive")

        if self._std_dev >= QL_EPSILON:
            self._d: float = (self._forward - self._strike) / self._std_dev
            self._cum_d: float = _PHI(self._d)
            self._n_d: float = _PHI.derivative(self._d)
        elif close(self._forward, self._strike):
            self._d = 0.0
            self._cum_d = 0.5
            self._n_d = _NORM_PDF_0
        elif self._forward > self._strike:
            self._d = _QL_MAX_REAL
            self._cum_d = 1.0
            self._n_d = 0.0
        else:
            self._d = _QL_MIN_REAL
            self._cum_d = 0.0
            self._n_d = 0.0

        self._x: float = self._strike
        self._dx_dstrike: float = 1.0
        self._dx_ds: float = 0.0

        if payoff.option_type() == OptionType.Call:
            self._alpha: float = self._cum_d
            self._dalpha_dd: float = self._n_d
            self._beta: float = -self._cum_d
            self._dbeta_dd: float = -self._n_d
        elif payoff.option_type() == OptionType.Put:
            self._alpha = self._cum_d - 1.0
            self._dalpha_dd = self._n_d
            self._beta = 1.0 - self._cum_d
            self._dbeta_dd = -self._n_d
        else:  # pragma: no cover - OptionType has exactly two members
            qassert.fail("invalid option type")

        self._dispatch(payoff)

    def _dispatch(self, payoff: StrikedTypePayoff) -> None:
        """C++ parity: ``BachelierCalculator::Calculator::visit`` overloads.

        Order matters only in that ``PlainVanillaPayoff`` is a no-op; the
        remaining three are disjoint types.
        """
        if isinstance(payoff, PlainVanillaPayoff):
            return  # cpp:128 — visit(PlainVanillaPayoff&) {}
        if isinstance(payoff, CashOrNothingPayoff):
            # cpp:130-146
            self._alpha = self._dalpha_dd = 0.0
            self._x = payoff.cash_payoff()
            self._dx_dstrike = 0.0
            if payoff.option_type() == OptionType.Call:
                self._beta = self._cum_d
                self._dbeta_dd = self._n_d
            else:
                self._beta = 1.0 - self._cum_d
                self._dbeta_dd = -self._n_d
            return
        if isinstance(payoff, AssetOrNothingPayoff):
            # cpp:148-162
            self._beta = self._dbeta_dd = 0.0
            if payoff.option_type() == OptionType.Call:
                self._alpha = self._cum_d
                self._dalpha_dd = self._n_d
            else:
                # NOTE: 1 - N(d) is >= 0, so `alpha_ >= 0` reads "Call" for an
                # asset-or-nothing PUT everywhere downstream. C++ behaviour.
                self._alpha = 1.0 - self._cum_d
                self._dalpha_dd = -self._n_d
            return
        if isinstance(payoff, GapPayoff):
            # cpp:164-167
            self._x = payoff.second_strike()
            self._dx_dstrike = 0.0
            return
        # cpp:124-126 — visit(Payoff&) { QL_FAIL("unsupported payoff type"); }
        qassert.fail(f"unsupported payoff type: {payoff.name()}")

    # --- inspectors -------------------------------------------------------

    def alpha(self) -> float:
        """C++ parity: bacheliercalculator.hpp:155-157."""
        return self._alpha

    def beta(self) -> float:
        """C++ parity: bacheliercalculator.hpp:159-161."""
        return self._beta

    # --- value ------------------------------------------------------------

    def value(self) -> float:
        """C++ parity: bacheliercalculator.cpp:169-189.

        Recomputes the payoff from ``cum_d``/``n_d`` and clamps at zero; it
        does not consult ``alpha_``/``beta_``/``x_`` except to decide the
        call/put branch.
        """
        intrinsic = self._forward - self._strike
        time_value = self._std_dev * self._n_d if self._std_dev > QL_EPSILON else 0.0
        if self._alpha >= 0:
            result = intrinsic * self._cum_d + time_value
        else:
            result = -intrinsic * (1.0 - self._cum_d) + time_value
        return self._discount * max(result, 0.0)

    # --- first-order greeks -----------------------------------------------

    def delta_forward(self) -> float:
        """C++ parity: cpp:204-214."""
        if self._alpha >= 0:
            return self._discount * self._cum_d
        return self._discount * (self._cum_d - 1.0)

    def delta(self, spot: float) -> float:
        """C++ parity: cpp:191-202 — ``deltaForward * (forward / spot)``."""
        return self.delta_forward() * (self._forward / spot)

    def elasticity_forward(self) -> float:
        """C++ parity: cpp:229-240."""
        val = self.value()
        der = self.delta_forward()
        if val > QL_EPSILON:
            return der / val * self._forward
        if abs(der) < QL_EPSILON:
            return 0.0
        return _QL_MAX_REAL if der > 0.0 else _QL_MIN_REAL

    def elasticity(self, spot: float) -> float:
        """C++ parity: cpp:216-227."""
        val = self.value()
        der = self.delta(spot)
        if val > QL_EPSILON:
            return der / val * spot
        if abs(der) < QL_EPSILON:
            return 0.0
        return _QL_MAX_REAL if der > 0.0 else _QL_MIN_REAL

    # --- second-order greeks ----------------------------------------------

    def gamma_forward(self) -> float:
        """C++ parity: cpp:259-268 — ``discount * n(d) / sigma``."""
        if self._std_dev <= QL_EPSILON:
            return 0.0
        return self._discount * self._n_d / self._std_dev

    def gamma(self, spot: float) -> float:
        """C++ parity: cpp:242-257 — gammaForward scaled by ``(F/S)^2``.

        Note the C++ computes the forward gamma inline WITHOUT the discount
        factor and multiplies by ``discount_`` once at the end, so this is
        ``gamma_forward() * (F/S)^2``.
        """
        if self._std_dev <= QL_EPSILON:
            return 0.0
        d_forward_ds = self._forward / spot
        gamma_forward = self._n_d / self._std_dev
        return self._discount * gamma_forward * d_forward_ds * d_forward_ds

    # --- time / vol / rate sensitivities -----------------------------------

    def theta(self, spot: float, maturity: float) -> float:
        """C++ parity: cpp:270-279.

        ``log(forward / spot)`` is NaN when forward and spot have opposite
        signs; C++ does not guard it and neither does this.
        """
        qassert.require(maturity >= 0.0, f"maturity ({maturity}) must be non-negative")
        if close(maturity, 0.0):
            return 0.0
        return (
            -(
                math.log(self._discount) * self.value()
                + math.log(self._forward / spot) * spot * self.delta(spot)
                + 0.5 * self._variance * self.gamma(spot)
            )
            / maturity
        )

    def theta_per_day(self, spot: float, maturity: float) -> float:
        """C++ parity: bacheliercalculator.hpp:125-128 — ``theta / 365``."""
        return self.theta(spot, maturity) / 365.0

    def vega(self, maturity: float) -> float:
        """C++ parity: cpp:281-297 — ``discount * sqrt(T) * n(d)``."""
        qassert.require(maturity >= 0.0, "negative maturity not allowed")
        if maturity <= QL_EPSILON or self._std_dev <= QL_EPSILON:
            return 0.0
        return self._discount * math.sqrt(maturity) * self._n_d

    def rho(self, maturity: float) -> float:
        """C++ parity: cpp:299-310.

        Note the C++ comment claims a ``discount *`` factor that the code does
        not apply: the body is ``maturity * (deltaForward() * forward - value())``
        and ``deltaForward()`` already carries the discount while ``value()``
        does too. Reproduced verbatim.
        """
        qassert.require(maturity >= 0.0, "negative maturity not allowed")
        return maturity * (self.delta_forward() * self._forward - self.value())

    def dividend_rho(self, maturity: float) -> float:
        """C++ parity: cpp:312-322.

        Unlike ``rho`` this one rebuilds the forward delta inline WITHOUT the
        discount and multiplies by ``discount_`` afterwards — same number,
        different route.
        """
        qassert.require(maturity >= 0.0, "negative maturity not allowed")
        delta_fwd = self._cum_d if self._alpha >= 0 else self._cum_d - 1.0
        return -maturity * self._discount * delta_fwd * self._forward

    # --- probabilities ------------------------------------------------------

    def itm_cash_probability(self) -> float:
        """C++ parity: bacheliercalculator.hpp:130-140."""
        return self._cum_d if self._alpha >= 0 else 1.0 - self._cum_d

    def itm_asset_probability(self) -> float:
        """C++ parity: bacheliercalculator.hpp:142-153.

        Identical to ``itm_cash_probability`` — the normal model has no drift
        adjustment between the two martingale measures.
        """
        return self._cum_d if self._alpha >= 0 else 1.0 - self._cum_d

    # --- strike sensitivities ----------------------------------------------

    def strike_sensitivity(self) -> float:
        """C++ parity: cpp:324-334."""
        if self._alpha >= 0:
            return -self._discount * self._cum_d
        return self._discount * (1.0 - self._cum_d)

    def strike_gamma(self) -> float:
        """C++ parity: cpp:337-347 — same for calls and puts."""
        if self._std_dev <= QL_EPSILON:
            return 0.0
        return self._discount * self._n_d / self._std_dev

    # --- cross greeks --------------------------------------------------------

    def vanna(self, maturity: float) -> float:
        """C++ parity: cpp:350-358 — ``-d n(d) sqrt(T) / sigma``.

        Note: no ``discount_`` factor, unlike every neighbouring greek.
        """
        if maturity <= QL_EPSILON or self._std_dev <= QL_EPSILON:
            return 0.0
        return -self._d * self._n_d * math.sqrt(maturity) / self._std_dev

    def volga(self, maturity: float) -> float:
        """C++ parity: cpp:361-368 — ``(d^2 / sigma) * vega``."""
        if maturity <= QL_EPSILON or self._std_dev <= QL_EPSILON:
            return 0.0
        return (self._d * self._d / self._std_dev) * self.vega(maturity)


__all__ = ["BachelierCalculator"]
