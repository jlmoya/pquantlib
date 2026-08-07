"""Heston analytic-expansion engine and the three expansion formulas.

# C++ parity: ql/pricingengines/vanilla/hestonexpansionengine.{hpp,cpp}
# (v1.43) — ``class HestonExpansionEngine : public
# GenericModelEngine<HestonModel, VanillaOption::arguments,
# VanillaOption::results>``, plus the ``HestonExpansion`` interface and its
# three implementations ``LPP2HestonExpansion``, ``LPP3HestonExpansion`` and
# ``FordeHestonExpansion``.

Instead of inverting a characteristic function, these engines produce a closed
form for the Black implied volatility as a polynomial in the log-moneyness
``x = log(K / F)``, then price with :func:`black_formula`::

    price = blackFormula(payoff, forward, vol * sqrt(term), riskFreeDiscount, 0)

Two families:

* **Lorig-Pagliarani-Pascucci** (``LPP2HestonExpansion``,
  ``LPP3HestonExpansion``) — order-2 and order-3 expansions of the implied
  *volatility* itself. Coefficients come from the authors' Mathematica
  notebook and are transliterated verbatim; they are exact rational functions
  of ``(kappa, theta, sigma, v0, rho, term)``.
* **Forde-Jacquier-Lee** (``FordeHestonExpansion``) — a small-time asymptotic
  expansion of the implied *variance*. It is the sharpest of the three near
  expiry and, per the C++ test-suite's own comment, "breaks down for long
  maturities" (the suite allows 100% relative error there). This port
  reproduces that behaviour rather than repairing it.

Notes a port gets wrong easily:

* LPP2/LPP3 ``impliedVolatility`` clamp with ``max(1e-8, vol)`` — the clamp is
  on the VOLATILITY. Forde clamps ``max(1e-8, var)`` and *then* takes the
  square root, so Forde's floor is ``1e-4``, not ``1e-8``.
* The engine passes ``blackFormula`` the RISK-FREE discount while the forward
  ``spot * dividendDiscount / riskFreeDiscount`` already carries the dividend
  discount; displacement is 0.
* ``LPP2HestonExpansion`` / ``LPP3HestonExpansion`` take their constructor
  arguments in the order ``(kappa, theta, sigma, v0, rho, term)`` but their
  private ``z*`` helpers receive them as ``(t, kappa, theta, delta, y, rho)``
  with ``delta = sigma`` and ``y = v0``. The two orderings are NOT the same;
  the names below follow the C++ exactly.

Cross-validated against C++ v1.43 in
``pquantlib/tests/pricingengines/vanilla/test_heston_expansion_engine.py``
(reference ``migration-harness/references/v143/pe/heston.json``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import IntEnum

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine

# C++ uses free ``pow`` / ``sqrt`` (and a ``fastpow`` that just forwards to
# ``pow``) inside the transliterated coefficient expressions below. Binding them
# to module-level names keeps those expressions a character-for-character match
# for the C++ source.
_pow = math.pow
_sqrt = math.sqrt


class HestonExpansion(ABC):
    """Interface for a Heston implied-volatility expansion formula.

    # C++ parity: ``class HestonExpansion`` in hestonexpansionengine.hpp:70-75
    # (v1.43) — pure-virtual ``impliedVolatility(strike, forward)``.

    During calibration an implementation is built once per implied-volatility
    surface slice (i.e. per expiry), then queried for every strike on that
    slice.
    """

    @abstractmethod
    def implied_volatility(self, strike: float, forward: float) -> float:
        """Black implied volatility for ``strike`` given ``forward``.

        # C++ parity: ``virtual Real impliedVolatility(Real strike,
        # Real forward) const = 0``.
        """


class LPP2HestonExpansion(HestonExpansion):
    """Lorig-Pagliarani-Pascucci order-2 expansion.

    # C++ parity: ``class LPP2HestonExpansion : public HestonExpansion`` in
    # hestonexpansionengine.hpp:82-95 and .cpp:106-196 (v1.43).

    ``impliedVolatility`` is the quadratic ``c0 + x*(c1 + x*c2)`` in
    ``x = log(strike / forward)``, floored at ``1e-8``.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        v0: float,
        rho: float,
        term: float,
    ) -> None:
        """Precompute the three polynomial coefficients for one expiry.

        # C++ parity: ``LPP2HestonExpansion::LPP2HestonExpansion`` in
        # hestonexpansionengine.cpp:106-115. Note the C++ passes ``sigma`` into
        # the ``z*`` helpers as ``delta`` and ``v0`` as ``y``.
        """
        self._ekt: float = math.exp(kappa * term)
        self._e2kt: float = self._ekt * self._ekt
        self._e3kt: float = self._e2kt * self._ekt
        self._e4kt: float = self._e2kt * self._e2kt
        self._coeffs: tuple[float, float, float] = (
            self._z0(term, kappa, theta, sigma, v0, rho),
            self._z1(term, kappa, theta, sigma, v0, rho),
            self._z2(term, kappa, theta, sigma, v0, rho),
        )

    def implied_volatility(self, strike: float, forward: float) -> float:
        """# C++ parity: ``LPP2HestonExpansion::impliedVolatility`` (.cpp:121-126)."""
        x = math.log(strike / forward)
        c = self._coeffs
        vol = c[0] + x * (c[1] + (x * c[2]))
        return max(1e-8, vol)

    def _z0(
        self,
        t: float,
        kappa: float,
        theta: float,
        delta: float,
        y: float,
        rho: float,
    ) -> float:
        """Constant coefficient of the LPP2 expansion polynomial.

        # C++ parity: ``LPP2HestonExpansion::z0`` in
        # hestonexpansionengine.cpp:128 (v1.43). The body below is a
        # token-for-token transliteration of the C++ expression: ``pow`` and
        # ``fastpow`` map to ``_pow``, ``sqrt`` to ``_sqrt``, and the member
        # scalars ``ekt``/``e2kt``/``e3kt``/``e4kt`` are bound as locals so the
        # arithmetic reads exactly as it does in C++.
        """
        ekt = self._ekt
        e2kt = self._e2kt
        e3kt = self._e3kt
        return (
            (4 * _pow(delta, 2) * kappa * (-theta -4 * ekt * (theta +kappa * t * (theta -y)) +e2kt *
            ((5 -2 * kappa * t) * theta -2 * y) +2 * y) * ((1 +ekt * (-1 +kappa * t)) * theta +(-1
            +ekt) * y) +128 * ekt * _pow(kappa, 3) * _pow((1 +ekt * (-1 +kappa * t)) * theta +(-1
            +ekt) * y, 2) +32 * delta * ekt * _pow(kappa, 2) * rho * ((1 +ekt * (-1 +kappa * t)) *
            theta +(-1 +ekt) * y) * ((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa
            * t) * y) +_pow(delta, 2) * ekt * _pow(rho, 2) * (-theta +kappa * t * theta +(theta -y)
            / ekt +y) * _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) *
            y, 2) +(48 * _pow(delta, 2) * e2kt * _pow(kappa, 2) * _pow(rho, 2) * _pow((2 +kappa * t
            +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 2)) / ((1 +ekt * (-1 +kappa *
            t)) * theta +(-1 +ekt) * y) -_pow(delta, 2) * _pow(rho, 2) * ((1 +ekt * (-1 +kappa * t))
            * theta +(-1 +ekt) * y) * _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt
            -kappa * t) * y, 2) +2 * _pow(delta, 2) * kappa * ((1 +ekt * (-1 +kappa * t)) * theta
            +(-1 +ekt) * y) * (theta -2 * y +e2kt * (-5 * theta +2 * kappa * t * theta +2 * y +8 *
            _pow(rho, 2) * ((-3 +kappa * t) * theta +y)) +4 * ekt * (theta +kappa * t * theta -kappa
            * t * y +_pow(rho, 2) * ((6 +kappa * t * (4 +kappa * t)) * theta -(2 +kappa * t * (2
            +kappa * t)) * y))) -(8 * _pow(delta, 2) * _pow(kappa, 2) * ((1 +ekt * (-1 +kappa * t))
            * theta +(-1 +ekt) * y) * (theta -2 * y +e2kt * (-5 * theta +2 * kappa * t * theta +2 *
            y +8 * _pow(rho, 2) * ((-3 +kappa * t) * theta +y)) +4 * ekt * (theta +kappa * t * theta
            -kappa * t * y +_pow(rho, 2) * ((6 +kappa * t * (4 +kappa * t)) * theta -(2 +kappa * t *
            (2 +kappa * t)) * y)))) / (-theta +kappa * t * theta +(theta -y) / ekt +y)) / (128. *
            e3kt * _pow(kappa, 5) * _pow(t, 2) * _pow((-theta +kappa * t * theta +(theta -y) / ekt
            +y) / (kappa * t), 1.5))
        )

    def _z1(
        self,
        t: float,
        kappa: float,
        theta: float,
        delta: float,
        y: float,
        rho: float,
    ) -> float:
        """Coefficient of x of the LPP2 expansion polynomial.

        # C++ parity: ``LPP2HestonExpansion::z1`` in
        # hestonexpansionengine.cpp:167 (v1.43). The body below is a
        # token-for-token transliteration of the C++ expression: ``pow`` and
        # ``fastpow`` map to ``_pow``, ``sqrt`` to ``_sqrt``, and the member
        # scalars ``ekt``/``e2kt``/``e3kt``/``e4kt`` are bound as locals so the
        # arithmetic reads exactly as it does in C++.
        """
        ekt = self._ekt
        e2kt = self._e2kt
        return (
            (delta * rho * (-(delta * _pow(-1 +ekt, 2) * rho * (4 * theta -y) * y) +2 * ekt *
            _pow(kappa, 3) * _pow(t, 2) * theta * ((2 +2 * ekt +delta * rho * t) * theta -(2 +delta
            * rho * t) * y) -2 * (-1 +ekt) * kappa * (2 * theta -y) * ((-1 +ekt) * (-2 +delta * rho
            * t) * theta +(-2 +2 * ekt +delta * rho * t) * y) +_pow(kappa, 2) * t * ((-1 +ekt) * (-4
            +delta * rho * t +ekt * (-12 +delta * rho * t)) * _pow(theta, 2) +2 * (-4 +4 * e2kt
            +delta * rho * t +3 * delta * ekt * rho * t) * theta * y -(-4 +delta * rho * t +2 * ekt
            * (2 +delta * rho * t)) * _pow(y, 2)))) / (8. * _pow(kappa, 2) * t * _sqrt((-theta
            +kappa * t * theta +(theta -y) / ekt +y) / (kappa * t)) * _pow((1 +ekt * (-1 +kappa *
            t)) * theta +(-1 +ekt) * y, 2))
        )

    def _z2(
        self,
        t: float,
        kappa: float,
        theta: float,
        delta: float,
        y: float,
        rho: float,
    ) -> float:
        """Coefficient of x^2 of the LPP2 expansion polynomial.

        # C++ parity: ``LPP2HestonExpansion::z2`` in
        # hestonexpansionengine.cpp:184 (v1.43). The body below is a
        # token-for-token transliteration of the C++ expression: ``pow`` and
        # ``fastpow`` map to ``_pow``, ``sqrt`` to ``_sqrt``, and the member
        # scalars ``ekt``/``e2kt``/``e3kt``/``e4kt`` are bound as locals so the
        # arithmetic reads exactly as it does in C++.
        """
        ekt = self._ekt
        e2kt = self._e2kt
        return (
            (_pow(delta, 2) * _sqrt((-theta +kappa * t * theta +(theta -y) / ekt +y) / (kappa * t))
            * (-12 * _pow(rho, 2) * _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt
            -kappa * t) * y, 2) +(-theta +kappa * t * theta +(theta -y) / ekt +y) * (theta -2 * y
            +e2kt * (-5 * theta +2 * kappa * t * theta +2 * y +8 * _pow(rho, 2) * ((-3 +kappa * t) *
            theta +y)) +4 * ekt * (theta +kappa * t * theta -kappa * t * y +_pow(rho, 2) * ((6
            +kappa * t * (4 +kappa * t)) * theta -(2 +kappa * t * (2 +kappa * t)) * y))))) / (16. *
            e2kt * _pow(-theta +kappa * t * theta +(theta -y) / ekt +y, 4))
        )


class LPP3HestonExpansion(HestonExpansion):
    """Lorig-Pagliarani-Pascucci order-3 expansion.

    # C++ parity: ``class LPP3HestonExpansion : public HestonExpansion`` in
    # hestonexpansionengine.hpp:102-117 and .cpp:224-738 (v1.43).

    ``impliedVolatility`` is the cubic ``c0 + x*(c1 + x*(c2 + x*c3))`` in
    ``x = log(strike / forward)``, floored at ``1e-8``.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        v0: float,
        rho: float,
        term: float,
    ) -> None:
        """Precompute the four polynomial coefficients for one expiry.

        # C++ parity: ``LPP3HestonExpansion::LPP3HestonExpansion``
        # (.cpp:721-731).
        """
        self._ekt: float = math.exp(kappa * term)
        self._e2kt: float = self._ekt * self._ekt
        self._e3kt: float = self._e2kt * self._ekt
        self._e4kt: float = self._e2kt * self._e2kt
        self._coeffs: tuple[float, float, float, float] = (
            self._z0(term, kappa, theta, sigma, v0, rho),
            self._z1(term, kappa, theta, sigma, v0, rho),
            self._z2(term, kappa, theta, sigma, v0, rho),
            self._z3(term, kappa, theta, sigma, v0, rho),
        )

    def implied_volatility(self, strike: float, forward: float) -> float:
        """# C++ parity: ``LPP3HestonExpansion::impliedVolatility`` (.cpp:733-738)."""
        x = math.log(strike / forward)
        c = self._coeffs
        vol = c[0] + x * (c[1] + x * (c[2] + x * (c[3])))
        return max(1e-8, vol)

    def _z0(
        self,
        t: float,
        kappa: float,
        theta: float,
        delta: float,
        y: float,
        rho: float,
    ) -> float:
        """Constant coefficient of the LPP3 expansion polynomial.

        # C++ parity: ``LPP3HestonExpansion::z0`` in
        # hestonexpansionengine.cpp:224 (v1.43). The body below is a
        # token-for-token transliteration of the C++ expression: ``pow`` and
        # ``fastpow`` map to ``_pow``, ``sqrt`` to ``_sqrt``, and the member
        # scalars ``ekt``/``e2kt``/``e3kt``/``e4kt`` are bound as locals so the
        # arithmetic reads exactly as it does in C++.
        """
        ekt = self._ekt
        e2kt = self._e2kt
        e3kt = self._e3kt
        e4kt = self._e4kt
        return (
            (96 * _pow(delta, 2) * ekt * _pow(kappa, 3) * (-theta -4 * ekt * (theta +kappa * t *
            (theta -y)) +e2kt * ((5 -2 * kappa * t) * theta -2 * y) +2 * y) * ((1 +ekt * (-1 +kappa
            * t)) * theta +(-1 +ekt) * y) +3072 * e2kt * _pow(kappa, 5) * _pow((1 +ekt * (-1 +kappa
            * t)) * theta +(-1 +ekt) * y, 2) +96 * _pow(delta, 3) * ekt * _pow(kappa, 2) * rho * ((1
            +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (-2 * theta -kappa * t * theta -2 *
            ekt * (2 +kappa * t) * (2 * theta +kappa * t * (theta -y)) +e2kt * ((10 -3 * kappa * t)
            * theta -3 * y) +3 * y +2 * kappa * t * y) +768 * delta * e2kt * _pow(kappa, 4) * rho *
            ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * ((2 +kappa * t +ekt * (-2 +kappa *
            t)) * theta +(-1 +ekt -kappa * t) * y) +6 * _pow(delta, 3) * kappa * rho * (-theta -4 *
            ekt * (theta +kappa * t * (theta -y)) +e2kt * ((5 -2 * kappa * t) * theta -2 * y) +2 *
            y) * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * ((2 +kappa * t +ekt * (-2
            +kappa * t)) * theta +(-1 +ekt -kappa * t) * y) +24 * _pow(delta, 2) * e2kt *
            _pow(kappa, 2) * _pow(rho, 2) * (-theta +kappa * t * theta +(theta -y) / ekt +y) *
            _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 2) +(1152
            * _pow(delta, 2) * e3kt * _pow(kappa, 4) * _pow(rho, 2) * _pow((2 +kappa * t +ekt * (-2
            +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 2)) / ((1 +ekt * (-1 +kappa * t)) *
            theta +(-1 +ekt) * y) -24 * _pow(delta, 2) * ekt * _pow(kappa, 2) * _pow(rho, 2) * ((1
            +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * _pow((2 +kappa * t +ekt * (-2 +kappa *
            t)) * theta +(-1 +ekt -kappa * t) * y, 2) +80 * _pow(delta, 3) * ekt * kappa * _pow(rho,
            3) * _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 3)
            +_pow(delta, 3) * ekt * _pow(rho, 3) * (-theta +kappa * t * theta +(theta -y) / ekt +y)
            * _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 3)
            -(1440 * _pow(delta, 3) * e3kt * _pow(kappa, 3) * _pow(rho, 3) * _pow((2 +kappa * t +ekt
            * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 3)) / _pow((1 +ekt * (-1 +kappa *
            t)) * theta +(-1 +ekt) * y, 2) -(528 * _pow(delta, 3) * e2kt * _pow(kappa, 2) *
            _pow(rho, 3) * _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t)
            * y, 3)) / ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) -3 * _pow(delta, 3) *
            _pow(rho, 3) * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * _pow((2 +kappa * t
            +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 3) +384 * _pow(delta, 3) *
            e2kt * _pow(kappa, 3) * rho * ((2 +kappa * t +2 * ekt * _pow(2 +kappa * t, 2) +e2kt *
            (-10 +3 * kappa * t)) * theta +(-3 +3 * e2kt -2 * kappa * t -2 * ekt * kappa * t * (2
            +kappa * t)) * y) -(576 * _pow(delta, 3) * e2kt * _pow(kappa, 3) * rho * ((2 +kappa * t
            +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y) * ((1 +e2kt * (-5 +2 * kappa
            * t +4 * _pow(rho, 2) * (-3 +kappa * t)) +2 * ekt * (2 +2 * kappa * t +_pow(rho, 2) * (6
            +4 * kappa * t +_pow(kappa, 2) * _pow(t, 2)))) * theta +2 * (-1 +e2kt * (1 +2 *
            _pow(rho, 2)) -ekt * (2 * kappa * t +_pow(rho, 2) * (2 +2 * kappa * t +_pow(kappa, 2) *
            _pow(t, 2)))) * y)) / ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) +_pow(delta,
            3) * rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * ((2 +kappa * t +ekt *
            (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y) * (theta * (12 * ekt * _pow(kappa,
            3) * _pow(rho, 2) * _pow(t, 2) +8 * _pow(-1 +ekt, 2) * _pow(rho, 2) * theta -(-1 +ekt) *
            kappa * (3 +8 * _pow(rho, 2) * t * theta +ekt * (15 +8 * _pow(rho, 2) * (9 +t * theta)))
            +2 * _pow(kappa, 2) * t * (_pow(rho, 2) * t * theta +2 * ekt * (3 +_pow(rho, 2) * (12 +t
            * theta)) +e2kt * (3 +_pow(rho, 2) * (12 +t * theta)))) -2 * (6 * ekt * _pow(kappa, 3) *
            _pow(rho, 2) * _pow(t, 2) +4 * _pow(-1 +ekt, 2) * _pow(rho, 2) * theta +2 * _pow(kappa,
            2) * t * (_pow(rho, 2) * t * theta +ekt * (3 +_pow(rho, 2) * (6 +t * theta))) -(-1 +ekt)
            * kappa * (3 +6 * _pow(rho, 2) * t * theta +ekt * (3 +2 * _pow(rho, 2) * (6 +t *
            theta)))) * y +2 * _pow(rho, 2) * _pow(1 -ekt +kappa * t, 2) * _pow(y, 2)) -(40 *
            _pow(delta, 3) * kappa * rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * ((2
            +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y) * (theta * (12 *
            ekt * _pow(kappa, 3) * _pow(rho, 2) * _pow(t, 2) +8 * _pow(-1 +ekt, 2) * _pow(rho, 2) *
            theta -(-1 +ekt) * kappa * (3 +8 * _pow(rho, 2) * t * theta +ekt * (15 +8 * _pow(rho, 2)
            * (9 +t * theta))) +2 * _pow(kappa, 2) * t * (_pow(rho, 2) * t * theta +2 * ekt * (3
            +_pow(rho, 2) * (12 +t * theta)) +e2kt * (3 +_pow(rho, 2) * (12 +t * theta)))) -2 * (6 *
            ekt * _pow(kappa, 3) * _pow(rho, 2) * _pow(t, 2) +4 * _pow(-1 +ekt, 2) * _pow(rho, 2) *
            theta +2 * _pow(kappa, 2) * t * (_pow(rho, 2) * t * theta +ekt * (3 +_pow(rho, 2) * (6
            +t * theta))) -(-1 +ekt) * kappa * (3 +6 * _pow(rho, 2) * t * theta +ekt * (3 +2 *
            _pow(rho, 2) * (6 +t * theta)))) * y +2 * _pow(rho, 2) * _pow(1 -ekt +kappa * t, 2) *
            _pow(y, 2))) / (-theta +kappa * t * theta +(theta -y) / ekt +y) -12 * _pow(delta, 3) *
            kappa * rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (2 * theta +kappa *
            t * theta -y -kappa * t * y +ekt * ((-2 +kappa * t) * theta +y)) * (theta -2 * y +e2kt *
            (-5 * theta +2 * kappa * t * theta +2 * y +4 * _pow(rho, 2) * ((-3 +kappa * t) * theta
            +y)) +2 * ekt * (2 * (theta +kappa * t * (theta -y)) +_pow(rho, 2) * ((6 +kappa * t * (4
            +kappa * t)) * theta -(2 +kappa * t * (2 +kappa * t)) * y))) +(288 * _pow(delta, 3) *
            _pow(kappa, 2) * rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (2 * theta
            +kappa * t * theta -y -kappa * t * y +ekt * ((-2 +kappa * t) * theta +y)) * (theta -2 *
            y +e2kt * (-5 * theta +2 * kappa * t * theta +2 * y +4 * _pow(rho, 2) * ((-3 +kappa * t)
            * theta +y)) +2 * ekt * (2 * (theta +kappa * t * (theta -y)) +_pow(rho, 2) * ((6 +kappa
            * t * (4 +kappa * t)) * theta -(2 +kappa * t * (2 +kappa * t)) * y)))) / (-theta +kappa
            * t * theta +(theta -y) / ekt +y) +48 * _pow(delta, 2) * ekt * _pow(kappa, 3) * ((1 +ekt
            * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (theta -2 * y +e2kt * (-5 * theta +2 *
            kappa * t * theta +2 * y +8 * _pow(rho, 2) * ((-3 +kappa * t) * theta +y)) +4 * ekt *
            (theta +kappa * t * theta -kappa * t * y +_pow(rho, 2) * ((6 +kappa * t * (4 +kappa *
            t)) * theta -(2 +kappa * t * (2 +kappa * t)) * y))) -(192 * _pow(delta, 2) * ekt *
            _pow(kappa, 4) * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (theta -2 * y
            +e2kt * (-5 * theta +2 * kappa * t * theta +2 * y +8 * _pow(rho, 2) * ((-3 +kappa * t) *
            theta +y)) +4 * ekt * (theta +kappa * t * theta -kappa * t * y +_pow(rho, 2) * ((6
            +kappa * t * (4 +kappa * t)) * theta -(2 +kappa * t * (2 +kappa * t)) * y)))) / (-theta
            +kappa * t * theta +(theta -y) / ekt +y) +3 * _pow(delta, 3) * kappa * rho * ((1 +ekt *
            (-1 +kappa * t)) * theta +(-1 +ekt) * y) * ((2 +kappa * t +ekt * (-2 +kappa * t)) *
            theta +(-1 +ekt -kappa * t) * y) * (theta -2 * y +e2kt * (-5 * theta +2 * kappa * t *
            theta +2 * y +8 * _pow(rho, 2) * ((-3 +kappa * t) * theta +y)) +4 * ekt * (theta +kappa
            * t * theta -kappa * t * y +_pow(rho, 2) * ((6 +kappa * t * (4 +kappa * t)) * theta -(2
            +kappa * t * (2 +kappa * t)) * y))) -(12 * _pow(delta, 3) * _pow(kappa, 2) * rho * ((1
            +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * ((2 +kappa * t +ekt * (-2 +kappa * t))
            * theta +(-1 +ekt -kappa * t) * y) * (theta -2 * y +e2kt * (-5 * theta +2 * kappa * t *
            theta +2 * y +8 * _pow(rho, 2) * ((-3 +kappa * t) * theta +y)) +4 * ekt * (theta +kappa
            * t * theta -kappa * t * y +_pow(rho, 2) * ((6 +kappa * t * (4 +kappa * t)) * theta -(2
            +kappa * t * (2 +kappa * t)) * y)))) / (-theta +kappa * t * theta +(theta -y) / ekt +y)
            +4 * _pow(delta, 3) * kappa * rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y)
            * (3 * (theta -2 * y) * ((2 +kappa * t) * theta -(1 +kappa * t) * y) +3 * ekt * (6 *
            _pow(theta, 2) +theta * y -2 * _pow(y, 2) +kappa * (13 * t * _pow(theta, 2) +theta * (8
            -18 * t * y) +4 * y * (-3 +t * y)) +4 * _pow(kappa, 2) * t * (theta +t * _pow(theta, 2)
            -2 * t * theta * y +y * (-2 +t * y))) +3 * e3kt * (10 * _pow(theta, 2) +2 * _pow(kappa,
            2) * t * theta * (6 +8 * _pow(rho, 2) +t * theta) -9 * theta * y +2 * _pow(y, 2) +kappa
            * (-9 * t * _pow(theta, 2) +4 * (3 +4 * _pow(rho, 2)) * y +theta * (-40 -64 * _pow(rho,
            2) +4 * t * y))) +e2kt * (-54 * _pow(theta, 2) +8 * _pow(kappa, 4) * _pow(rho, 2) *
            _pow(t, 3) * (theta -y) +39 * theta * y -6 * _pow(y, 2) +24 * _pow(kappa, 3) * _pow(t,
            2) * (theta +2 * _pow(rho, 2) * theta -(1 +_pow(rho, 2)) * y) +6 * _pow(kappa, 2) * t *
            (3 * t * _pow(theta, 2) -8 * (1 +_pow(rho, 2)) * y +theta * (16 +24 * _pow(rho, 2) -3 *
            t * y)) -3 * kappa * (5 * t * _pow(theta, 2) +2 * y * (8 * _pow(rho, 2) +3 * t * y)
            -theta * (32 +64 * _pow(rho, 2) +17 * t * y)))) -(48 * _pow(delta, 3) * _pow(kappa, 2) *
            rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (3 * (theta -2 * y) * ((2
            +kappa * t) * theta -(1 +kappa * t) * y) +3 * ekt * (6 * _pow(theta, 2) +theta * y -2 *
            _pow(y, 2) +kappa * (13 * t * _pow(theta, 2) +theta * (8 -18 * t * y) +4 * y * (-3 +t *
            y)) +4 * _pow(kappa, 2) * t * (theta +t * _pow(theta, 2) -2 * t * theta * y +y * (-2 +t
            * y))) +3 * e3kt * (10 * _pow(theta, 2) +2 * _pow(kappa, 2) * t * theta * (6 +8 *
            _pow(rho, 2) +t * theta) -9 * theta * y +2 * _pow(y, 2) +kappa * (-9 * t * _pow(theta,
            2) +4 * (3 +4 * _pow(rho, 2)) * y +theta * (-40 -64 * _pow(rho, 2) +4 * t * y))) +e2kt *
            (-54 * _pow(theta, 2) +8 * _pow(kappa, 4) * _pow(rho, 2) * _pow(t, 3) * (theta -y) +39 *
            theta * y -6 * _pow(y, 2) +24 * _pow(kappa, 3) * _pow(t, 2) * (theta +2 * _pow(rho, 2) *
            theta -(1 +_pow(rho, 2)) * y) +6 * _pow(kappa, 2) * t * (3 * t * _pow(theta, 2) -8 * (1
            +_pow(rho, 2)) * y +theta * (16 +24 * _pow(rho, 2) -3 * t * y)) -3 * kappa * (5 * t *
            _pow(theta, 2) +2 * y * (8 * _pow(rho, 2) +3 * t * y) -theta * (32 +64 * _pow(rho, 2)
            +17 * t * y))))) / (-theta +kappa * t * theta +(theta -y) / ekt +y) +(240 * _pow(delta,
            3) * e2kt * _pow(kappa, 2) * rho * ((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1
            +ekt -kappa * t) * y) * (12 * ekt * _pow(kappa, 3) * _pow(rho, 2) * _pow(t, 2) * (theta
            -y) +2 * _pow(-1 +ekt, 2) * _pow(rho, 2) * _pow(-2 * theta +y, 2) -(-1 +ekt) * kappa *
            (8 * (1 +ekt) * _pow(rho, 2) * t * _pow(theta, 2) +2 * y * (-3 -3 * ekt * (1 +4 *
            _pow(rho, 2)) +2 * _pow(rho, 2) * t * y) +theta * (3 -12 * _pow(rho, 2) * t * y +ekt *
            (15 +_pow(rho, 2) * (72 -4 * t * y)))) +2 * _pow(kappa, 2) * t * (e2kt * theta * (3
            +_pow(rho, 2) * (12 +t * theta)) +_pow(rho, 2) * t * _pow(theta -y, 2) +2 * ekt *
            (_pow(rho, 2) * t * _pow(theta, 2) -3 * (y +2 * _pow(rho, 2) * y) +theta * (3 +_pow(rho,
            2) * (12 -t * y)))))) / ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y)) / (3072. *
            e4kt * _pow(kappa, 7) * _pow(t, 2) * _pow((-theta +kappa * t * theta +(theta -y) / ekt
            +y) / (kappa * t), 1.5))
        )

    def _z1(
        self,
        t: float,
        kappa: float,
        theta: float,
        delta: float,
        y: float,
        rho: float,
    ) -> float:
        """Coefficient of x of the LPP3 expansion polynomial.

        # C++ parity: ``LPP3HestonExpansion::z1`` in
        # hestonexpansionengine.cpp:422 (v1.43). The body below is a
        # token-for-token transliteration of the C++ expression: ``pow`` and
        # ``fastpow`` map to ``_pow``, ``sqrt`` to ``_sqrt``, and the member
        # scalars ``ekt``/``e2kt``/``e3kt``/``e4kt`` are bound as locals so the
        # arithmetic reads exactly as it does in C++.
        """
        ekt = self._ekt
        e2kt = self._e2kt
        e3kt = self._e3kt
        return (
            (delta * (768 * e2kt * _pow(kappa, 4) * rho * ((2 +kappa * t +ekt * (-2 +kappa * t)) *
            theta +(-1 +ekt -kappa * t) * y) -(576 * delta * e2kt * _pow(kappa, 3) * _pow(rho, 2) *
            _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 2)) / ((1
            +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) -10 * _pow(delta, 2) * _pow(rho, 3) *
            _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 3) +(6 *
            _pow(delta, 2) * kappa * _pow(rho, 3) * _pow((2 +kappa * t +ekt * (-2 +kappa * t)) *
            theta +(-1 +ekt -kappa * t) * y, 3)) / (-theta +kappa * t * theta +(theta -y) / ekt +y)
            -(3360 * _pow(delta, 2) * e3kt * _pow(kappa, 3) * _pow(rho, 3) * _pow((2 +kappa * t +ekt
            * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y, 3)) / _pow((1 +ekt * (-1 +kappa *
            t)) * theta +(-1 +ekt) * y, 3) -(288 * _pow(delta, 2) * e2kt * _pow(kappa, 2) *
            _pow(rho, 3) * _pow((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t)
            * y, 3)) / _pow((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y, 2) +(234 *
            _pow(delta, 2) * ekt * kappa * _pow(rho, 3) * _pow((2 +kappa * t +ekt * (-2 +kappa * t))
            * theta +(-1 +ekt -kappa * t) * y, 3)) / ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt)
            * y) -96 * delta * ekt * _pow(kappa, 3) * ((1 +4 * ekt * (1 +kappa * t) +e2kt * (-5 +2 *
            kappa * t)) * theta +2 * (-1 +e2kt -2 * ekt * kappa * t) * y) -12 * _pow(delta, 2) *
            kappa * rho * ((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y)
            * ((1 +4 * ekt * (1 +kappa * t) +e2kt * (-5 +2 * kappa * t)) * theta +2 * (-1 +e2kt -2 *
            ekt * kappa * t) * y) -192 * _pow(delta, 2) * ekt * _pow(kappa, 2) * rho * ((2 +kappa *
            t +2 * ekt * _pow(2 +kappa * t, 2) +e2kt * (-10 +3 * kappa * t)) * theta +(-3 +3 * e2kt
            -2 * kappa * t -2 * ekt * kappa * t * (2 +kappa * t)) * y) -(12 * _pow(delta, 2) * ekt *
            _pow(kappa, 2) * rho * ((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa *
            t) * y) * ((1 +e2kt * (-5 +2 * kappa * t +8 * _pow(rho, 2) * (-3 +kappa * t)) +4 * ekt *
            (1 +kappa * t +_pow(rho, 2) * (6 +4 * kappa * t +_pow(kappa, 2) * _pow(t, 2)))) * theta
            +2 * (-1 +e2kt * (1 +4 * _pow(rho, 2)) -2 * ekt * (kappa * t +_pow(rho, 2) * (2 +2 *
            kappa * t +_pow(kappa, 2) * _pow(t, 2)))) * y)) / ((1 +ekt * (-1 +kappa * t)) * theta
            +(-1 +ekt) * y) +(576 * _pow(delta, 2) * ekt * _pow(kappa, 2) * rho * ((2 +kappa * t
            +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y) * ((1 +e2kt * (-5 +2 * kappa
            * t +4 * _pow(rho, 2) * (-3 +kappa * t)) +2 * ekt * (2 +2 * kappa * t +_pow(rho, 2) * (6
            +4 * kappa * t +_pow(kappa, 2) * _pow(t, 2)))) * theta +2 * (-1 +e2kt * (1 +2 *
            _pow(rho, 2)) -ekt * (2 * kappa * t +_pow(rho, 2) * (2 +2 * kappa * t +_pow(kappa, 2) *
            _pow(t, 2)))) * y)) / ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) +(5 *
            _pow(delta, 2) * rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * ((2 +kappa
            * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y) * (theta * (12 * ekt *
            _pow(kappa, 3) * _pow(rho, 2) * _pow(t, 2) +8 * _pow(-1 +ekt, 2) * _pow(rho, 2) * theta
            -(-1 +ekt) * kappa * (3 +8 * _pow(rho, 2) * t * theta +ekt * (15 +8 * _pow(rho, 2) * (9
            +t * theta))) +2 * _pow(kappa, 2) * t * (_pow(rho, 2) * t * theta +2 * ekt * (3
            +_pow(rho, 2) * (12 +t * theta)) +e2kt * (3 +_pow(rho, 2) * (12 +t * theta)))) -2 * (6 *
            ekt * _pow(kappa, 3) * _pow(rho, 2) * _pow(t, 2) +4 * _pow(-1 +ekt, 2) * _pow(rho, 2) *
            theta +2 * _pow(kappa, 2) * t * (_pow(rho, 2) * t * theta +ekt * (3 +_pow(rho, 2) * (6
            +t * theta))) -(-1 +ekt) * kappa * (3 +6 * _pow(rho, 2) * t * theta +ekt * (3 +2 *
            _pow(rho, 2) * (6 +t * theta)))) * y +2 * _pow(rho, 2) * _pow(1 -ekt +kappa * t, 2) *
            _pow(y, 2))) / (ekt * (-theta +kappa * t * theta +(theta -y) / ekt +y)) -(48 *
            _pow(delta, 2) * kappa * rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (2
            * theta +kappa * t * theta -y -kappa * t * y +ekt * ((-2 +kappa * t) * theta +y)) *
            (theta -2 * y +e2kt * (-5 * theta +2 * kappa * t * theta +2 * y +4 * _pow(rho, 2) * ((-3
            +kappa * t) * theta +y)) +2 * ekt * (2 * (theta +kappa * t * (theta -y)) +_pow(rho, 2) *
            ((6 +kappa * t * (4 +kappa * t)) * theta -(2 +kappa * t * (2 +kappa * t)) * y)))) / (ekt
            * (-theta +kappa * t * theta +(theta -y) / ekt +y)) +(96 * delta * _pow(kappa, 3) * ((1
            +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (theta -2 * y +e2kt * (-5 * theta +2 *
            kappa * t * theta +2 * y +8 * _pow(rho, 2) * ((-3 +kappa * t) * theta +y)) +4 * ekt *
            (theta +kappa * t * theta -kappa * t * y +_pow(rho, 2) * ((6 +kappa * t * (4 +kappa *
            t)) * theta -(2 +kappa * t * (2 +kappa * t)) * y)))) / (-theta +kappa * t * theta
            +(theta -y) / ekt +y) +(9 * _pow(delta, 2) * kappa * rho * ((1 +ekt * (-1 +kappa * t)) *
            theta +(-1 +ekt) * y) * ((2 +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa
            * t) * y) * (theta -2 * y +e2kt * (-5 * theta +2 * kappa * t * theta +2 * y +8 *
            _pow(rho, 2) * ((-3 +kappa * t) * theta +y)) +4 * ekt * (theta +kappa * t * theta -kappa
            * t * y +_pow(rho, 2) * ((6 +kappa * t * (4 +kappa * t)) * theta -(2 +kappa * t * (2
            +kappa * t)) * y)))) / (ekt * (-theta +kappa * t * theta +(theta -y) / ekt +y)) -(48 *
            _pow(delta, 2) * ekt * _pow(kappa, 2) * rho * (3 * (theta -2 * y) * ((2 +kappa * t) *
            theta -(1 +kappa * t) * y) +3 * ekt * (6 * _pow(theta, 2) +theta * y -2 * _pow(y, 2)
            +kappa * (13 * t * _pow(theta, 2) +theta * (8 -18 * t * y) +4 * y * (-3 +t * y)) +4 *
            _pow(kappa, 2) * t * (theta +t * _pow(theta, 2) -2 * t * theta * y +y * (-2 +t * y))) +3
            * e3kt * (10 * _pow(theta, 2) +2 * _pow(kappa, 2) * t * theta * (6 +8 * _pow(rho, 2) +t
            * theta) -9 * theta * y +2 * _pow(y, 2) +kappa * (-9 * t * _pow(theta, 2) +4 * (3 +4 *
            _pow(rho, 2)) * y +theta * (-40 -64 * _pow(rho, 2) +4 * t * y))) +e2kt * (-54 *
            _pow(theta, 2) +8 * _pow(kappa, 4) * _pow(rho, 2) * _pow(t, 3) * (theta -y) +39 * theta
            * y -6 * _pow(y, 2) +24 * _pow(kappa, 3) * _pow(t, 2) * (theta +2 * _pow(rho, 2) * theta
            -(1 +_pow(rho, 2)) * y) +6 * _pow(kappa, 2) * t * (3 * t * _pow(theta, 2) -8 * (1
            +_pow(rho, 2)) * y +theta * (16 +24 * _pow(rho, 2) -3 * t * y)) -3 * kappa * (5 * t *
            _pow(theta, 2) +2 * y * (8 * _pow(rho, 2) +3 * t * y) -theta * (32 +64 * _pow(rho, 2)
            +17 * t * y))))) / ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) +(12 *
            _pow(delta, 2) * kappa * rho * ((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y) * (3
            * (theta -2 * y) * ((2 +kappa * t) * theta -(1 +kappa * t) * y) +3 * ekt * (6 *
            _pow(theta, 2) +theta * y -2 * _pow(y, 2) +kappa * (13 * t * _pow(theta, 2) +theta * (8
            -18 * t * y) +4 * y * (-3 +t * y)) +4 * _pow(kappa, 2) * t * (theta +t * _pow(theta, 2)
            -2 * t * theta * y +y * (-2 +t * y))) +3 * e3kt * (10 * _pow(theta, 2) +2 * _pow(kappa,
            2) * t * theta * (6 +8 * _pow(rho, 2) +t * theta) -9 * theta * y +2 * _pow(y, 2) +kappa
            * (-9 * t * _pow(theta, 2) +4 * (3 +4 * _pow(rho, 2)) * y +theta * (-40 -64 * _pow(rho,
            2) +4 * t * y))) +e2kt * (-54 * _pow(theta, 2) +8 * _pow(kappa, 4) * _pow(rho, 2) *
            _pow(t, 3) * (theta -y) +39 * theta * y -6 * _pow(y, 2) +24 * _pow(kappa, 3) * _pow(t,
            2) * (theta +2 * _pow(rho, 2) * theta -(1 +_pow(rho, 2)) * y) +6 * _pow(kappa, 2) * t *
            (3 * t * _pow(theta, 2) -8 * (1 +_pow(rho, 2)) * y +theta * (16 +24 * _pow(rho, 2) -3 *
            t * y)) -3 * kappa * (5 * t * _pow(theta, 2) +2 * y * (8 * _pow(rho, 2) +3 * t * y)
            -theta * (32 +64 * _pow(rho, 2) +17 * t * y))))) / (ekt * (-theta +kappa * t * theta
            +(theta -y) / ekt +y)) +(240 * _pow(delta, 2) * e2kt * _pow(kappa, 2) * rho * ((2 +kappa
            * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y) * (12 * ekt * _pow(kappa,
            3) * _pow(rho, 2) * _pow(t, 2) * (theta -y) +2 * _pow(-1 +ekt, 2) * _pow(rho, 2) *
            _pow(-2 * theta +y, 2) -(-1 +ekt) * kappa * (8 * (1 +ekt) * _pow(rho, 2) * t *
            _pow(theta, 2) +2 * y * (-3 -3 * ekt * (1 +4 * _pow(rho, 2)) +2 * _pow(rho, 2) * t * y)
            +theta * (3 -12 * _pow(rho, 2) * t * y +ekt * (15 +_pow(rho, 2) * (72 -4 * t * y)))) +2
            * _pow(kappa, 2) * t * (e2kt * theta * (3 +_pow(rho, 2) * (12 +t * theta)) +_pow(rho, 2)
            * t * _pow(theta -y, 2) +2 * ekt * (_pow(rho, 2) * t * _pow(theta, 2) -3 * (y +2 *
            _pow(rho, 2) * y) +theta * (3 +_pow(rho, 2) * (12 -t * y)))))) / _pow((1 +ekt * (-1
            +kappa * t)) * theta +(-1 +ekt) * y, 2) -(120 * _pow(delta, 2) * ekt * kappa * rho * ((2
            +kappa * t +ekt * (-2 +kappa * t)) * theta +(-1 +ekt -kappa * t) * y) * (12 * ekt *
            _pow(kappa, 3) * _pow(rho, 2) * _pow(t, 2) * (theta -y) +2 * _pow(-1 +ekt, 2) *
            _pow(rho, 2) * _pow(-2 * theta +y, 2) -(-1 +ekt) * kappa * (8 * (1 +ekt) * _pow(rho, 2)
            * t * _pow(theta, 2) +2 * y * (-3 -3 * ekt * (1 +4 * _pow(rho, 2)) +2 * _pow(rho, 2) * t
            * y) +theta * (3 -12 * _pow(rho, 2) * t * y +ekt * (15 +_pow(rho, 2) * (72 -4 * t *
            y)))) +2 * _pow(kappa, 2) * t * (e2kt * theta * (3 +_pow(rho, 2) * (12 +t * theta))
            +_pow(rho, 2) * t * _pow(theta -y, 2) +2 * ekt * (_pow(rho, 2) * t * _pow(theta, 2) -3 *
            (y +2 * _pow(rho, 2) * y) +theta * (3 +_pow(rho, 2) * (12 -t * y)))))) / ((1 +ekt * (-1
            +kappa * t)) * theta +(-1 +ekt) * y))) / (1536. * e3kt * _pow(kappa, 6) * _pow(t, 2) *
            _pow((-theta +kappa * t * theta +(theta -y) / ekt +y) / (kappa * t), 1.5))
        )

    def _z2(
        self,
        t: float,
        kappa: float,
        theta: float,
        delta: float,
        y: float,
        rho: float,
    ) -> float:
        """Coefficient of x^2 of the LPP3 expansion polynomial.

        # C++ parity: ``LPP3HestonExpansion::z2`` in
        # hestonexpansionengine.cpp:597 (v1.43). The body below is a
        # token-for-token transliteration of the C++ expression: ``pow`` and
        # ``fastpow`` map to ``_pow``, ``sqrt`` to ``_sqrt``, and the member
        # scalars ``ekt``/``e2kt``/``e3kt``/``e4kt`` are bound as locals so the
        # arithmetic reads exactly as it does in C++.
        """
        ekt = self._ekt
        e2kt = self._e2kt
        e3kt = self._e3kt
        return (
            (_pow(delta, 2) * (8 * e3kt * _pow(kappa, 5) * _pow(rho, 2) * _pow(t, 4) * (2 +delta *
            rho * t) * _pow(theta, 2) * (theta -y) -delta * _pow(-1 +ekt, 3) * rho * (2 * (-1 +ekt *
            (-5 +24 * _pow(rho, 2))) * _pow(theta, 3) +(7 +ekt * (3 +56 * _pow(rho, 2))) *
            _pow(theta, 2) * y -3 * (1 +ekt * (-3 +8 * _pow(rho, 2))) * theta * _pow(y, 2) +2 * (-1
            +ekt * (-1 +2 * _pow(rho, 2))) * _pow(y, 3)) -_pow(-1 +ekt, 2) * kappa * ((-4 +delta *
            rho * t -8 * ekt * (2 -12 * _pow(rho, 2) -4 * delta * rho * t +25 * delta * _pow(rho, 3)
            * t) +e2kt * (20 -96 * _pow(rho, 2) +3 * delta * rho * t +56 * delta * _pow(rho, 3) *
            t)) * _pow(theta, 3) -2 * (-8 +2 * delta * rho * t +e2kt * (24 -80 * _pow(rho, 2) -9 *
            delta * rho * t +24 * delta * _pow(rho, 3) * t) -4 * ekt * (4 -20 * _pow(rho, 2) -10 *
            delta * rho * t +39 * delta * _pow(rho, 3) * t)) * _pow(theta, 2) * y +(5 * (-4 +delta *
            rho * t) +ekt * (-16 +80 * _pow(rho, 2) +57 * delta * rho * t -140 * delta * _pow(rho,
            3) * t) +2 * e2kt * (18 -40 * _pow(rho, 2) -3 * delta * rho * t +6 * delta * _pow(rho,
            3) * t)) * theta * _pow(y, 2) +2 * (4 +e2kt * (-4 +8 * _pow(rho, 2)) -delta * rho * t
            +ekt * rho * (-8 * rho -7 * delta * t +14 * delta * _pow(rho, 2) * t)) * _pow(y, 3))
            +ekt * (-1 +ekt) * _pow(kappa, 2) * t * ((-24 +128 * _pow(rho, 2) +9 * delta * rho * t
            -144 * delta * _pow(rho, 3) * t -4 * ekt * (6 -8 * _pow(rho, 2) -9 * delta * rho * t +6
            * delta * _pow(rho, 3) * t) +e2kt * (48 -160 * _pow(rho, 2) -9 * delta * rho * t +24 *
            delta * _pow(rho, 3) * t)) * _pow(theta, 3) -(-72 +320 * _pow(rho, 2) +27 * delta * rho
            * t -360 * delta * _pow(rho, 3) * t -ekt * rho * (160 * rho -81 * delta * t +348 * delta
            * _pow(rho, 2) * t) +2 * e2kt * (36 -80 * _pow(rho, 2) -3 * delta * rho * t +6 * delta *
            _pow(rho, 3) * t)) * _pow(theta, 2) * y -2 * (32 -128 * _pow(rho, 2) +12 * e2kt * (-1 +2
            * _pow(rho, 2)) -15 * delta * rho * t +144 * delta * _pow(rho, 3) * t +2 * ekt * (-10
            +52 * _pow(rho, 2) -13 * delta * rho * t +58 * delta * _pow(rho, 3) * t)) * theta *
            _pow(y, 2) +4 * (4 -16 * _pow(rho, 2) -3 * delta * rho * t +18 * delta * _pow(rho, 3) *
            t +ekt * (-4 +16 * _pow(rho, 2) -2 * delta * rho * t +11 * delta * _pow(rho, 3) * t)) *
            _pow(y, 3)) -4 * e2kt * _pow(kappa, 4) * _pow(t, 3) * theta * (2 * e2kt * (-1 +2 *
            _pow(rho, 2)) * _pow(theta, 2) +_pow(rho, 2) * (4 +13 * delta * rho * t) * _pow(theta
            -y, 2) +ekt * ((-4 +16 * _pow(rho, 2) -2 * delta * rho * t +9 * delta * _pow(rho, 3) *
            t) * _pow(theta, 2) +(4 -32 * _pow(rho, 2) +2 * delta * rho * t -19 * delta * _pow(rho,
            3) * t) * theta * y +4 * _pow(rho, 2) * (2 +delta * rho * t) * _pow(y, 2))) -2 * ekt *
            _pow(kappa, 3) * _pow(t, 2) * (-4 * _pow(rho, 2) * (-4 +3 * delta * rho * t) *
            _pow(theta -y, 3) +e3kt * _pow(theta, 2) * ((18 -40 * _pow(rho, 2) -delta * rho * t +2 *
            delta * _pow(rho, 3) * t) * theta +12 * (-1 +2 * _pow(rho, 2)) * y) +2 * ekt * ((-9 +36
            * _pow(rho, 2) +19 * delta * _pow(rho, 3) * t) * _pow(theta, 3) +2 * (9 -30 * _pow(rho,
            2) +7 * delta * _pow(rho, 3) * t) * _pow(theta, 2) * y +(-8 +20 * _pow(rho, 2) +delta *
            rho * t -46 * delta * _pow(rho, 3) * t) * theta * _pow(y, 2) +_pow(rho, 2) * (4 +13 *
            delta * rho * t) * _pow(y, 3)) +e2kt * (8 * theta * y * (-3 * theta +2 * y) +delta * rho
            * t * theta * (7 * _pow(theta, 2) -23 * theta * y +8 * _pow(y, 2)) -8 * _pow(rho, 2) *
            (6 * _pow(theta, 3) -18 * _pow(theta, 2) * y +11 * theta * _pow(y, 2) -_pow(y, 3)) +4 *
            delta * _pow(rho, 3) * t * (-13 * _pow(theta, 3) +31 * _pow(theta, 2) * y -14 * theta *
            _pow(y, 2) +_pow(y, 3)))))) / (64. * _pow(kappa, 2) * t * _sqrt((-theta +kappa * t *
            theta +(theta -y) / ekt +y) / (kappa * t)) * _pow((1 +ekt * (-1 +kappa * t)) * theta
            +(-1 +ekt) * y, 4))
        )

    def _z3(
        self,
        t: float,
        kappa: float,
        theta: float,
        delta: float,
        y: float,
        rho: float,
    ) -> float:
        """Coefficient of x^3 of the LPP3 expansion polynomial.

        # C++ parity: ``LPP3HestonExpansion::z3`` in
        # hestonexpansionengine.cpp:660 (v1.43). The body below is a
        # token-for-token transliteration of the C++ expression: ``pow`` and
        # ``fastpow`` map to ``_pow``, ``sqrt`` to ``_sqrt``, and the member
        # scalars ``ekt``/``e2kt``/``e3kt``/``e4kt`` are bound as locals so the
        # arithmetic reads exactly as it does in C++.
        """
        ekt = self._ekt
        e2kt = self._e2kt
        e3kt = self._e3kt
        e4kt = self._e4kt
        return (
            (_pow(delta, 3) * ekt * rho * ((-15 * (2 +kappa * t) +3 * e4kt * (50 -79 * kappa * t +35
            * _pow(kappa, 2) * _pow(t, 2) -6 * _pow(kappa, 3) * _pow(t, 3) +8 * _pow(rho, 2) * (-18
            +15 * kappa * t -6 * _pow(kappa, 2) * _pow(t, 2) +_pow(kappa, 3) * _pow(t, 3))) +ekt *
            (-3 * (20 +86 * kappa * t +29 * _pow(kappa, 2) * _pow(t, 2)) +_pow(rho, 2) * (432 +936 *
            kappa * t +552 * _pow(kappa, 2) * _pow(t, 2) +92 * _pow(kappa, 3) * _pow(t, 3))) +e2kt *
            (360 +324 * kappa * t -261 * _pow(kappa, 2) * _pow(t, 2) -48 * _pow(kappa, 3) * _pow(t,
            3) -4 * _pow(rho, 2) * (324 +378 * kappa * t -12 * _pow(kappa, 2) * _pow(t, 2) -2 *
            _pow(kappa, 3) * _pow(t, 3) +23 * _pow(kappa, 4) * _pow(t, 4))) +e3kt * (3 * (-140 +62 *
            kappa * t +81 * _pow(kappa, 2) * _pow(t, 2) -38 * _pow(kappa, 3) * _pow(t, 3) +8 *
            _pow(kappa, 4) * _pow(t, 4)) +4 * _pow(rho, 2) * (324 +54 * kappa * t -114 * _pow(kappa,
            2) * _pow(t, 2) +77 * _pow(kappa, 3) * _pow(t, 3) -19 * _pow(kappa, 4) * _pow(t, 4) +2 *
            _pow(kappa, 5) * _pow(t, 5)))) * _pow(theta, 3) +(15 * (7 +4 * kappa * t) +3 * e4kt *
            (-79 +70 * kappa * t -18 * _pow(kappa, 2) * _pow(t, 2) +24 * _pow(rho, 2) * (5 -4 *
            kappa * t +_pow(kappa, 2) * _pow(t, 2))) -3 * ekt * (26 -200 * kappa * t -87 *
            _pow(kappa, 2) * _pow(t, 2) +4 * _pow(rho, 2) * (30 +142 * kappa * t +115 * _pow(kappa,
            2) * _pow(t, 2) +23 * _pow(kappa, 3) * _pow(t, 3))) +2 * e2kt * (3 * (-66 -195 * kappa *
            t +63 * _pow(kappa, 2) * _pow(t, 2) +16 * _pow(kappa, 3) * _pow(t, 3)) +4 * _pow(rho, 2)
            * (135 +390 * kappa * t -9 * _pow(kappa, 2) * _pow(t, 2) -48 * _pow(kappa, 3) * _pow(t,
            3) +23 * _pow(kappa, 4) * _pow(t, 4))) +e3kt * (606 +300 * kappa * t -585 * _pow(kappa,
            2) * _pow(t, 2) +210 * _pow(kappa, 3) * _pow(t, 3) -24 * _pow(kappa, 4) * _pow(t, 4) -4
            * _pow(rho, 2) * (270 +282 * kappa * t -345 * _pow(kappa, 2) * _pow(t, 2) +153 *
            _pow(kappa, 3) * _pow(t, 3) -29 * _pow(kappa, 4) * _pow(t, 4) +2 * _pow(kappa, 5) *
            _pow(t, 5)))) * _pow(theta, 2) * y +(-93 -75 * kappa * t +3 * e4kt * (35 -18 * kappa * t
            +24 * _pow(rho, 2) * (-2 +kappa * t)) +3 * ekt * (58 -123 * kappa * t -86 * _pow(kappa,
            2) * _pow(t, 2) +4 * _pow(rho, 2) * (12 +80 * kappa * t +92 * _pow(kappa, 2) * _pow(t,
            2) +23 * _pow(kappa, 3) * _pow(t, 3))) +e3kt * (-3 * (74 +137 * kappa * t -100 *
            _pow(kappa, 2) * _pow(t, 2) +16 * _pow(kappa, 3) * _pow(t, 3)) -16 * _pow(rho, 2) * (-27
            -51 * kappa * t +45 * _pow(kappa, 2) * _pow(t, 2) -12 * _pow(kappa, 3) * _pow(t, 3)
            +_pow(kappa, 4) * _pow(t, 4))) +e2kt * (36 +909 * kappa * t -42 * _pow(kappa, 2) *
            _pow(t, 2) -60 * _pow(kappa, 3) * _pow(t, 3) -4 * _pow(rho, 2) * (108 +462 * kappa * t
            +96 * _pow(kappa, 2) * _pow(t, 2) -117 * _pow(kappa, 3) * _pow(t, 3) +23 * _pow(kappa,
            4) * _pow(t, 4)))) * theta * _pow(y, 2) +2 * (9 +3 * e4kt * (-3 +4 * _pow(rho, 2)) +15 *
            kappa * t +e2kt * (-3 * kappa * t * (33 +10 * kappa * t) +_pow(rho, 2) * (36 +192 *
            kappa * t +96 * _pow(kappa, 2) * _pow(t, 2) -46 * _pow(kappa, 3) * _pow(t, 3))) +e3kt *
            (18 +57 * kappa * t -12 * _pow(kappa, 2) * _pow(t, 2) -2 * _pow(rho, 2) * (18 +48 *
            kappa * t -21 * _pow(kappa, 2) * _pow(t, 2) +2 * _pow(kappa, 3) * _pow(t, 3))) +ekt * (3
            * (-6 +9 * kappa * t +14 * _pow(kappa, 2) * _pow(t, 2)) -2 * _pow(rho, 2) * (6 +48 *
            kappa * t +69 * _pow(kappa, 2) * _pow(t, 2) +23 * _pow(kappa, 3) * _pow(t, 3)))) *
            _pow(y, 3))) / (96. * kappa * t * _sqrt((-theta +kappa * t * theta +(theta -y) / ekt +y)
            / (kappa * t)) * _pow((1 +ekt * (-1 +kappa * t)) * theta +(-1 +ekt) * y, 5))
        )


class FordeHestonExpansion(HestonExpansion):
    """Forde-Jacquier-Lee small-time expansion.

    # C++ parity: ``class FordeHestonExpansion : public HestonExpansion`` in
    # hestonexpansionengine.hpp:124-132 and .cpp:198-222 (v1.43).

    Unlike LPP2/LPP3 the quartic polynomial approximates the implied
    *variance*; ``impliedVolatility`` floors that at ``1e-8`` and then takes the
    square root, so the smallest volatility it can return is ``1e-4``.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        v0: float,
        rho: float,
        term: float,
    ) -> None:
        """Precompute the five variance-polynomial coefficients for one expiry.

        # C++ parity: ``FordeHestonExpansion::FordeHestonExpansion``
        # (.cpp:198-214), transliterated line for line.
        """
        v0_sqrt = _sqrt(v0)
        rho_bar_square = 1 - rho * rho
        sigma00 = v0_sqrt
        # term in x
        sigma01 = v0_sqrt * (rho * sigma / (4 * v0))
        # term in x*x
        sigma02 = v0_sqrt * ((1 - 5 * rho * rho / 2) / 24 * sigma * sigma / (v0 * v0))
        a00 = (
            -sigma * sigma / 12 * (1 - rho * rho / 4)
            + v0 * rho * sigma / 4
            + kappa / 2 * (theta - v0)
        )
        # term in x
        a01 = (
            rho
            * sigma
            / (24 * v0)
            * (sigma * sigma * rho_bar_square - 2 * kappa * (theta + v0) + v0 * rho * sigma)
        )
        a02 = (
            (
                176 * sigma * sigma
                - 480 * kappa * theta
                - 712 * rho * rho * sigma * sigma
                + 521 * rho * rho * rho * rho * sigma * sigma
                + 40 * sigma * rho * rho * rho * v0
                + 1040 * kappa * theta * rho * rho
                - 80 * v0 * kappa * rho * rho
            )
            * sigma
            * sigma
            / (v0 * v0 * 7680)
        )
        self._coeffs: tuple[float, float, float, float, float] = (
            sigma00 * sigma00 + a00 * term,
            sigma00 * sigma01 * 2 + a01 * term,
            sigma00 * sigma02 * 2 + sigma01 * sigma01 + a02 * term,
            sigma01 * sigma02 * 2,
            sigma02 * sigma02,
        )

    def implied_volatility(self, strike: float, forward: float) -> float:
        """# C++ parity: ``FordeHestonExpansion::impliedVolatility`` (.cpp:216-222)."""
        x = math.log(strike / forward)
        c = self._coeffs
        var = c[0] + x * (c[1] + (x * (c[2] + x * (c[3] + (x * c[4])))))
        var = max(1e-8, var)
        return _sqrt(var)


class HestonExpansionFormula(IntEnum):
    """Which expansion the engine uses.

    # C++ parity: ``enum HestonExpansionFormula { LPP2, LPP3, Forde }`` nested
    # in ``HestonExpansionEngine`` (hestonexpansionengine.hpp:53). Exposed here
    # both at module level and as ``HestonExpansionEngine.LPP2`` etc., so Python
    # call sites read like the C++ ``HestonExpansionEngine::LPP2``.
    """

    LPP2 = 0
    LPP3 = 1
    Forde = 2


class HestonExpansionEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """European-vanilla Heston engine built on an analytic expansion.

    # C++ parity: ``class HestonExpansionEngine`` in
    # hestonexpansionengine.hpp:39-66 and .cpp:39-104 (v1.43).

    Fills ``results.value`` only — no Greeks, matching C++.
    """

    LPP2 = HestonExpansionFormula.LPP2
    LPP3 = HestonExpansionFormula.LPP3
    Forde = HestonExpansionFormula.Forde

    def __init__(self, model: HestonModel, formula: HestonExpansionFormula) -> None:
        """Construct from a Heston model and the expansion to use.

        # C++ parity: ``HestonExpansionEngine::HestonExpansionEngine``
        # (.cpp:39-46) — stores the formula and registers with the model so a
        # recalibration invalidates the cached price.
        """
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._model: HestonModel = model
        self._formula: HestonExpansionFormula = formula
        model.register_with(self)

    def model(self) -> HestonModel:
        """The underlying Heston model."""
        return self._model

    def formula(self) -> HestonExpansionFormula:
        """Which expansion this engine was constructed with."""
        return self._formula

    def calculate(self) -> None:
        """Price a European vanilla via the selected expansion.

        # C++ parity: ``HestonExpansionEngine::calculate`` (.cpp:48-104).
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European option",
        )
        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff),
            "non plain vanilla payoff given",
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        model = self._model
        process = model.process()
        last_date = args.exercise.last_date()

        risk_free_discount = process.risk_free_rate().discount(last_date)
        dividend_discount = process.dividend_yield().discount(last_date)

        spot_price = process.s0().value()
        qassert.require(spot_price > 0.0, "negative or null underlying given")

        strike_price = payoff.strike()
        term = process.time(last_date)

        forward = spot_price * dividend_discount / risk_free_discount

        expansion: HestonExpansion
        if self._formula == HestonExpansionFormula.LPP2:
            expansion = LPP2HestonExpansion(
                model.kappa(), model.theta(), model.sigma(), model.v0(), model.rho(), term
            )
        elif self._formula == HestonExpansionFormula.LPP3:
            expansion = LPP3HestonExpansion(
                model.kappa(), model.theta(), model.sigma(), model.v0(), model.rho(), term
            )
        elif self._formula == HestonExpansionFormula.Forde:
            expansion = FordeHestonExpansion(
                model.kappa(), model.theta(), model.sigma(), model.v0(), model.rho(), term
            )
        else:
            raise LibraryException(f"unknown expansion formula: {self._formula}")

        vol = expansion.implied_volatility(strike_price, forward)

        # C++ parity: .cpp:101-103 — blackFormula(payoff, forward,
        # vol*sqrt(term), riskFreeDiscount, 0). The discount is the RISK-FREE
        # one; the dividend discount is already inside `forward`.
        results.reset()
        results.value = black_formula(
            payoff.option_type(),
            strike_price,
            forward,
            vol * _sqrt(term),
            risk_free_discount,
            0.0,
        )

    def update(self) -> None:
        """Observer.update — model parameters or curves changed."""
        self.notify_observers()


__all__ = [
    "FordeHestonExpansion",
    "HestonExpansion",
    "HestonExpansionEngine",
    "HestonExpansionFormula",
    "LPP2HestonExpansion",
    "LPP3HestonExpansion",
]
