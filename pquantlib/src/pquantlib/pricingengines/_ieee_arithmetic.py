"""IEEE-754 arithmetic helpers for faithful C++ ports.

# C++ parity: none — this bridges a language difference, not a QuantLib class.

C++ evaluates ``0.0/0.0`` to NaN, ``x/0.0`` to a signed infinity and
``sqrt(-1)`` to NaN, and lets those values propagate.  Python raises
:class:`ZeroDivisionError` / :class:`ValueError` instead.

Several QuantLib closed forms genuinely reach those expressions on their
degenerate inputs, and what C++ does *there* is part of the behaviour a port
has to reproduce:

* :class:`~pquantlib.pricingengines.american_payoff_at_hit.AmericanPayoffAtHit`
  and :class:`~pquantlib.pricingengines.american_payoff_at_expiry.AmericanPayoffAtExpiry`
  compute ``mu = log(qDF/rDF) / variance``, which is 0/0 at zero variance,
  and C++ returns a NaN price rather than refusing.
* :meth:`~pquantlib.pricingengines.vanilla.barone_adesi_whaley_engine.BaroneAdesiWhaleyApproximationEngine.critical_price`
  divides by ``variance`` in four places.  At zero variance the C++ chain is
  ``n = -inf`` -> ``Si = NaN`` -> ``blackFormula`` rejecting a NaN forward
  with *its own* error message.  Letting Python trap at the first division
  would raise a different exception, from a different place, for a different
  reason.

Using these helpers keeps the arithmetic on the C++ path so the *first*
thing that objects is the same guard clause in both languages.
"""

from __future__ import annotations

import math


def div(numerator: float, denominator: float) -> float:
    """IEEE-754 division: NaN for 0/0, a signed infinity for x/0."""
    if denominator == 0.0:
        if numerator == 0.0 or math.isnan(numerator):
            return math.nan
        return math.copysign(math.inf, numerator) * math.copysign(1.0, denominator)
    return numerator / denominator


def sqrt(x: float) -> float:
    """``std::sqrt``: NaN for a negative argument rather than a ValueError."""
    if x < 0.0:
        return math.nan
    return math.sqrt(x)


def log(x: float) -> float:
    """``std::log``: -inf at zero and NaN for a negative argument."""
    if x == 0.0:
        return -math.inf
    if x < 0.0:
        return math.nan
    return math.log(x)


def power(base: float, exponent: float) -> float:
    """``std::pow``: infinities instead of OverflowError / ValueError."""
    try:
        return math.pow(base, exponent)
    except OverflowError:
        # C++ returns +-inf where Python reports overflow; the sign follows
        # base < 0 with an odd integer exponent, as in IEEE-754 pow.
        if base < 0.0 and exponent == int(exponent) and int(exponent) % 2 != 0:
            return -math.inf
        return math.inf
    except ValueError:
        return math.nan


__all__ = ["div", "log", "power", "sqrt"]
