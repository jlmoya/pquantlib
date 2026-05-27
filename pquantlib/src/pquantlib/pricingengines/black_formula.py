"""Black 1976 formula family + Bachelier (normal) variant.

# C++ parity: ql/pricingengines/blackformula.{hpp,cpp} (v1.42.1).

Module-level free functions (matches C++ — no class wrapper):

* ``black_formula`` — lognormal Black-76 option value.
* ``black_formula_implied_std_dev`` — Newton-safe implied std dev
  given a target price.
* ``black_formula_implied_std_dev_approximation`` — closed-form initial
  guess for the solver above (Brenner-Subrahmanyan / Corrado-Miller).
* ``black_formula_std_dev_derivative`` — derivative wrt std dev
  (Black vega without ``sqrt(T)``).
* ``black_formula_vol_derivative`` — Black vega (derivative wrt vol).
* ``bachelier_black_formula`` — Bachelier (normal model) option value.
* ``bachelier_black_formula_std_dev_derivative`` — Bachelier vega
  without ``sqrt(T)``.
* ``bachelier_black_formula_implied_vol`` — exact Bachelier implied
  volatility (Jaeckel 2017 algorithm).

The ``std_dev`` argument is the *cumulative* standard deviation, i.e.
``volatility * sqrt(time_to_maturity)`` — not the annualized vol.

C++ free functions deferred (not used in vanilla path): the
``shared_ptr<PlainVanillaPayoff>`` overloads (Python callers can
unwrap themselves), Chambers / RS / LiRS approximations,
``cashItmProbability`` / ``assetItmProbability``,
``stdDevSecondDerivative``, ``bachelierBlackFormulaImpliedVolChoi``
(approximation — we ship the exact Jaeckel variant).
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.solvers1d.newton_safe import NewtonSafe
from pquantlib.payoffs import OptionType

_PHI: Final[CumulativeNormalDistribution] = CumulativeNormalDistribution()
_SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)


def _check_parameters(strike: float, forward: float, displacement: float) -> None:
    qassert.require(displacement >= 0.0, f"displacement ({displacement}) must be non-negative")
    qassert.require(
        strike + displacement >= 0.0,
        f"strike + displacement ({strike} + {displacement}) must be non-negative",
    )
    qassert.require(
        forward + displacement > 0.0,
        f"forward + displacement ({forward} + {displacement}) must be positive",
    )


# --- Black 1976 lognormal -----------------------------------------------


def black_formula(
    option_type: OptionType,
    strike: float,
    forward: float,
    std_dev: float,
    discount: float = 1.0,
    displacement: float = 0.0,
) -> float:
    """Black 1976 option value.

    # C++ parity: blackformula.cpp ``blackFormula(Option::Type, ...)``.

    ``std_dev`` is the cumulative standard deviation,
    ``volatility * sqrt(time_to_maturity)``.
    """
    _check_parameters(strike, forward, displacement)
    qassert.require(std_dev >= 0.0, f"stdDev ({std_dev}) must be non-negative")
    qassert.require(discount > 0.0, f"discount ({discount}) must be positive")

    sign = int(option_type)

    if std_dev == 0.0:
        return max((forward - strike) * sign, 0.0) * discount

    forward = forward + displacement
    strike = strike + displacement

    # Since displacement is non-negative, strike==0 iff displacement==0.
    if strike == 0.0:
        return forward * discount if option_type == OptionType.Call else 0.0

    d1 = math.log(forward / strike) / std_dev + 0.5 * std_dev
    d2 = d1 - std_dev
    nd1 = _PHI(sign * d1)
    nd2 = _PHI(sign * d2)
    result = discount * sign * (forward * nd1 - strike * nd2)
    qassert.require(
        result >= 0.0,
        f"negative value ({result}) for {std_dev} stdDev, {option_type} option, "
        f"{strike} strike, {forward} forward",
    )
    return result


def black_formula_std_dev_derivative(
    strike: float,
    forward: float,
    std_dev: float,
    discount: float = 1.0,
    displacement: float = 0.0,
) -> float:
    """Black derivative wrt ``std_dev`` (Black vega without ``sqrt(T)``).

    # C++ parity: ``blackFormulaStdDevDerivative``.
    """
    _check_parameters(strike, forward, displacement)
    qassert.require(std_dev >= 0.0, f"stdDev ({std_dev}) must be non-negative")
    qassert.require(discount > 0.0, f"discount ({discount}) must be positive")

    forward = forward + displacement
    strike = strike + displacement

    if std_dev == 0.0 or strike == 0.0:
        return 0.0

    d1 = math.log(forward / strike) / std_dev + 0.5 * std_dev
    return discount * forward * _PHI.derivative(d1)


def black_formula_vol_derivative(
    strike: float,
    forward: float,
    std_dev: float,
    ttm: float,
    discount: float = 1.0,
    displacement: float = 0.0,
) -> float:
    """Black derivative wrt volatility (Black vega).

    # C++ parity: ``blackFormulaVolDerivative`` —
    # ``stdDevDerivative * sqrt(ttm)``.
    """
    return black_formula_std_dev_derivative(
        strike, forward, std_dev, discount, displacement
    ) * math.sqrt(ttm)


def black_formula_implied_std_dev_approximation(
    option_type: OptionType,
    strike: float,
    forward: float,
    black_price: float,
    discount: float = 1.0,
    displacement: float = 0.0,
) -> float:
    """Brenner-Subrahmanyan / Corrado-Miller approximate implied std dev.

    # C++ parity: ``blackFormulaImpliedStdDevApproximation``.

    Provides a robust starting point for ``black_formula_implied_std_dev``.
    """
    _check_parameters(strike, forward, displacement)
    qassert.require(black_price >= 0.0, f"blackPrice ({black_price}) must be non-negative")
    qassert.require(discount > 0.0, f"discount ({discount}) must be positive")

    f = forward + displacement
    k = strike + displacement

    if k == f:
        # Brenner-Subrahmanyan / Feinstein ATM approximation.
        std_dev = black_price / discount * _SQRT_2PI / f
    else:
        moneyness_delta = int(option_type) * (f - k)
        moneyness_delta_2 = moneyness_delta / 2.0
        temp = black_price / discount - moneyness_delta_2
        moneyness_delta_pi = moneyness_delta * moneyness_delta / math.pi
        temp2 = temp * temp - moneyness_delta_pi
        temp2 = max(temp2, 0.0)
        temp2 = math.sqrt(temp2)
        temp += temp2
        temp *= _SQRT_2PI
        std_dev = temp / (f + k)

    qassert.require(std_dev >= 0.0, f"stdDev ({std_dev}) must be non-negative")
    return std_dev


class _BlackImpliedStdDevHelper:
    """Functor for the implied-std-dev Newton-safe solver.

    # C++ parity: anonymous-namespace ``BlackImpliedStdDevHelper`` in
    # ``blackformula.cpp``.
    """

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        undiscounted_black_price: float,
        displacement: float = 0.0,
    ) -> None:
        _check_parameters(strike, forward, displacement)
        qassert.require(
            undiscounted_black_price >= 0.0,
            f"undiscounted Black price ({undiscounted_black_price}) must be non-negative",
        )
        sign = int(option_type)
        self._half_option_type: float = 0.5 * sign
        self._signed_strike: float = sign * (strike + displacement)
        self._signed_forward: float = sign * (forward + displacement)
        self._undiscounted_black_price: float = undiscounted_black_price
        self._signed_moneyness: float = sign * math.log(
            (forward + displacement) / (strike + displacement)
        )

    def __call__(self, std_dev: float) -> float:
        if std_dev == 0.0:
            return (
                max(self._signed_forward - self._signed_strike, 0.0)
                - self._undiscounted_black_price
            )
        temp = self._half_option_type * std_dev
        d = self._signed_moneyness / std_dev
        signed_d1 = d + temp
        signed_d2 = d - temp
        result = self._signed_forward * _PHI(signed_d1) - self._signed_strike * _PHI(signed_d2)
        # Numerical inaccuracies can yield a slightly negative answer.
        return max(0.0, result) - self._undiscounted_black_price

    def derivative(self, std_dev: float) -> float:
        signed_d1 = self._signed_moneyness / std_dev + self._half_option_type * std_dev
        return self._signed_forward * _PHI.derivative(signed_d1)


def black_formula_implied_std_dev(
    option_type: OptionType,
    strike: float,
    forward: float,
    black_price: float,
    discount: float = 1.0,
    displacement: float = 0.0,
    guess: float | None = None,
    accuracy: float = 1.0e-6,
    max_iterations: int = 100,
) -> float:
    """Newton-safe implied standard deviation.

    # C++ parity: ``blackFormulaImpliedStdDev``.

    Solves ``black_formula(option_type, strike, forward, x, discount,
    displacement) == black_price`` for ``x``. Uses Newton-safe with
    the Brenner-Subrahmanyan / Corrado-Miller approximation as the
    initial guess.
    """
    _check_parameters(strike, forward, displacement)
    qassert.require(discount > 0.0, f"discount ({discount}) must be positive")
    qassert.require(black_price >= 0.0, f"option price ({black_price}) must be non-negative")

    # Check the price of the "other" option implied by put-call parity.
    other_option_price = black_price - int(option_type) * (forward - strike) * discount
    qassert.require(
        other_option_price >= 0.0,
        f"negative {OptionType(-1 * int(option_type))} price ({other_option_price}) "
        f"implied by put-call parity. No solution exists for "
        f"{option_type} strike {strike}, forward {forward}, price {black_price}, "
        f"deflator {discount}",
    )

    # Solve for the OTM option (greater vega/price ratio, more robust).
    if option_type == OptionType.Put and strike > forward:
        option_type = OptionType.Call
        black_price = other_option_price
    if option_type == OptionType.Call and strike < forward:
        option_type = OptionType.Put
        black_price = other_option_price

    strike_d = strike + displacement
    forward_d = forward + displacement

    if guess is None:
        guess = black_formula_implied_std_dev_approximation(
            option_type, strike_d, forward_d, black_price, discount, displacement=0.0
        )
    else:
        qassert.require(guess >= 0.0, f"stdDev guess ({guess}) must be non-negative")

    # Note: the C++ helper was constructed with displacement = 0
    # because strike/forward have already been displaced. The
    # solver bracket is [0, 24] = 300% * sqrt(60).
    f = _BlackImpliedStdDevHelper(
        option_type, strike_d, forward_d, black_price / discount, displacement=0.0
    )
    solver = NewtonSafe()
    solver.set_max_evaluations(max_iterations)
    std_dev = solver.solve(f, accuracy, guess, 0.0, 24.0)
    qassert.require(std_dev >= 0.0, f"stdDev ({std_dev}) must be non-negative")
    return std_dev


# --- Bachelier (normal) variant -----------------------------------------


def bachelier_black_formula(
    option_type: OptionType,
    strike: float,
    forward: float,
    std_dev: float,
    discount: float = 1.0,
) -> float:
    """Bachelier (normal) option value.

    # C++ parity: ``bachelierBlackFormula``.

    Bachelier model uses *absolute* volatility; ``std_dev`` is
    ``absolute_vol * sqrt(time_to_maturity)``.
    """
    qassert.require(std_dev >= 0.0, f"stdDev ({std_dev}) must be non-negative")
    qassert.require(discount > 0.0, f"discount ({discount}) must be positive")

    sign = int(option_type)
    d = (forward - strike) * sign
    if std_dev == 0.0:
        return discount * max(d, 0.0)
    h = d / std_dev
    result = discount * (std_dev * _PHI.derivative(h) + d * _PHI(h))
    qassert.require(
        result >= 0.0,
        f"negative value ({result}) for {std_dev} stdDev, {option_type} option, "
        f"{strike} strike, {forward} forward",
    )
    return result


def bachelier_black_formula_std_dev_derivative(
    strike: float,
    forward: float,
    std_dev: float,
    discount: float = 1.0,
) -> float:
    """Bachelier derivative wrt ``std_dev``.

    # C++ parity: ``bachelierBlackFormulaStdDevDerivative``.
    """
    qassert.require(std_dev >= 0.0, f"stdDev ({std_dev}) must be non-negative")
    qassert.require(discount > 0.0, f"discount ({discount}) must be positive")

    if std_dev == 0.0:
        return 0.0
    d1 = (forward - strike) / std_dev
    return discount * _PHI.derivative(d1)


# --- Bachelier implied vol (exact, Jaeckel 2017) ------------------------

# Module-level normal pdf / cdf / PhiTilde helpers — mirrors
# ``blackformula.cpp`` anonymous namespace.


def _phi(x: float) -> float:
    return _PHI.derivative(x)


def _phi_cap(x: float) -> float:
    """Cumulative normal CDF Phi(x). C++ parity: ``Phi`` in
    ``blackformula.cpp`` anonymous namespace."""
    return _PHI(x)


def _phi_tilde(x: float) -> float:
    """``PhiTilde(x) = Phi(x) + phi(x)/x``. C++ parity: ``PhiTilde`` in
    ``blackformula.cpp`` anonymous namespace.
    """
    return _phi_cap(x) + _phi(x) / x


def _inverse_phi_tilde(phi_tilde_star: float) -> float:
    """Jaeckel inverse of PhiTilde at a negative argument.

    # C++ parity: ``inversePhiTilde`` (anonymous-namespace) in
    # ``blackformula.cpp``.
    """
    qassert.require(
        phi_tilde_star < 0.0,
        f"inversePhiTilde({phi_tilde_star}): negative argument required",
    )
    if phi_tilde_star < -0.001882039271:
        g = 1.0 / (phi_tilde_star - 0.5)
        xibar = (
            0.032114372355
            - g * g * (0.016969777977 - g * g * (2.6207332461e-3 - 9.6066952861e-5 * g * g))
        ) / (
            1.0
            - g * g * (0.6635646938 - g * g * (0.14528712196 - 0.010472855461 * g * g))
        )
        xbar = g * (0.3989422804014326 + xibar * g * g)
    else:
        h = math.sqrt(-math.log(-phi_tilde_star))
        xbar = (
            9.4883409779 - h * (9.6320903635 - h * (0.58556997323 + 2.1464093351 * h))
        ) / (1.0 - h * (0.65174820867 + h * (1.5120247828 + 6.6437847132e-5 * h)))
    q = (_phi_tilde(xbar) - phi_tilde_star) / _phi(xbar)
    xstar = xbar + 3.0 * q * xbar * xbar * (2.0 - q * xbar * (2.0 + xbar * xbar)) / (
        6.0
        + q
        * xbar
        * (-12.0 + xbar * (6.0 * q + xbar * (-6.0 + q * xbar * (3.0 + xbar * xbar))))
    )
    return xstar


def bachelier_black_formula_implied_vol(
    option_type: OptionType,
    strike: float,
    forward: float,
    ttm: float,
    bachelier_price: float,
    discount: float = 1.0,
) -> float:
    """Bachelier exact implied (annualized) volatility (Jaeckel 2017).

    # C++ parity: ``bachelierBlackFormulaImpliedVol``.

    Returns the *annualized* volatility (not the cumulative std dev).
    """
    theta = 1.0 if option_type == OptionType.Call else -1.0
    bachelier_price = bachelier_price / discount

    # Strike == forward edge: closed-form.
    if math.isclose(strike, forward, abs_tol=QL_EPSILON, rel_tol=QL_EPSILON):
        return bachelier_price / (math.sqrt(ttm) * _phi(0.0))

    time_value = bachelier_price - max(theta * (forward - strike), 0.0)
    if math.isclose(time_value, 0.0, abs_tol=QL_EPSILON, rel_tol=QL_EPSILON):
        return 0.0
    qassert.require(
        time_value > 0.0,
        f"bachelierBlackFormulaImpliedVol(theta={theta},strike={strike},"
        f"forward={forward},tte={ttm},price={bachelier_price}): "
        f"option price implies negative time value ({time_value})",
    )

    phi_tilde_star = -abs(time_value / (strike - forward))
    xstar = _inverse_phi_tilde(phi_tilde_star)
    return abs((strike - forward) / (xstar * math.sqrt(ttm)))


__all__ = [
    "bachelier_black_formula",
    "bachelier_black_formula_implied_vol",
    "bachelier_black_formula_std_dev_derivative",
    "black_formula",
    "black_formula_implied_std_dev",
    "black_formula_implied_std_dev_approximation",
    "black_formula_std_dev_derivative",
    "black_formula_vol_derivative",
]
