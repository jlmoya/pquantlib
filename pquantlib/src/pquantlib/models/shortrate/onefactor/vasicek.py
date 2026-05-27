"""Vasicek 1977 single-factor short-rate model.

# C++ parity: ql/models/shortrate/onefactormodels/vasicek.{hpp,cpp} (v1.42.1).

Implements the SDE

    dr_t = a (b - r_t) dt + sigma dW_t

with constants ``a`` (mean-reversion speed), ``b`` (long-run mean),
``sigma`` (volatility). A risk premium ``lambda`` can also be specified
(it shifts the affine ``A(t, T)`` term by ``lambda*sigma/a``).

Discount-bond closed form:

    P(t, T, r) = A(t, T) * exp(-B(t, T) * r)

with

    B(t, T) = (1 - exp(-a*(T-t))) / a
    A(t, T) = exp((b + lambda*sigma/a - 0.5*sigma^2/a^2) * (B(t,T) - (T-t))
                  - 0.25 * sigma^2 * B(t,T)^2 / a)

Discount-bond European option uses ``blackFormula`` with the bond
volatility ``v = sigma * B(maturity, bond_maturity) * sqrt(0.5*(1 -
exp(-2*a*maturity))/a)`` (small-a limit: ``v = sigma * B * sqrt(maturity)``).

# C++ parity for A(t,T): vasicek.cpp:36-47 — the ``_a < sqrt(QL_EPSILON)``
# branch returns A=0.0, which yields a degenerate discount_bond=0.0.
# We preserve this for C++ parity (the limit is documented but rarely
# useful for callers).
"""

from __future__ import annotations

import math

from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.optimization.constraint import NoConstraint, PositiveConstraint
from pquantlib.models.parameter import ConstantParameter, Parameter
from pquantlib.models.shortrate.onefactor.one_factor_affine_model import (
    OneFactorAffineModel,
)
from pquantlib.models.shortrate.onefactor.one_factor_model import (
    ShortRateDynamics,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess


class _VasicekDynamics(ShortRateDynamics):
    """Vasicek short-rate dynamics: OU process centred at zero, shifted by b.

    # C++ parity: ``class Vasicek::Dynamics`` in vasicek.hpp:78-93.

    The OU process state ``x_t`` follows
    ``dx = -a*x dt + sigma dW`` (mean-zero), and the short rate is
    ``r_t = x_t + b``. Mapping:

    - ``variable(t, r) = r - b``
    - ``short_rate(t, x) = x + b``
    """

    __slots__ = ("_b",)

    def __init__(self, a: float, b: float, sigma: float, r0: float) -> None:
        # C++ parity: vasicek.hpp:83-86 — process is OU(a, sigma, r0-b, 0).
        # In our OU naming, that's speed=a, vol=sigma, x0=r0-b, level=0.
        process = OrnsteinUhlenbeckProcess(speed=a, vol=sigma, x0=r0 - b, level=0.0)
        super().__init__(process)
        self._b: float = b

    def variable(self, t: float, r: float) -> float:
        # C++ parity: vasicek.hpp:88 — ``variable(t, r) = r - b_``.
        return r - self._b

    def short_rate(self, t: float, variable: float) -> float:
        # C++ parity: vasicek.hpp:89 — ``shortRate(t, x) = x + b_``.
        return variable + self._b


class Vasicek(OneFactorAffineModel):
    """Vasicek (1977) short-rate model.

    # C++ parity: ``class Vasicek : public OneFactorAffineModel`` in
    # vasicek.hpp:42-72 + vasicek.cpp:26-75 (v1.42.1).

    Four free parameters: ``a`` (positive), ``b``, ``sigma`` (positive),
    ``lambda``. Constructor takes the initial short rate ``r0`` (which
    is held verbatim, not as a Parameter).
    """

    __slots__ = ("_a_param", "_b_param", "_lambda_param", "_r0", "_sigma_param")

    def __init__(
        self,
        r0: float = 0.05,
        a: float = 0.1,
        b: float = 0.05,
        sigma: float = 0.01,
        lambda_: float = 0.0,
    ) -> None:
        """Construct the Vasicek model.

        # C++ parity: vasicek.cpp:26-34 — four arguments via
        # OneFactorAffineModel(4) ctor.

        ``lambda_`` is named with trailing underscore in Python to avoid
        clashing with the ``lambda`` keyword.
        """
        # 4 Parameter slots: a, b, sigma, lambda (positions 0..3).
        super().__init__(n_arguments=4)
        # Cache r0 directly (it's not a Parameter in C++).
        self._r0: float = float(r0)
        # C++ parity: assign each Parameter slot in arguments_[i].
        # Hold typed references for fast access via @property.
        self.arguments[0] = ConstantParameter(a, PositiveConstraint())
        self.arguments[1] = ConstantParameter(b, NoConstraint())
        self.arguments[2] = ConstantParameter(sigma, PositiveConstraint())
        self.arguments[3] = ConstantParameter(lambda_, NoConstraint())
        # Cache aliases for ergonomic access (mirrors the C++ private
        # ``Parameter& a_`` references). Useful for HullWhite subclass.
        self._a_param: Parameter = self.arguments[0]
        self._b_param: Parameter = self.arguments[1]
        self._sigma_param: Parameter = self.arguments[2]
        self._lambda_param: Parameter = self.arguments[3]

    # --- accessors ------------------------------------------------------

    def r0(self) -> float:
        """Initial short rate (constant; not optimised).

        # C++ parity: ``Vasicek::r0`` at vasicek.hpp:58.
        """
        return self._r0

    def a(self) -> float:
        """Mean-reversion speed.

        # C++ parity: ``Vasicek::a`` — ``a_(0.0)`` (evaluates the
        # ConstantParameter at any time).
        """
        return self._a_param(0.0)

    def b(self) -> float:
        """Long-run mean of the short rate.

        # C++ parity: ``Vasicek::b`` at vasicek.hpp:55.
        """
        return self._b_param(0.0)

    def sigma(self) -> float:
        """Short-rate volatility.

        # C++ parity: ``Vasicek::sigma`` at vasicek.hpp:57.
        """
        return self._sigma_param(0.0)

    def lambda_(self) -> float:
        """Risk premium (a.k.a. market price of risk).

        # C++ parity: ``Vasicek::lambda`` at vasicek.hpp:56.
        """
        return self._lambda_param(0.0)

    # --- ShortRateModel surface -----------------------------------------

    def dynamics(self) -> ShortRateDynamics:
        # C++ parity: vasicek.hpp:98-102.
        return _VasicekDynamics(self.a(), self.b(), self.sigma(), self._r0)

    # --- OneFactorAffineModel A(t,T) / B(t,T) ---------------------------

    def _a(self, t: float, t_maturity: float) -> float:
        # C++ parity: vasicek.cpp:36-47.
        a = self.a()
        if a < math.sqrt(QL_EPSILON):
            # Degenerate small-a limit; C++ returns 0.0 here, which makes
            # discount_bond=0.0. We preserve this for parity (the limit
            # is rarely useful as a financial model; HullWhite would be
            # used instead for a≈0).
            return 0.0
        sigma = self.sigma()
        sigma2 = sigma * sigma
        bt = self._b(t, t_maturity)
        b = self.b()
        lambda_ = self.lambda_()
        return math.exp(
            (b + lambda_ * sigma / a - 0.5 * sigma2 / (a * a)) * (bt - (t_maturity - t))
            - 0.25 * sigma2 * bt * bt / a
        )

    def _b(self, t: float, t_maturity: float) -> float:
        # C++ parity: vasicek.cpp:49-55.
        a = self.a()
        if a < math.sqrt(QL_EPSILON):
            return t_maturity - t
        return (1.0 - math.exp(-a * (t_maturity - t))) / a

    # --- AffineModel option pricing -------------------------------------

    def discount_bond_option(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """Closed-form European option on a discount bond.

        # C++ parity: vasicek.cpp:57-75.

        Uses the standard Vasicek bond-volatility formula and
        delegates to the Black formula. The small-a algebraic limit
        ``v = sigma * B * sqrt(maturity)`` is preserved.
        """
        a = self.a()
        sigma = self.sigma()
        b = self._b(maturity, bond_maturity)
        if abs(maturity) < QL_EPSILON:
            v = 0.0
        elif a < math.sqrt(QL_EPSILON):
            v = sigma * b * math.sqrt(maturity)
        else:
            v = sigma * b * math.sqrt(0.5 * (1.0 - math.exp(-2.0 * a * maturity)) / a)
        f = self.discount_bond_scalar(0.0, bond_maturity, self._r0)
        k = self.discount_bond_scalar(0.0, maturity, self._r0) * strike
        return black_formula(option_type, k, f, v)
