"""Cox-Ingersoll-Ross (1985) single-factor short-rate model.

# C++ parity: ql/models/shortrate/onefactormodels/coxingersollross.{hpp,cpp} (v1.42.1).

Implements the SDE

    dr_t = k (theta - r_t) dt + sigma sqrt(r_t) dW_t

with mean-reversion speed ``k``, long-run mean ``theta``, volatility
``sigma``, and initial state ``r0`` (which is itself a Parameter — the
``r0_`` cached field is replaced by ``arguments_[3]``).

The Feller condition ``2*k*theta > sigma^2`` ensures the short rate
stays strictly positive almost surely; when enabled (the default), the
sigma parameter is constrained to satisfy it.

Discount-bond closed form:

    B(t, T) = 2 (e^{h(T-t)} - 1) / (2h + (k+h)(e^{h(T-t)} - 1))
    A(t, T) = ( 2h e^{0.5(k+h)(T-t)} / (2h + (k+h)(e^{h(T-t)} - 1)) )^{2k theta / sigma^2}

with ``h = sqrt(k^2 + 2 sigma^2)``.

Discount-bond options use the non-central chi-square decomposition
(Cox-Ingersoll-Ross 1985, eq. 23). Two non-central chi-square CDFs
are evaluated and combined with the discount factors and strike.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.non_central_chi_square_distribution import (
    NonCentralCumulativeChiSquareDistribution,
)
from pquantlib.math.optimization.constraint import Constraint, PositiveConstraint
from pquantlib.models.parameter import ConstantParameter, Parameter
from pquantlib.models.shortrate.onefactor.one_factor_affine_model import (
    OneFactorAffineModel,
)
from pquantlib.models.shortrate.onefactor.one_factor_model import (
    ShortRateDynamics,
)
from pquantlib.payoffs import OptionType
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess


class _CIRFellerVolatilityConstraint(Constraint):
    """Composite constraint: sigma>0 AND sigma^2 < 2*k*theta (Feller).

    # C++ parity: nested ``CoxIngersollRoss::VolatilityConstraint`` in
    # coxingersollross.cpp:26-41 (v1.42.1).
    """

    __slots__ = ("_k", "_theta")

    def __init__(self, k: float, theta: float) -> None:
        self._k: float = float(k)
        self._theta: float = float(theta)

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        # params[0] is sigma; require positive AND below the Feller bound.
        sigma = float(params[0])
        return sigma > 0.0 and sigma * sigma < 2.0 * self._k * self._theta


class _CoxIngersollRossDynamics(ShortRateDynamics):
    """CIR short-rate dynamics: identity mapping between state and short rate.

    # C++ parity: ``class CoxIngersollRoss::Dynamics`` in coxingersollross.hpp:86-98.

    The state is the short rate itself — ``variable(t, r) = r`` and
    ``short_rate(t, y) = y``. The backing process is a CIRProcess with
    the model's (k, sigma, x0, theta) parameters.
    """

    def __init__(self, theta: float, k: float, sigma: float, x0: float) -> None:
        # C++ parity: coxingersollross.hpp:93-94 — process ctor is
        # CoxIngersollRossProcess(k, sigma, x0, theta).
        process = CoxIngersollRossProcess(speed=k, vol=sigma, x0=x0, level=theta)
        super().__init__(process)

    def variable(self, t: float, r: float) -> float:
        # C++ parity: coxingersollross.hpp:96.
        return r

    def short_rate(self, t: float, variable: float) -> float:
        # C++ parity: coxingersollross.hpp:97.
        return variable


class CoxIngersollRoss(OneFactorAffineModel):
    """Cox-Ingersoll-Ross (1985) short-rate model.

    # C++ parity: ``class CoxIngersollRoss : public OneFactorAffineModel``
    # in coxingersollross.hpp:44-79 + .cpp:43-123 (v1.42.1).

    Four free Parameter slots: ``theta`` (positive), ``k`` (positive),
    ``sigma`` (positive, optionally Feller-constrained), and ``r0``
    (positive). Unlike Vasicek, ``r0`` is a Parameter rather than a
    plain float field so the optimizer can calibrate it.
    """

    __slots__ = ("_k_param", "_r0_param", "_sigma_param", "_theta_param")

    def __init__(
        self,
        r0: float = 0.05,
        theta: float = 0.1,
        k: float = 0.1,
        sigma: float = 0.1,
        with_feller_constraint: bool = True,
    ) -> None:
        """Construct the CIR model.

        # C++ parity: coxingersollross.cpp:43-56.

        Parameter slot ordering matches C++: ``theta``, ``k``, ``sigma``,
        ``r0``. ``with_feller_constraint=True`` applies the Feller
        bound to the sigma slot (sigma^2 < 2*k*theta).
        """
        super().__init__(n_arguments=4)
        self.arguments[0] = ConstantParameter(theta, PositiveConstraint())
        self.arguments[1] = ConstantParameter(k, PositiveConstraint())
        if with_feller_constraint:
            self.arguments[2] = ConstantParameter(
                sigma, _CIRFellerVolatilityConstraint(k, theta)
            )
        else:
            self.arguments[2] = ConstantParameter(sigma, PositiveConstraint())
        self.arguments[3] = ConstantParameter(r0, PositiveConstraint())
        self._theta_param: Parameter = self.arguments[0]
        self._k_param: Parameter = self.arguments[1]
        self._sigma_param: Parameter = self.arguments[2]
        self._r0_param: Parameter = self.arguments[3]

    # --- accessors ------------------------------------------------------

    def theta(self) -> float:
        """Long-run mean."""
        # C++ parity: coxingersollross.hpp:67.
        return self._theta_param(0.0)

    def k(self) -> float:
        """Mean-reversion speed."""
        # C++ parity: coxingersollross.hpp:68.
        return self._k_param(0.0)

    def sigma(self) -> float:
        """Volatility."""
        # C++ parity: coxingersollross.hpp:69.
        return self._sigma_param(0.0)

    def x0(self) -> float:
        """Initial state value (== initial short rate)."""
        # C++ parity: coxingersollross.hpp:70.
        return self._r0_param(0.0)

    # --- ShortRateModel surface -----------------------------------------

    def dynamics(self) -> ShortRateDynamics:
        # C++ parity: coxingersollross.cpp:58-62.
        return _CoxIngersollRossDynamics(self.theta(), self.k(), self.sigma(), self.x0())

    # --- OneFactorAffineModel A(t,T) / B(t,T) ---------------------------

    def _a(self, t: float, t_maturity: float) -> float:
        # C++ parity: coxingersollross.cpp:64-72.
        sigma2 = self.sigma() * self.sigma()
        h = math.sqrt(self.k() * self.k() + 2.0 * sigma2)
        numerator = 2.0 * h * math.exp(0.5 * (self.k() + h) * (t_maturity - t))
        denominator = 2.0 * h + (self.k() + h) * (math.exp((t_maturity - t) * h) - 1.0)
        return math.exp(math.log(numerator / denominator) * 2.0 * self.k() * self.theta() / sigma2)

    def _b(self, t: float, t_maturity: float) -> float:
        # C++ parity: coxingersollross.cpp:74-81.
        h = math.sqrt(self.k() * self.k() + 2.0 * self.sigma() * self.sigma())
        temp = math.exp((t_maturity - t) * h) - 1.0
        numerator = 2.0 * temp
        denominator = 2.0 * h + (self.k() + h) * temp
        return numerator / denominator

    # --- option pricing -------------------------------------------------

    def discount_bond_option(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """Closed-form European option on a discount bond.

        # C++ parity: coxingersollross.cpp:83-123.

        Uses the non-central chi-square decomposition (Cox-Ingersoll-Ross
        1985, eq. 23). At ``maturity < QL_EPSILON`` the option is at
        immediate exercise and pays the intrinsic value.
        """
        qassert.require(strike > 0.0, "strike must be positive")
        discount_t = self.discount_bond_scalar(0.0, maturity, self.x0())
        discount_s = self.discount_bond_scalar(0.0, bond_maturity, self.x0())

        if maturity < QL_EPSILON:
            if option_type == OptionType.Call:
                return max(discount_s - strike, 0.0)
            return max(strike - discount_s, 0.0)

        sigma2 = self.sigma() * self.sigma()
        h = math.sqrt(self.k() * self.k() + 2.0 * sigma2)
        b = self._b(maturity, bond_maturity)
        rho = 2.0 * h / (sigma2 * (math.exp(h * maturity) - 1.0))
        psi = (self.k() + h) / sigma2

        df = 4.0 * self.k() * self.theta() / sigma2
        ncps = 2.0 * rho * rho * self.x0() * math.exp(h * maturity) / (rho + psi + b)
        ncpt = 2.0 * rho * rho * self.x0() * math.exp(h * maturity) / (rho + psi)
        chis = NonCentralCumulativeChiSquareDistribution(df, ncps)
        chit = NonCentralCumulativeChiSquareDistribution(df, ncpt)
        z = math.log(self._a(maturity, bond_maturity) / strike) / b
        call = (
            discount_s * chis(2.0 * z * (rho + psi + b))
            - strike * discount_t * chit(2.0 * z * (rho + psi))
        )
        if option_type == OptionType.Call:
            return call
        return call - discount_s + strike * discount_t
