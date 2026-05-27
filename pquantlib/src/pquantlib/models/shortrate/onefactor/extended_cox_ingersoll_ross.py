"""Extended Cox-Ingersoll-Ross — fits the input yield curve.

# C++ parity: ql/models/shortrate/onefactormodels/extendedcoxingersollross.{hpp,cpp} (v1.42.1).

Extends CIR with a deterministic time-dependent shift ``phi(t)``:

    r(t) = phi(t) + y(t)

where ``y(t)`` is a standard CIR process and

    phi(t) = f(t) - 2 k theta (e^{th} - 1) / (2h + (k+h)(e^{th} - 1))
                  - 4 x0 h^2 e^{th} / (2h + (k+h)(e^{th} - 1))^2

with ``h = sqrt(k^2 + 2 sigma^2)``. The phi(t) is chosen so the model
exactly reprices the input curve.

The discount-bond A(t, T) is adjusted using the CIR A(t, T)/B(t, T)
and the curve discounts (extendedcoxingersollross.cpp:53-60), while
the discount-bond European option uses a shifted non-central chi-square
decomposition.

NOTE: as in the CIR module, the C++ class is marked with a bug warning
("not tested enough to guarantee functionality"). We preserve all
formulae verbatim.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.non_central_chi_square_distribution import (
    NonCentralCumulativeChiSquareDistribution,
)
from pquantlib.models.model import TermStructureConsistentModel
from pquantlib.models.parameter import (
    Parameter,
    ParameterImpl,
    TermStructureFittingParameter,
)
from pquantlib.models.shortrate.onefactor.cox_ingersoll_ross import CoxIngersollRoss
from pquantlib.models.shortrate.onefactor.one_factor_model import (
    ShortRateDynamics,
)
from pquantlib.payoffs import OptionType
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class _ECIRFittingImpl(ParameterImpl):
    """Analytical phi(t) for Extended Cox-Ingersoll-Ross.

    # C++ parity: nested ``ExtendedCoxIngersollRoss::FittingParameter::Impl``
    # in extendedcoxingersollross.hpp:116-137.
    """

    __slots__ = ("_k", "_sigma", "_term_structure", "_theta", "_x0")

    def __init__(
        self,
        term_structure: YieldTermStructure,
        theta: float,
        k: float,
        sigma: float,
        x0: float,
    ) -> None:
        self._term_structure: YieldTermStructure = term_structure
        self._theta: float = float(theta)
        self._k: float = float(k)
        self._sigma: float = float(sigma)
        self._x0: float = float(x0)

    def value(self, params: npt.NDArray[np.float64], t: float) -> float:
        # C++ parity: extendedcoxingersollross.hpp:122-132.
        forward = self._term_structure.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        h = math.sqrt(self._k * self._k + 2.0 * self._sigma * self._sigma)
        expth = math.exp(t * h)
        temp = 2.0 * h + (self._k + h) * (expth - 1.0)
        return (
            forward
            - 2.0 * self._k * self._theta * (expth - 1.0) / temp
            - self._x0 * 4.0 * h * h * expth / (temp * temp)
        )


class _ECIRFittingParameter(TermStructureFittingParameter):
    """Wrapper Parameter for the ECIR phi(t).

    # C++ parity: ``class ExtendedCoxIngersollRoss::FittingParameter``
    # at extendedcoxingersollross.hpp:113-144.
    """

    def __init__(
        self,
        term_structure: YieldTermStructure,
        theta: float,
        k: float,
        sigma: float,
        x0: float,
    ) -> None:
        super().__init__(_ECIRFittingImpl(term_structure, theta, k, sigma, x0))


class _ECIRDynamics(ShortRateDynamics):
    """ECIR short-rate dynamics: r(t) = phi(t) + y(t), y a CIR process.

    # C++ parity: ``class ExtendedCoxIngersollRoss::Dynamics`` in
    # extendedcoxingersollross.hpp:90-101.
    """

    __slots__ = ("_phi",)

    def __init__(
        self,
        phi: Parameter,
        theta: float,
        k: float,
        sigma: float,
        x0: float,
    ) -> None:
        process = CoxIngersollRossProcess(speed=k, vol=sigma, x0=x0, level=theta)
        super().__init__(process)
        self._phi: Parameter = phi

    def variable(self, t: float, r: float) -> float:
        # C++ parity: extendedcoxingersollross.hpp:96.
        return r - self._phi(t)

    def short_rate(self, t: float, variable: float) -> float:
        # C++ parity: extendedcoxingersollross.hpp:97.
        return variable + self._phi(t)


class ExtendedCoxIngersollRoss(CoxIngersollRoss, TermStructureConsistentModel):
    """Extended Cox-Ingersoll-Ross model — fits the input curve.

    # C++ parity: ``class ExtendedCoxIngersollRoss : public
    # CoxIngersollRoss, public TermStructureConsistentModel`` in
    # extendedcoxingersollross.hpp:47-76 (v1.42.1).
    """

    __slots__ = ("_phi",)

    def __init__(
        self,
        term_structure: YieldTermStructure,
        theta: float = 0.1,
        k: float = 0.1,
        sigma: float = 0.1,
        x0: float = 0.05,
        with_feller_constraint: bool = True,
    ) -> None:
        """Construct ECIR consistent with the given curve.

        # C++ parity: extendedcoxingersollross.cpp:27-34.

        Forwards (x0, theta, k, sigma, with_feller_constraint) to the
        CIR base, then attaches the curve via TermStructureConsistentModel
        with the cooperative-super() default-None escape (see HullWhite
        for the same MRO pattern).
        """
        # C++ parity: extendedcoxingersollross.cpp:31 — CIR base.
        CoxIngersollRoss.__init__(self, x0, theta, k, sigma, with_feller_constraint)
        # cooperative-super() escape (per L4-B HullWhite pattern).
        self._term_structure = term_structure
        # C++ parity: extendedcoxingersollross.cpp:33 — populate phi
        # using the just-set self.x0() / self.theta() / etc.
        self._phi: Parameter = _ECIRFittingParameter(
            term_structure, self.theta(), self.k(), self.sigma(), self.x0()
        )
        term_structure.register_with(self)

    # --- generated arguments hook ---------------------------------------

    def generate_arguments(self) -> None:
        """Refresh ``phi(t)`` after a calibration step changes (theta, k, sigma, r0).

        # C++ parity: inline ``ExtendedCoxIngersollRoss::generateArguments``
        # at extendedcoxingersollross.hpp:154-156.
        """
        self._phi = _ECIRFittingParameter(
            self.term_structure, self.theta(), self.k(), self.sigma(), self.x0()
        )

    # --- ShortRateModel surface -----------------------------------------

    def dynamics(self) -> ShortRateDynamics:
        # C++ parity: extendedcoxingersollross.hpp:148-152 (inline).
        return _ECIRDynamics(self._phi, self.theta(), self.k(), self.sigma(), self.x0())

    # --- A(t, T) override -----------------------------------------------

    def _a(self, t: float, t_maturity: float) -> float:
        # C++ parity: extendedcoxingersollross.cpp:53-60.
        ts = self.term_structure
        pt = ts.discount(t)
        ps = ts.discount(t_maturity)
        cir_a_ts = CoxIngersollRoss._a(self, t, t_maturity)
        cir_a_0t = CoxIngersollRoss._a(self, 0.0, t)
        cir_a_0s = CoxIngersollRoss._a(self, 0.0, t_maturity)
        b_ts = self._b(t, t_maturity)
        b_0t = self._b(0.0, t)
        b_0s = self._b(0.0, t_maturity)
        numerator = ps * cir_a_0t * math.exp(-b_0t * self.x0())
        denominator = pt * cir_a_0s * math.exp(-b_0s * self.x0())
        return cir_a_ts * math.exp(b_ts * self._phi(t)) * (numerator / denominator)

    # --- option pricing -------------------------------------------------

    def discount_bond_option(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """Closed-form European discount-bond option (shifted CIR).

        # C++ parity: extendedcoxingersollross.cpp:62-106.
        """
        qassert.require(strike > 0.0, "strike must be positive")
        discount_t = self.term_structure.discount(maturity)
        discount_s = self.term_structure.discount(bond_maturity)
        if maturity < QL_EPSILON:
            if option_type == OptionType.Call:
                return max(discount_s - strike, 0.0)
            return max(strike - discount_s, 0.0)

        sigma2 = self.sigma() * self.sigma()
        h = math.sqrt(self.k() * self.k() + 2.0 * sigma2)
        # r0_curve: the curve's instantaneous forward at 0 (NOT the model's x0,
        # which is the dy process's starting value).
        r0_curve = self.term_structure.forward_rate(
            0.0, 0.0, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        b = self._b(maturity, bond_maturity)
        rho = 2.0 * h / (sigma2 * (math.exp(h * maturity) - 1.0))
        psi = (self.k() + h) / sigma2

        df = 4.0 * self.k() * self.theta() / sigma2
        ncps = 2.0 * rho * rho * (r0_curve - self._phi(0.0)) * math.exp(h * maturity) / (rho + psi + b)
        ncpt = 2.0 * rho * rho * (r0_curve - self._phi(0.0)) * math.exp(h * maturity) / (rho + psi)

        chis = NonCentralCumulativeChiSquareDistribution(df, ncps)
        chit = NonCentralCumulativeChiSquareDistribution(df, ncpt)

        cir_a_0t = CoxIngersollRoss._a(self, 0.0, maturity)
        cir_a_0s = CoxIngersollRoss._a(self, 0.0, bond_maturity)
        b_0t = self._b(0.0, maturity)
        b_0s = self._b(0.0, bond_maturity)
        discount_shift = (
            (discount_t * cir_a_0s * math.exp(-b_0s * self.x0()))
            / (discount_s * cir_a_0t * math.exp(-b_0t * self.x0()))
        )
        cir_a_ts = CoxIngersollRoss._a(self, maturity, bond_maturity)
        z = (math.log(cir_a_ts / strike) - math.log(discount_shift)) / b
        call = (
            discount_s * chis(2.0 * z * (rho + psi + b))
            - strike * discount_t * chit(2.0 * z * (rho + psi))
        )
        if option_type == OptionType.Call:
            return call
        return call - discount_s + strike * discount_t
