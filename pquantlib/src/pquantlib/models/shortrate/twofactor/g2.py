"""G2 — Brigo-Mercurio two-additive-factor Gaussian short-rate model (G2++).

# C++ parity: ql/models/shortrate/twofactormodels/g2.{hpp,cpp} (v1.42.1).

Dynamics:

* ``r_t = phi(t) + x_t + y_t``
* ``dx_t = -a x_t dt + sigma dW^1_t``,  ``x_0 = 0``
* ``dy_t = -b y_t dt + eta   dW^2_t``,  ``y_0 = 0``
* ``dW^1 dW^2 = rho dt``

``phi(t)`` is fixed analytically by the term-structure (matches the
``FittingParameter`` nested class in C++ g2.hpp:142-180):

::

    phi(t) = f(t) + 0.5*((sigma*(1-exp(-a*t))/a))^2
                  + 0.5*((eta  *(1-exp(-b*t))/b))^2
                  + rho * (sigma*(1-exp(-a*t))/a) * (eta*(1-exp(-b*t))/b)

where ``f(t)`` is the instantaneous forward rate.

Closed-form ``discount_bond(now, maturity, x, y)`` via ``A(t,T) *
exp(-B(a,T-t)*x - B(b,T-t)*y)`` (g2.cpp:75-77).

Closed-form ``discount_bond_option`` via Black-76 with ``sigmaP(t,T)``
log-normal vol surface (g2.cpp:79-87).

``swaption(...)`` via 1-D segment integration of an inner Brent root
search — porting the ``SwaptionPricingFunction`` nested class (g2.cpp:111-216).

Multiple inheritance note: the C++ class derives from ``TwoFactorModel,
AffineModel, TermStructureConsistentModel``. Python doesn't ship a
concrete ``AffineModel`` base — its surface is captured by
``ShortRateModelProtocol`` (structural typing). The Python class
therefore inherits ``TwoFactorModel + TermStructureConsistentModel``
only; ``discount`` / ``discount_bond`` / ``discount_bond_option`` are
implemented directly on G2 to satisfy ``ShortRateModelProtocol``.

Deferred from L4-D: the C++ ``tree(grid)`` override from TwoFactorModel
is inherited but raises LibraryException (see two_factor_model.py for
the deferral note). G2's calibration via ``calibrate`` (inherited from
CalibratedModel) works once concrete SwaptionHelpers from L4-C land.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.integrals.segment import SegmentIntegral
from pquantlib.math.optimization.constraint import BoundaryConstraint, PositiveConstraint
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.models.model import (
    TermStructureConsistentModel,
    _PrivateConstraint,  # pyright: ignore[reportPrivateUsage]
)
from pquantlib.models.parameter import ConstantParameter
from pquantlib.models.shortrate.two_factor_model import ShortRateDynamics, TwoFactorModel
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

_PHI = CumulativeNormalDistribution()


class _FittingParameter:
    """Analytical phi(t) for the G2++ fitting parameter.

    # C++ parity: nested ``G2::FittingParameter`` in g2.hpp:142-180
    # (v1.42.1). The C++ class derives from ``TermStructureFittingParameter``
    # with a custom ``Impl`` that evaluates phi(t) closed-form; pquantlib
    # wraps the same closed-form as a plain callable since the only
    # consumer is the ``Dynamics`` nested class.
    """

    __slots__ = ("_a", "_b", "_eta", "_rho", "_sigma", "_term_structure")

    def __init__(
        self,
        term_structure: YieldTermStructure,
        a: float,
        sigma: float,
        b: float,
        eta: float,
        rho: float,
    ) -> None:
        self._term_structure: YieldTermStructure = term_structure
        self._a: float = a
        self._sigma: float = sigma
        self._b: float = b
        self._eta: float = eta
        self._rho: float = rho

    def __call__(self, t: float) -> float:
        # C++ parity: g2.hpp:155-164 — ``FittingParameter::Impl::value``.
        forward = self._term_structure.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        temp1 = self._sigma * (1.0 - math.exp(-self._a * t)) / self._a
        temp2 = self._eta * (1.0 - math.exp(-self._b * t)) / self._b
        return 0.5 * temp1 * temp1 + 0.5 * temp2 * temp2 + self._rho * temp1 * temp2 + forward


class _Dynamics(ShortRateDynamics):
    """G2 short-rate dynamics — phi(t) + x + y.

    # C++ parity: nested ``G2::Dynamics`` in g2.hpp:118-130.
    """

    __slots__ = ("_fitting",)

    def __init__(
        self,
        fitting: _FittingParameter,
        a: float,
        sigma: float,
        b: float,
        eta: float,
        rho: float,
    ) -> None:
        super().__init__(
            x_process=OrnsteinUhlenbeckProcess(a, sigma),
            y_process=OrnsteinUhlenbeckProcess(b, eta),
            correlation=rho,
        )
        self._fitting: _FittingParameter = fitting

    def short_rate(self, t: float, x: float, y: float) -> float:
        # C++ parity: g2.hpp:126 — ``fitting_(t) + x + y``.
        return self._fitting(t) + x + y


class _SwaptionPricingFunction:
    """Inner integrand for G2.swaption.

    # C++ parity: nested ``G2::SwaptionPricingFunction`` in
    # g2.cpp:111-216 (v1.42.1).
    """

    __slots__ = (
        "_A",
        "_Ba",
        "_Bb",
        "_T",
        "_a",
        "_b",
        "_eta",
        "_mu_x",
        "_mu_y",
        "_pay_times",
        "_rate",
        "_rho",
        "_rho_xy",
        "_sigma",
        "_sigma_x",
        "_sigma_y",
        "_size",
        "_w",
    )

    def __init__(
        self,
        a: float,
        sigma: float,
        b: float,
        eta: float,
        rho: float,
        w: float,
        start: float,
        pay_times: list[float],
        fixed_rate: float,
        model: G2,
    ) -> None:
        # C++ parity: g2.cpp:121-150 — caches the constants and the
        # A/Ba/Bb arrays at the start time T_.
        self._a: float = a
        self._sigma: float = sigma
        self._b: float = b
        self._eta: float = eta
        self._rho: float = rho
        self._w: float = w
        self._T: float = start  # pyright: ignore[reportConstantRedefinition]
        self._pay_times: list[float] = list(pay_times)
        self._rate: float = fixed_rate
        self._size: int = len(pay_times)

        self._sigma_x: float = sigma * math.sqrt(
            0.5 * (1.0 - math.exp(-2.0 * a * start)) / a
        )
        self._sigma_y: float = eta * math.sqrt(
            0.5 * (1.0 - math.exp(-2.0 * b * start)) / b
        )
        self._rho_xy: float = (
            rho * eta * sigma * (1.0 - math.exp(-(a + b) * start))
        ) / ((a + b) * self._sigma_x * self._sigma_y)

        temp = sigma * sigma / (a * a)
        self._mu_x: float = -(
            (temp + rho * sigma * eta / (a * b)) * (1.0 - math.exp(-a * start))
            - 0.5 * temp * (1.0 - math.exp(-2.0 * a * start))
            - rho * sigma * eta / (b * (a + b)) * (1.0 - math.exp(-(b + a) * start))
        )
        temp = eta * eta / (b * b)
        self._mu_y: float = -(
            (temp + rho * sigma * eta / (a * b)) * (1.0 - math.exp(-b * start))
            - 0.5 * temp * (1.0 - math.exp(-2.0 * b * start))
            - rho * sigma * eta / (a * (a + b)) * (1.0 - math.exp(-(b + a) * start))
        )

        # Precompute A, Ba, Bb arrays.
        self._A: npt.NDArray[np.float64] = np.empty(self._size, dtype=np.float64)  # pyright: ignore[reportConstantRedefinition]
        self._Ba: npt.NDArray[np.float64] = np.empty(self._size, dtype=np.float64)
        self._Bb: npt.NDArray[np.float64] = np.empty(self._size, dtype=np.float64)
        for i, ti in enumerate(self._pay_times):
            self._A[i] = model.A(start, ti)
            self._Ba[i] = model.B(a, ti - start)
            self._Bb[i] = model.B(b, ti - start)

    @property
    def mu_x(self) -> float:
        return self._mu_x

    @property
    def sigma_x(self) -> float:
        return self._sigma_x

    def __call__(self, x: float) -> float:
        # C++ parity: g2.cpp:154-189 — the integrand evaluated at x.
        # Builds lambda[i] = c_i * A_i * exp(-Ba_i * x), then solves
        # 1 - sum(lambda_i * exp(-Bb_i * y)) = 0 for y_bar via Brent,
        # then computes the analytical option-value contribution.
        temp = (x - self._mu_x) / self._sigma_x
        txy = math.sqrt(1.0 - self._rho_xy * self._rho_xy)

        lam = np.empty(self._size, dtype=np.float64)
        for i in range(self._size):
            tau = (
                self._pay_times[0] - self._T
                if i == 0
                else self._pay_times[i] - self._pay_times[i - 1]
            )
            c = 1.0 + self._rate * tau if i == self._size - 1 else self._rate * tau
            lam[i] = c * self._A[i] * math.exp(-self._Ba[i] * x)

        # Inner Brent on f(y) = 1 - sum(lam[i] * exp(-Bb[i] * y)).
        bb = self._Bb

        def f(y: float) -> float:
            value = 1.0
            for i in range(self._size):
                value -= float(lam[i]) * math.exp(-float(bb[i]) * y)
            return value

        s1d = Brent()
        s1d.set_max_evaluations(1000)
        search_bound = max(10.0 * self._sigma_y, 1.0)
        yb = s1d.solve(f, 1e-6, 0.00, -search_bound, search_bound)

        h1 = (yb - self._mu_y) / (self._sigma_y * txy) - self._rho_xy * (
            x - self._mu_x
        ) / (self._sigma_x * txy)
        value = _PHI(-self._w * h1)

        for i in range(self._size):
            h2 = h1 + self._Bb[i] * self._sigma_y * math.sqrt(
                1.0 - self._rho_xy * self._rho_xy
            )
            kappa = -self._Bb[i] * (
                self._mu_y
                - 0.5 * txy * txy * self._sigma_y * self._sigma_y * self._Bb[i]
                + self._rho_xy * self._sigma_y * (x - self._mu_x) / self._sigma_x
            )
            value -= float(lam[i]) * math.exp(kappa) * _PHI(-self._w * h2)

        return math.exp(-0.5 * temp * temp) * value / (self._sigma_x * math.sqrt(2.0 * math.pi))


class G2(TwoFactorModel, TermStructureConsistentModel):
    """Brigo-Mercurio G2++ Gaussian two-additive-factor short-rate model.

    # C++ parity: ``class G2`` in
    # ql/models/shortrate/twofactormodels/g2.hpp:54-116 (v1.42.1).

    Multiple-inheritance note: see module docstring — Python omits the
    explicit ``AffineModel`` base since its surface is captured by
    ``ShortRateModelProtocol``.
    """

    def __init__(
        self,
        term_structure: YieldTermStructure,
        a: float = 0.1,
        sigma: float = 0.01,
        b: float = 0.1,
        eta: float = 0.01,
        rho: float = -0.75,
    ) -> None:
        # C++ parity: g2.cpp:31-46 — initialises five ConstantParameters
        # bound to arguments_[0..4]. Python rebinds the list slots.
        # Diamond inheritance note: the MRO chains
        # TwoFactorModel -> ShortRateModel -> CalibratedModel -> Model ->
        # TermStructureConsistentModel -> Observable. We call
        # ``TwoFactorModel.__init__`` once; the cooperative super()
        # chain threads through ``TermStructureConsistentModel.__init__``
        # with ``term_structure=None`` (the diamond-cooperative default).
        # We then late-bind the term structure via ``_set_term_structure``.
        TwoFactorModel.__init__(self, n_params=5)
        self._set_term_structure(term_structure)
        self._arguments[0] = ConstantParameter(a, PositiveConstraint())
        self._arguments[1] = ConstantParameter(sigma, PositiveConstraint())
        self._arguments[2] = ConstantParameter(b, PositiveConstraint())
        self._arguments[3] = ConstantParameter(eta, PositiveConstraint())
        self._arguments[4] = ConstantParameter(rho, BoundaryConstraint(-1.0, 1.0))
        # Rebuild the model's PrivateConstraint over the new arguments
        # list since the C++ ctor doesn't but pquantlib has a stale
        # constraint snapshot.
        self._constraint = _PrivateConstraint(self._arguments)

        # phi initialised analytically — no fitting cache required.
        self._phi: _FittingParameter = _FittingParameter(
            term_structure, a, sigma, b, eta, rho
        )
        self.generate_arguments()
        term_structure.register_with(self)

    # --- inspectors (param-evaluating at t=0) ---------------------------

    def a(self) -> float:
        return self._arguments[0](0.0)

    def sigma(self) -> float:
        return self._arguments[1](0.0)

    def b(self) -> float:
        return self._arguments[2](0.0)

    def eta(self) -> float:
        return self._arguments[3](0.0)

    def rho(self) -> float:
        return self._arguments[4](0.0)

    # --- generate_arguments hook ----------------------------------------

    def generate_arguments(self) -> None:
        # C++ parity: g2.cpp:53-57. Refreshes phi after a param change.
        self._phi = _FittingParameter(
            self.term_structure, self.a(), self.sigma(), self.b(), self.eta(), self.rho()
        )

    # --- closed-form helpers --------------------------------------------

    def sigma_p(self, t: float, s: float) -> float:
        """Lognormal vol of P(t,s) used in ``discount_bond_option``.

        # C++ parity: ``G2::sigmaP`` in g2.cpp:59-73.
        """
        a = self.a()
        b = self.b()
        sigma = self.sigma()
        eta = self.eta()
        rho = self.rho()
        temp = 1.0 - math.exp(-(a + b) * t)
        temp1 = 1.0 - math.exp(-a * (s - t))
        temp2 = 1.0 - math.exp(-b * (s - t))
        a3 = a * a * a
        b3 = b * b * b
        sigma2 = sigma * sigma
        eta2 = eta * eta
        value = (
            0.5 * sigma2 * temp1 * temp1 * (1.0 - math.exp(-2.0 * a * t)) / a3
            + 0.5 * eta2 * temp2 * temp2 * (1.0 - math.exp(-2.0 * b * t)) / b3
            + 2.0 * rho * sigma * eta / (a * b * (a + b)) * temp1 * temp2 * temp
        )
        return math.sqrt(value)

    def V(self, t: float) -> float:  # noqa: N802 — math symbol from C++ source
        """Integrated variance helper for the analytical ``A(t,T)``.

        # C++ parity: ``G2::V`` in g2.cpp:89-100.
        """
        a = self.a()
        b = self.b()
        sigma = self.sigma()
        eta = self.eta()
        rho = self.rho()
        expat = math.exp(-a * t)
        expbt = math.exp(-b * t)
        cx = sigma / a
        cy = eta / b
        value_x = cx * cx * (t + (2.0 * expat - 0.5 * expat * expat - 1.5) / a)
        value_y = cy * cy * (t + (2.0 * expbt - 0.5 * expbt * expbt - 1.5) / b)
        cross = (
            2.0 * rho * cx * cy
            * (
                t
                + (expat - 1.0) / a
                + (expbt - 1.0) / b
                - (expat * expbt - 1.0) / (a + b)
            )
        )
        return value_x + value_y + cross

    def A(self, t: float, T: float) -> float:  # noqa: N802, N803 — math symbol
        """Bond-price coefficient ``A(t,T)``.

        # C++ parity: ``G2::A`` in g2.cpp:102-105.
        """
        return (
            self.term_structure.discount(T)
            / self.term_structure.discount(t)
            * math.exp(0.5 * (self.V(T - t) - self.V(T) + self.V(t)))
        )

    def B(self, x: float, t: float) -> float:  # noqa: N802 — math symbol
        """Bond-price coefficient ``B(x,t) = (1-exp(-x*t))/x``.

        # C++ parity: ``G2::B`` in g2.cpp:107-109.
        """
        return (1.0 - math.exp(-x * t)) / x

    # --- TwoFactorModel.dynamics override -------------------------------

    def dynamics(self) -> ShortRateDynamics:
        # C++ parity: g2.cpp:48-51.
        return _Dynamics(self._phi, self.a(), self.sigma(), self.b(), self.eta(), self.rho())

    # --- AffineModel-shaped methods (Protocol-compatible) ---------------

    def discount(self, t: float) -> float:
        # C++ parity: g2.hpp:85 — ``return termStructure()->discount(t)``.
        return self.term_structure.discount(t)

    def discount_bond(
        self,
        now: float,
        maturity: float,
        x_or_factors: float | npt.NDArray[np.float64],
        y: float | None = None,
    ) -> float:
        """Closed-form discount-bond ``P(now, maturity, x, y)``.

        # C++ parity: g2.hpp:67-77 — two overloads merged. The Array
        # overload calls into the (Real, Real) overload; in Python we
        # branch on the type of ``x_or_factors``.
        """
        if isinstance(x_or_factors, np.ndarray):
            qassert.require(
                x_or_factors.size > 1, "g2 model needs two factors to compute discount bond"
            )
            xv = float(x_or_factors[0])
            yv = float(x_or_factors[1])
        else:
            assert y is not None, "discount_bond: scalar overload requires y"
            xv = float(x_or_factors)
            yv = float(y)
        a = self.a()
        b = self.b()
        return self.A(now, maturity) * math.exp(
            -self.B(a, maturity - now) * xv - self.B(b, maturity - now) * yv
        )

    def discount_bond_option(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """Closed-form Black-formula discount-bond option.

        # C++ parity: ``G2::discountBondOption`` in g2.cpp:79-87.
        """
        v = self.sigma_p(maturity, bond_maturity)
        f = self.term_structure.discount(bond_maturity)
        k = self.term_structure.discount(maturity) * strike
        return black_formula(option_type, k, f, v)

    # --- swaption -------------------------------------------------------

    def swaption(
        self,
        nominal: float,
        is_payer: bool,
        start: float,
        fixed_pay_times: list[float],
        fixed_rate: float,
        range_: float,
        intervals: int,
    ) -> float:
        """G2 swaption analytical via 1-D segment integration.

        # C++ parity: ``G2::swaption(const Swaption::arguments&, ...)`` in
        # g2.cpp:218-246.

        Python signature divergence: the C++ method takes a
        ``Swaption::arguments`` struct (which carries Type / Nominal /
        floatingResetDates / fixedPayDates). pquantlib hasn't ported the
        Swaption instrument yet (L4-C scope), so this method takes the
        unpacked primitives directly:

        * ``nominal`` — swap notional.
        * ``is_payer`` — True for payer-of-fixed (w=+1), False for
          receiver (w=-1).
        * ``start`` — year fraction to swap start (corresponds to the
          C++ ``floatingResetDates[0]`` via the curve's daycounter).
        * ``fixed_pay_times`` — year fractions to each fixed leg
          payment date.
        * ``fixed_rate`` — fixed coupon.
        * ``range_`` — number of standard deviations to integrate over
          (matches the C++ ``range`` arg).
        * ``intervals`` — number of segment subdivisions.
        """
        w = 1.0 if is_payer else -1.0
        fn = _SwaptionPricingFunction(
            self.a(),
            self.sigma(),
            self.b(),
            self.eta(),
            self.rho(),
            w,
            start,
            fixed_pay_times,
            fixed_rate,
            self,
        )
        upper = fn.mu_x + range_ * fn.sigma_x
        lower = fn.mu_x - range_ * fn.sigma_x
        integrator = SegmentIntegral(intervals)
        return nominal * w * self.term_structure.discount(start) * integrator(fn, lower, upper)


__all__ = ["G2"]
