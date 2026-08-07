"""GeneralizedHullWhite — Hull-White with time-dependent reversion + vol.

# C++ parity: ql/experimental/shortrate/generalizedhullwhite.{hpp,cpp}
# (v1.42.1).

Implements the generalized / Black-Karasinski-style short-rate model

    d f(r_t) = (theta(t) - alpha(t) f(r_t)) dt + sigma(t) dW_t,

where the mean reversion ``alpha(t)`` and volatility ``sigma(t)`` are
piecewise (linear-flat-interpolated) functions of time, and ``f`` /
``fInverse`` are optional state-mapping functions (identity by default,
giving classical Hull-White).

Two construction paths, both ported:

1. **Piecewise** — ``(yield_ts, speed_dates, vol_dates, speeds, vols,
   f, f_inverse)``: builds ``LinearFlat`` interpolations over the
   year-fraction pillars.
2. **Analytic-fitting** — ``(yield_ts, a, sigma)``: a single constant
   reversion + volatility, fitted analytically to the input curve. This
   is the path used for fast European-bond-option pricing.

The analytic surface — ``A(t,T)``, ``B(t,T)``, ``V(t,T)`` (Gurrieri,
Nakabayashi & Wong 2009 "Calibration Methods of Hull-White Model",
https://ssrn.com/abstract=1514192, equations 30/31/37/43) and
``discount_bond_option`` (a ``blackFormula`` once the bond vol is
known) — is fully ported and cross-validated against the classical
``HullWhite`` model at constant reversion + vol.

Carve-out: the numerical ``tree(grid)`` path (TrinomialTree +
ShortRateTree + Brent term-structure fitting) and the ``f``-calibrated
non-linear mapping are **deferred**. ``dynamics()`` raises (matching the
C++ ``QL_FAIL("use HWdynamics()")``); the analytic Hull-White dynamics
are available via ``hw_dynamics()``. See module ``W8-D`` notes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.shortrate.generalized_ornstein_uhlenbeck_process import (
    GeneralizedOrnsteinUhlenbeckProcess,
)
from pquantlib.experimental.shortrate.linear_flat_interpolation import (
    LinearFlat,
    LinearFlatInterpolation,
)
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.integrals.simpson import SimpsonIntegral
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.optimization.constraint import (
    Constraint,
    NoConstraint,
    PositiveConstraint,
)
from pquantlib.models.model import TermStructureConsistentModel
from pquantlib.models.parameter import (
    Parameter,
    ParameterImpl,
    TermStructureFittingParameter,
)
from pquantlib.models.shortrate.onefactor.one_factor_affine_model import (
    OneFactorAffineModel,
)
from pquantlib.models.shortrate.onefactor.one_factor_model import ShortRateDynamics
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.date import Date


def _identity(x: float) -> float:
    return x


class InterpolationParameter(Parameter):
    """Parameter that holds an interpolation object.

    # C++ parity: ``class InterpolationParameter : public Parameter`` in
    # generalizedhullwhite.hpp:37-61 (v1.43).

    The whole point of the class is the *aliasing* between the parameter's
    free-parameter vector and the interpolation's ordinates: C++ builds the
    interpolation with ``a_.params().begin()`` as its y-iterator
    (generalizedhullwhite.hpp:200-203), so a calibrator writing through
    :meth:`Parameter.set_param` moves the curve with no rebuild. PQuantLib
    reproduces this by handing :meth:`Parameter.params` — the parameter's own
    ``numpy`` array — to the interpolation constructor, which keeps it by
    reference.

    # C++ parity note: only the *ordinates* are live. The interpolation's
    # cached slopes are recomputed by ``Interpolation::update()``, which C++
    # calls from ``GeneralizedHullWhite::generateArguments``
    # (generalizedhullwhite.cpp:122-126) — not from ``setParam``. So between a
    # bare ``set_param`` and the next ``update()`` the curve reads *new*
    # ordinates through *stale* slopes. That is reproduced verbatim and pinned
    # by ``ip_values_after_setparam`` in the misc probe.
    """

    class Impl(ParameterImpl):
        """Interpolation-backed lookup strategy.

        # C++ parity: ``InterpolationParameter::Impl`` in
        # generalizedhullwhite.hpp:39-45.
        """

        __slots__ = ("_interpolator",)

        def __init__(self) -> None:
            self._interpolator: Interpolation | None = None

        def value(self, params: npt.NDArray[np.float64], t: float) -> float:
            # C++ parity: generalizedhullwhite.hpp:41 — ``return
            # interpolator_(t);``. ``params`` is deliberately unused: the
            # numbers reach the interpolation through the shared array, and
            # the extrapolation policy is baked into the interpolation object
            # (C++ calls ``enableExtrapolation()`` before ``reset``).
            del params
            if self._interpolator is None:
                raise RuntimeError("InterpolationParameter has not been reset()")
            return self._interpolator(t)

        def reset(self, interp: Interpolation) -> None:
            # C++ parity: generalizedhullwhite.hpp:42.
            self._interpolator = interp

    def __init__(self, count: int, constraint: Constraint | None = None) -> None:
        # C++ parity: generalizedhullwhite.hpp:47-54.
        super().__init__(
            count,
            InterpolationParameter.Impl(),
            constraint if constraint is not None else NoConstraint(),
        )

    def reset(self, interp: Interpolation) -> None:
        """Install the interpolation this parameter evaluates.

        # C++ parity: generalizedhullwhite.hpp:55-60 — a
        # ``dynamic_pointer_cast`` that silently does nothing when the Impl is
        # not an ``InterpolationParameter::Impl``; the ``isinstance`` guard is
        # the same no-op-on-mismatch behaviour.
        """
        impl = self.impl
        if isinstance(impl, InterpolationParameter.Impl):
            impl.reset(interp)


class _GeneralizedHullWhiteFittingImpl(ParameterImpl):
    """Analytic term-structure fitting parameter phi(t).

    # C++ parity: ``GeneralizedHullWhite::FittingParameter::Impl`` in
    # generalizedhullwhite.hpp:270-296.

        phi(t) = f(t) + 0.5 * [sigma (1 - e^{-a t}) / a]^2

    with the algebraic ``a -> 0`` limit ``sigma * t`` for the bracket.
    """

    __slots__ = ("_a", "_sigma", "_term_structure")

    def __init__(self, term_structure: YieldTermStructure, a: float, sigma: float) -> None:
        self._term_structure: YieldTermStructure = term_structure
        self._a: float = a
        self._sigma: float = sigma

    def value(self, params: npt.NDArray[np.float64], t: float) -> float:
        # C++ parity: generalizedhullwhite.hpp:278-285.
        forward_rate = self._term_structure.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        if self._a < math.sqrt(QL_EPSILON):
            temp = self._sigma * t
        else:
            temp = self._sigma * (1.0 - math.exp(-self._a * t)) / self._a
        return forward_rate + 0.5 * temp * temp


class _GeneralizedHullWhiteFittingParameter(TermStructureFittingParameter):
    """phi(t) wrapper for the analytic Hull-White fitting impl.

    # C++ parity: ``GeneralizedHullWhite::FittingParameter`` in
    # generalizedhullwhite.hpp:270-296.
    """

    def __init__(self, term_structure: YieldTermStructure, a: float, sigma: float) -> None:
        super().__init__(_GeneralizedHullWhiteFittingImpl(term_structure, a, sigma))


class _GeneralizedHullWhiteDynamics(ShortRateDynamics):
    """Short-rate dynamics for the generalized Hull-White model.

    # C++ parity: ``GeneralizedHullWhite::Dynamics`` in
    # generalizedhullwhite.hpp:232-261.

    f(r_t) = x_t + g(t), where x_t follows a (generalized) OU process.
    The analytic flavour uses a plain ``OrnsteinUhlenbeckProcess``; the
    numerical flavour uses a ``GeneralizedOrnsteinUhlenbeckProcess``.
    """

    __slots__ = ("_f", "_f_inverse", "_fitting")

    def __init__(
        self,
        fitting: Parameter,
        process: StochasticProcess1D,
        f: Callable[[float], float],
        f_inverse: Callable[[float], float],
    ) -> None:
        super().__init__(process)
        self._fitting: Parameter = fitting
        self._f: Callable[[float], float] = f
        self._f_inverse: Callable[[float], float] = f_inverse

    def variable(self, t: float, r: float) -> float:
        # C++ parity: generalizedhullwhite.hpp:250 — f(r) - fitting(t).
        return self._f(r) - self._fitting(t)

    def short_rate(self, t: float, variable: float) -> float:
        # C++ parity: generalizedhullwhite.hpp:252 — fInverse(x + fitting(t)).
        return self._f_inverse(variable + self._fitting(t))


class GeneralizedHullWhite(OneFactorAffineModel, TermStructureConsistentModel):
    """Generalized Hull-White model (time-dependent reversion + vol).

    # C++ parity: ``class GeneralizedHullWhite : public OneFactorAffineModel,
    # public TermStructureConsistentModel`` in generalizedhullwhite.hpp:72-216.

    See the module docstring for the two construction paths and the
    ``tree()`` carve-out. The analytic-fitting accessors ``a()`` /
    ``sigma()`` return the constant reversion / vol at ``t = 0``.
    """

    __slots__ = (
        "_f",
        "_f_inverse",
        "_phi",
        "_speed_interp",
        "_speed_periods",
        "_vol_interp",
        "_vol_periods",
    )

    def __init__(
        self,
        yield_term_structure: YieldTermStructure,
        a: float = 0.1,
        sigma: float = 0.01,
    ) -> None:
        """Analytic-fitting constructor (constant a, sigma).

        # C++ parity: generalizedhullwhite.cpp:102-120 — the
        # ``(yieldtermStructure, a, sigma)`` overload. Internally builds a
        # single-pillar BackwardFlat structure at the reference date; with
        # one node every interpolation query is constant, so we model it as
        # constant callables directly.
        """
        OneFactorAffineModel.__init__(self, 2)
        # Cooperative super() reaches TermStructureConsistentModel with the
        # default None; set the curve explicitly (see HullWhite rationale).
        self._term_structure = yield_term_structure

        self._f: Callable[[float], float] = _identity
        self._f_inverse: Callable[[float], float] = _identity

        ref = yield_term_structure.reference_date()
        self._init_interpolations(
            yield_term_structure,
            [ref],
            [ref],
            [a],
            [sigma],
        )
        self._phi: Parameter = _GeneralizedHullWhiteFittingParameter(
            yield_term_structure, self.a(), self.sigma()
        )
        yield_term_structure.register_with(self)

    @classmethod
    def piecewise(
        cls,
        yield_term_structure: YieldTermStructure,
        speed_structure: list[Date],
        vol_structure: list[Date],
        speed: list[float],
        vol: list[float],
        f: Callable[[float], float] | None = None,
        f_inverse: Callable[[float], float] | None = None,
    ) -> GeneralizedHullWhite:
        """Piecewise constructor — LinearFlat reversion / vol structures.

        # C++ parity: generalizedhullwhite.cpp:82-99 — the main
        # ``(yieldtermStructure, speedstructure, volstructure, speed, vol,
        # f, fInverse)`` overload (LinearFlat traits).
        """
        obj = cls.__new__(cls)
        OneFactorAffineModel.__init__(obj, 2)
        obj._term_structure = yield_term_structure
        obj._f = f if f is not None else _identity
        obj._f_inverse = f_inverse if f_inverse is not None else _identity
        obj._init_interpolations(
            yield_term_structure, speed_structure, vol_structure, speed, vol
        )
        obj._phi = _GeneralizedHullWhiteFittingParameter(
            yield_term_structure, obj.a(), obj.sigma()
        )
        yield_term_structure.register_with(obj)
        return obj

    def _init_interpolations(
        self,
        yield_term_structure: YieldTermStructure,
        speed_structure: list[Date],
        vol_structure: list[Date],
        speed: list[float],
        vol: list[float],
    ) -> None:
        # C++ parity: generalizedhullwhite.hpp:166-215 (initialize) —
        # convert the date pillars to year-fractions, build LinearFlat
        # interpolations with extrapolation enabled. A single pillar gives
        # a constant function.
        qassert.require(
            len(speed_structure) == len(speed),
            "mean reversion inputs inconsistent",
        )
        qassert.require(
            len(vol_structure) == len(vol),
            "volatility inputs inconsistent",
        )
        dc = yield_term_structure.day_counter()
        ref = yield_term_structure.reference_date()
        self._speed_periods: list[float] = [
            dc.year_fraction(ref, d) for d in speed_structure
        ]
        self._vol_periods: list[float] = [
            dc.year_fraction(ref, d) for d in vol_structure
        ]
        # C++ parity: generalizedhullwhite.hpp:194-212 — a_ and sigma_ ARE
        # ``InterpolationParameter``s bound to arguments_[0] / arguments_[1],
        # with the reversion unconstrained and the volatility positive. The
        # interpolations are built over the parameters' own arrays so that
        # calibration (``set_params``) moves the curves.
        a_param = InterpolationParameter(len(self._speed_periods), NoConstraint())
        for i, s in enumerate(speed):
            a_param.set_param(i, float(s))
        sigma_param = InterpolationParameter(
            len(self._vol_periods), PositiveConstraint()
        )
        for i, v in enumerate(vol):
            sigma_param.set_param(i, float(v))
        self._arguments[0] = a_param
        self._arguments[1] = sigma_param
        self._rebuild_interpolations()

    def _rebuild_interpolations(self) -> None:
        """Rebuild the LinearFlat curves over the parameters' live arrays.

        # C++ parity: generalizedhullwhite.hpp:200-212 (initial build) and
        # generalizedhullwhite.cpp:122-124 (``speed_.update(); vol_.update();``
        # after a calibration step). C++ can call ``update()`` in place because
        # its interpolation holds an iterator into ``params()``; PQuantLib's
        # ``Interpolation`` recomputes its cached slopes only in its own
        # ``update()``, so re-installing an interpolation built over the same
        # (shared) array is the exact equivalent.
        """
        a_param = self._arguments[0]
        sigma_param = self._arguments[1]
        # LinearFlat requires >= 1 point; a single pillar -> constant curve
        # (the C++ analytic ctor uses BackwardFlat over one pillar, which is
        # likewise constant).
        self._speed_interp: LinearFlatInterpolation = LinearFlat.interpolate(
            np.asarray(self._speed_periods, dtype=np.float64), a_param.params
        )
        self._speed_interp.enable_extrapolation()
        self._vol_interp: LinearFlatInterpolation = LinearFlat.interpolate(
            np.asarray(self._vol_periods, dtype=np.float64), sigma_param.params
        )
        self._vol_interp.enable_extrapolation()
        if isinstance(a_param, InterpolationParameter):
            a_param.reset(self._speed_interp)
        if isinstance(sigma_param, InterpolationParameter):
            sigma_param.reset(self._vol_interp)

    # --- analytic accessors ---------------------------------------------

    def a(self) -> float:
        # C++ parity: generalizedhullwhite.hpp:130 — ``a_(0.0)``, i.e. the
        # InterpolationParameter evaluated at t = 0.
        return self._arguments[0](0.0)

    def sigma(self) -> float:
        # C++ parity: generalizedhullwhite.hpp:131 — ``sigma_(0.0)``.
        return self._arguments[1](0.0)

    def speed(self, t: float) -> float:
        """Reversion alpha(t).

        # C++ parity: generalizedhullwhite.hpp:133 returns the ``speed_``
        # interpolation itself as a ``std::function<Real(Time)>``; Python
        # exposes it as a call.
        """
        return self._speed_interp(t, allow_extrapolation=True)

    def vol(self, t: float) -> float:
        """Volatility sigma(t).

        # C++ parity: generalizedhullwhite.hpp:134 (``vol_``).
        """
        return self._vol_interp(t, allow_extrapolation=True)

    # --- Gurrieri B / V / A integrals -----------------------------------

    def _integrate_mean_reversion(self, t: float, t_to: float) -> float:
        # C++ parity: generalizedhullwhite.cpp:33-40 (anonymous helper).
        if (t_to - t) < QL_EPSILON:
            return 0.0
        integrator = SimpsonIntegral(1e-5, 1000)
        return integrator(lambda u: self._speed_interp(u, allow_extrapolation=True), t, t_to)

    def _b(self, t: float, t_maturity: float) -> float:
        # C++ parity: generalizedhullwhite.cpp:128-150 (eqns 30/31).
        ln_et = self._integrate_mean_reversion(0.0, t)
        et = math.exp(ln_et)
        b = 0.0
        n = min(int((t_maturity - t) * 365), 2000)
        if n == 0:
            n = 1
        dt = 0.5 * (t_maturity - t) / n
        total = 0.0
        _t = t
        c = self._speed_interp(_t, allow_extrapolation=True)
        _t += dt
        ln_acc = ln_et
        for _ in range(n):
            av = c
            bv = self._speed_interp(_t, allow_extrapolation=True)
            c = self._speed_interp(_t + dt, allow_extrapolation=True)
            total += (dt * (2.0 / 6.0)) * (av + 4.0 * bv + c)
            b += (2.0 * dt) / math.exp(ln_acc + total)
            _t += 2.0 * dt
        b *= et
        return b

    def _v(self, t: float, t_maturity: float) -> float:
        # C++ parity: generalizedhullwhite.cpp:152-179 (eqn 37).
        ln_et = self._integrate_mean_reversion(0.0, t)
        v = 0.0
        n = min(int((t_maturity - t) * 365), 2000)
        if n == 0:
            n = 1
        dt = 0.5 * (t_maturity - t) / n
        ln_e = ln_et
        _t = t
        vol = self._vol_interp(_t, allow_extrapolation=True)
        eu = math.exp(ln_e)
        c = eu * eu * vol * vol
        _t += dt
        for _ in range(n):
            av = c
            vol = self._vol_interp(_t, allow_extrapolation=True)
            ln_e += self._speed_interp(_t, allow_extrapolation=True) * dt
            eu = math.exp(ln_e)
            bv = eu * eu * vol * vol
            vol = self._vol_interp(_t + dt, allow_extrapolation=True)
            ln_e += self._speed_interp(_t + dt, allow_extrapolation=True) * dt
            eu = math.exp(ln_e)
            c = eu * eu * vol * vol
            v += (dt * (2.0 / 6.0)) * (av + 4.0 * bv + c)
            _t += 2.0 * dt
        return v / (eu * eu)

    def _a(self, t: float, t_maturity: float) -> float:
        # C++ parity: generalizedhullwhite.cpp:199-208 (eqn 43).
        ts = self.term_structure
        discount1 = ts.discount(t)
        discount2 = ts.discount(t_maturity)
        forward = ts.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        bt_t = self._b(t, t_maturity)
        vr = self._v(0.0, t)
        at_t = math.log(discount2 / discount1) + bt_t * forward - 0.5 * bt_t * bt_t * vr
        return math.exp(at_t)

    # --- option pricing -------------------------------------------------

    def discount_bond_option(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """European option on a discount bond (valid under Hull-White).

        # C++ parity: generalizedhullwhite.cpp:181-197 — Gurrieri et al.
        # bond-option pricing with time-varying sigma and mean reversion.
        """
        bt_t = self._b(maturity, bond_maturity)
        vr = self._v(0.0, maturity)
        vp = vr * bt_t * bt_t
        vol = math.sqrt(vp)
        f = self.term_structure.discount(bond_maturity)
        k = self.term_structure.discount(maturity) * strike
        return black_formula(option_type, k, f, vol)

    # --- dynamics -------------------------------------------------------

    def dynamics(self) -> ShortRateDynamics:
        # C++ parity: generalizedhullwhite.hpp:105-108 — deliberately fails;
        # use hw_dynamics() for the analytic Hull-White dynamics.
        raise LibraryException(
            "no defined process for generalized Hull-White model, use hw_dynamics()"
        )

    def hw_dynamics(self) -> ShortRateDynamics:
        # C++ parity: generalizedhullwhite.hpp:299-303 (HWdynamics inline).
        process = OrnsteinUhlenbeckProcess(self.a(), self.sigma())
        return _GeneralizedHullWhiteDynamics(self._phi, process, _identity, _identity)

    def numeric_dynamics(self) -> ShortRateDynamics:
        """Numerical (generalized-OU) dynamics used by the tree.

        # C++ parity: the ``Dynamics(phi, speed(), vol(), f_, fInverse_)``
        # built inside ``tree()`` — exposed here for completeness; the
        # ``tree()`` driver itself is carved out (see module docstring).
        """
        process = GeneralizedOrnsteinUhlenbeckProcess(self.speed, self.vol)
        phi = TermStructureFittingParameter(self.term_structure)
        return _GeneralizedHullWhiteDynamics(phi, process, self._f, self._f_inverse)

    def generate_arguments(self) -> None:
        # C++ parity: generalizedhullwhite.cpp:122-126 —
        # ``speed_.update(); vol_.update(); phi_ = FittingParameter(...);``
        self._rebuild_interpolations()
        self._phi = _GeneralizedHullWhiteFittingParameter(
            self.term_structure, self.a(), self.sigma()
        )

    def fixed_reversion(self) -> list[bool]:
        """Mask to pass to ``calibrate`` to fit only volatility.

        # C++ parity: generalizedhullwhite.cpp:252-259 — True for the
        # reversion slots, False for the vol slots.
        """
        na = len(self._speed_periods)
        nsigma = len(self._vol_periods)
        return [True] * na + [False] * nsigma


__all__ = ["GeneralizedHullWhite", "InterpolationParameter"]
