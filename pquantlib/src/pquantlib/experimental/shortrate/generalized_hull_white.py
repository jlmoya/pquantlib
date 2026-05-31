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
        # LinearFlat requires >= 1 point; a single point -> constant.
        self._speed_interp: LinearFlatInterpolation = self._build_flat(
            self._speed_periods, list(speed)
        )
        self._vol_interp: LinearFlatInterpolation = self._build_flat(
            self._vol_periods, list(vol)
        )

    @staticmethod
    def _build_flat(xs: list[float], ys: list[float]) -> LinearFlatInterpolation:
        # A single pillar can't form a numpy x-range with > 1 element, but
        # LinearFlat with one node returns ys[0] for every query; duplicate
        # the single node a hair apart so the impl's slope loop is well
        # defined while keeping the flat (constant) behaviour exactly.
        if len(xs) == 1:
            interp = LinearFlat.interpolate(
                np.array([xs[0], xs[0] + 1.0], dtype=np.float64),
                np.array([ys[0], ys[0]], dtype=np.float64),
            )
        else:
            interp = LinearFlat.interpolate(
                np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
            )
        interp.enable_extrapolation()
        return interp

    # --- analytic accessors ---------------------------------------------

    def a(self) -> float:
        # C++ parity: generalizedhullwhite.hpp:132 — a_(0.0).
        return self._speed_interp(0.0, allow_extrapolation=True)

    def sigma(self) -> float:
        # C++ parity: generalizedhullwhite.hpp:133 — sigma_(0.0).
        return self._vol_interp(0.0, allow_extrapolation=True)

    def speed(self, t: float) -> float:
        """Reversion alpha(t)."""
        return self._speed_interp(t, allow_extrapolation=True)

    def vol(self, t: float) -> float:
        """Volatility sigma(t)."""
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
        # C++ parity: generalizedhullwhite.cpp:122-126.
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


__all__ = ["GeneralizedHullWhite"]
