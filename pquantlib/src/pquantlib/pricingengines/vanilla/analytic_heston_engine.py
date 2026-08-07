"""AnalyticHestonEngine — Heston model analytic engine via Fourier transform.

# C++ parity: ql/pricingengines/vanilla/analytichestonengine.{hpp,cpp}
# (v1.43) — ``class AnalyticHestonEngine : public
# GenericModelEngine<HestonModel, VanillaOption::arguments,
# VanillaOption::results>``.

Prices a plain-vanilla European option under the Heston dynamics by inverting
the characteristic function. Two families of complex-log formula, exactly as in
C++:

* ``Gatheral`` / ``BranchCorrection`` — the classic two-integral form::

      Call = spot*dd*(p1 + 0.5) - strike*dr*(p2 + 0.5)
      Put  = spot*dd*(p1 - 0.5) - strike*dr*(p2 - 0.5)

  with ``p_j = (1/pi) * Integral_0^inf Fj(phi) dphi``.

* ``AndersenPiterbarg`` and relatives (``AndersenPiterbargOptCV``,
  ``AsymptoticChF``, ``AngledContour``, ``AngledContourNoCV``, ``OptimalCV``) —
  a single integral of the difference between the Heston chF and a control
  variate whose price is known in closed form::

      Call = (controlVariateValue + h_cv) * dr
      Put  = (controlVariateValue + h_cv - (fwd - strike)) * dr

Quadrature
----------
``Integration`` mirrors ``AnalyticHestonEngine::Integration``: a small tagged
union over QuantLib's Gaussian quadratures and adaptive integrators, with the
same ``integrand1`` / ``integrand2`` / ``integrand3`` substitutions. The
``integrationOrder`` argument is HONOURED — a previous revision of this module
accepted it and silently used ``scipy.integrate.quad`` instead, so
``AnalyticHestonEngine(model, 1)`` and ``AnalyticHestonEngine(model, 144)``
returned the same number while C++ returned 9.1988042562105914 and
8.4559223723439541 respectively.

NOT YET PINNED: that order sweep has no test. An earlier draft of this
docstring cited ``tests/pricingengines/vanilla/test_bates_engines.py``, which
does not exist — the file is ``test_bates_engine.py`` and it has no sweep.
Recorded here rather than quietly dropped, because the sweep is the assertion
that would have caught the ignored-argument defect.

Constructors
------------
C++ has three; Python spells the two non-default ones as classmethods, matching
the repo's ``FlatForward.from_rate`` / ``TimeGrid.with_mandatory`` convention:

* ``AnalyticHestonEngine(model, integration_order=144)``
  -> Gauss-Laguerre of that order, ``OptimalCV``,
  ``andersenPiterbargEpsilon = Null<Real>()``. That last one is not an
  oversight and not a "must be supplied later" marker: Gauss-Laguerre never
  asks the Andersen-Piterbarg limit for an upper bound, so the epsilon is
  computed into a value nothing reads. This is the constructor almost every
  caller uses, including ``BatesEngine``'s siblings and the test suite.
* ``AnalyticHestonEngine.with_lobatto(model, rel_tolerance, max_evaluations)``
  -> adaptive Gauss-Lobatto, ``OptimalCV``, ``andersenPiterbargEpsilon = 1e-40``.
* ``AnalyticHestonEngine.with_integration(model, cpx_log, integration,
  andersen_piterbarg_epsilon=1e-25, alpha=-0.5)`` -> full control.

Carve-outs
----------
``AnalyticHestonEngine::OptimalAlpha`` (the alpha optimiser used only for
extreme strikes) is not ported; ``alpha`` is an explicit argument here, as it is
in C++ when ``OptimalAlpha`` is not consulted.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.integrals.discrete_integrals import (
    DiscreteSimpsonIntegrator,
    DiscreteTrapezoidIntegrator,
)
from pquantlib.math.integrals.exp_sinh_integral import ExpSinhIntegral
from pquantlib.math.integrals.exponential_integrals import ci, si
from pquantlib.math.integrals.gaussian_quadrature import (
    GaussChebyshev2ndIntegration,
    GaussChebyshevIntegration,
    GaussianQuadrature,
    GaussLaguerreIntegration,
    GaussLegendreIntegration,
)
from pquantlib.math.integrals.integrator import Integrator
from pquantlib.math.integrals.kronrod import GaussKronrodAdaptive
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.math.integrals.simpson import SimpsonIntegral
from pquantlib.math.integrals.trapezoid import TrapezoidIntegral
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine

if TYPE_CHECKING:
    from pquantlib.time.date import Date

_MAX_REAL = 1.7976931348623157e308

#: C++ ``Null<Real>()`` == ``std::numeric_limits<float>::max()``
#: (ql/utilities/null.hpp). NOT ``QL_MAX_REAL``: ``Null<Real>`` deliberately
#: picks the *float* maximum so it round-trips through a ``float``. The
#: distinction matters here — the first ``AnalyticHestonEngine`` constructor
#: leaves ``andersenPiterbargEpsilon_`` at ``Null<Real>()``, and
#: ``Integration::calculate`` compares its ``maxBound()`` against the same
#: sentinel.
_NULL_REAL = 3.4028234663852886e38


# ---------------------------------------------------------------------------
# complex expm1 / log1p — ql/math/expm1.{hpp,cpp}
# ---------------------------------------------------------------------------
def _cexpm1(z: complex) -> complex:
    """``exp(z) - 1`` without cancellation for small ``|z|``.

    # C++ parity: ``QuantLib::expm1(const std::complex<Real>&)`` in
    # ql/math/expm1.cpp:26-40 (v1.43).
    """
    if abs(z) < 1.0:
        a, b = z.real, z.imag
        exp_1 = math.expm1(a)
        s = math.sin(0.5 * b)
        cos_1 = -2 * s * s
        return complex(exp_1 * cos_1 + exp_1 + cos_1, math.sin(b) * math.exp(a))
    return cmath.exp(z) - 1.0


def _clog1p(z: complex) -> complex:
    """``log(1 + z)`` without cancellation for small ``|z|``.

    # C++ parity: ``QuantLib::log1p(const std::complex<Real>&)`` in
    # ql/math/expm1.cpp:42-53 (v1.43).
    """
    a, b = z.real, z.imag
    if abs(a) < 0.5 and abs(b) < 0.5:
        return complex(0.5 * math.log1p(a * a + 2 * a + b * b), cmath.phase(1.0 + z))
    return cmath.log(1.0 + z)


class ComplexLogFormula(IntEnum):
    """Which complex-logarithm / control-variate formulation the engine uses.

    # C++ parity: ``enum ComplexLogFormula`` nested in
    # ``AnalyticHestonEngine`` (analytichestonengine.hpp:100-118, v1.43).
    # Exposed at module level and re-exported as ``AnalyticHestonEngine.Gatheral``
    # etc. so call sites read like the C++ ``AnalyticHestonEngine::Gatheral``.
    """

    #: Gatheral form of the characteristic function w/o control variate.
    Gatheral = 0
    #: Old branch-correction form of the characteristic function w/o control variate.
    BranchCorrection = 1
    #: Gatheral form with Andersen-Piterbarg control variate.
    AndersenPiterbarg = 2
    #: As AndersenPiterbarg, with a slightly better control variate.
    AndersenPiterbargOptCV = 3
    #: Gatheral form with an asymptotic expansion of the chF as control variate.
    AsymptoticChF = 4
    #: Angled contour shift integral with control variate.
    AngledContour = 5
    #: Angled contour shift integral w/o control variate.
    AngledContourNoCV = 6
    #: Auto-select the best control variate from the above.
    OptimalCV = 7


class _Algorithm(IntEnum):
    """# C++ parity: ``AnalyticHestonEngine::Integration::Algorithm`` (hpp:241-246)."""

    GaussLobatto = 0
    GaussKronrod = 1
    Simpson = 2
    Trapezoid = 3
    DiscreteTrapezoid = 4
    DiscreteSimpson = 5
    GaussLaguerre = 6
    GaussLegendre = 7
    GaussChebyshev = 8
    GaussChebyshev2nd = 9
    ExpSinh = 10


def _integrand1(c_inf: float, f: Callable[[float], float]) -> Callable[[float], float]:
    """# C++ parity: anonymous-namespace ``integrand1`` (analytichestonengine.cpp:54-66)."""

    def g(x: float) -> float:
        if (1.0 - x) * c_inf > QL_EPSILON:
            return f(-math.log(0.5 - 0.5 * x) / c_inf) / ((1.0 - x) * c_inf)
        return 0.0

    return g


def _integrand2(c_inf: float, f: Callable[[float], float]) -> Callable[[float], float]:
    """# C++ parity: anonymous-namespace ``integrand2`` (analytichestonengine.cpp:68-80)."""

    def g(x: float) -> float:
        if x * c_inf > QL_EPSILON:
            return f(-math.log(x) / c_inf) / (x * c_inf)
        return 0.0

    return g


def _integrand3(c_inf: float, f: Callable[[float], float]) -> Callable[[float], float]:
    """# C++ parity: anonymous-namespace ``integrand3`` (analytichestonengine.cpp:82-90)."""
    inner = _integrand2(c_inf, f)

    def g(x: float) -> float:
        return inner(1.0 - x)

    return g


class Integration:
    """Quadrature strategy for the Heston Fourier integral.

    # C++ parity: ``class AnalyticHestonEngine::Integration``
    # (analytichestonengine.hpp:198-255, .cpp:878-1065, v1.43).

    Constructed only through the named factories, exactly as in C++ where the
    constructors are private. Also reachable as
    ``AnalyticHestonEngine.Integration``.
    """

    __slots__ = ("_gaussian_quadrature", "_int_algo", "_integrator")

    def __init__(
        self,
        int_algo: _Algorithm,
        integrator: Integrator | None = None,
        gaussian_quadrature: GaussianQuadrature | None = None,
    ) -> None:
        """# C++ parity: the two private ctors (analytichestonengine.cpp:878-885)."""
        self._int_algo: _Algorithm = int_algo
        self._integrator: Integrator | None = integrator
        self._gaussian_quadrature: GaussianQuadrature | None = gaussian_quadrature

    # --- non-adaptive factories -------------------------------------------

    @classmethod
    def gauss_laguerre(cls, integration_order: int = 128) -> Integration:
        """# C++ parity: ``Integration::gaussLaguerre`` (.cpp:924-930)."""
        qassert.require(
            integration_order <= 192, "maximum integraton order (192) exceeded"
        )
        return cls(
            _Algorithm.GaussLaguerre,
            gaussian_quadrature=GaussLaguerreIntegration(integration_order),
        )

    @classmethod
    def gauss_legendre(cls, integration_order: int = 128) -> Integration:
        """# C++ parity: ``Integration::gaussLegendre`` (.cpp:932-937)."""
        return cls(
            _Algorithm.GaussLegendre,
            gaussian_quadrature=GaussLegendreIntegration(integration_order),
        )

    @classmethod
    def gauss_chebyshev(cls, integration_order: int = 128) -> Integration:
        """# C++ parity: ``Integration::gaussChebyshev`` (.cpp:939-944)."""
        return cls(
            _Algorithm.GaussChebyshev,
            gaussian_quadrature=GaussChebyshevIntegration(integration_order),
        )

    @classmethod
    def gauss_chebyshev_2nd(cls, integration_order: int = 128) -> Integration:
        """# C++ parity: ``Integration::gaussChebyshev2nd`` (.cpp:946-951)."""
        return cls(
            _Algorithm.GaussChebyshev2nd,
            gaussian_quadrature=GaussChebyshev2ndIntegration(integration_order),
        )

    # --- adaptive factories ------------------------------------------------

    @classmethod
    def gauss_lobatto(
        cls,
        rel_tolerance: float | None,
        abs_tolerance: float | None,
        max_evaluations: int = 1000,
        use_convergence_estimate: bool = False,
    ) -> Integration:
        """# C++ parity: ``Integration::gaussLobatto`` (.cpp:886-895).

        ``None`` stands for C++'s ``Null<Real>()``, which every caller in
        ``analytichestonengine.cpp`` passes as the *absolute* tolerance. It is
        translated to the sentinel's actual value rather than forwarded:
        ``Integrator.__init__`` evaluates ``absolute_accuracy > QL_EPSILON``,
        which raises ``TypeError`` on ``None``, and ``GaussLobattoIntegral``
        then selects ``min(absAccuracy, accuracy * relTolerance)`` — with
        ``Null<Real>()`` in that slot the relative criterion always wins, which
        is the behaviour C++ relies on.
        """
        return cls(
            _Algorithm.GaussLobatto,
            integrator=GaussLobattoIntegral(
                max_evaluations,
                _NULL_REAL if abs_tolerance is None else abs_tolerance,
                rel_tolerance,
                use_convergence_estimate,
            ),
        )

    @classmethod
    def gauss_kronrod(cls, abs_tolerance: float, max_evaluations: int = 1000) -> Integration:
        """# C++ parity: ``Integration::gaussKronrod`` (.cpp:897-904)."""
        return cls(
            _Algorithm.GaussKronrod,
            integrator=GaussKronrodAdaptive(abs_tolerance, max_evaluations),
        )

    @classmethod
    def simpson(cls, abs_tolerance: float, max_evaluations: int = 1000) -> Integration:
        """# C++ parity: ``Integration::simpson`` (.cpp:906-913)."""
        return cls(
            _Algorithm.Simpson,
            integrator=SimpsonIntegral(abs_tolerance, max_evaluations),
        )

    @classmethod
    def trapezoid(cls, abs_tolerance: float, max_evaluations: int = 1000) -> Integration:
        """# C++ parity: ``Integration::trapezoid`` (.cpp:915-922)."""
        return cls(
            _Algorithm.Trapezoid,
            integrator=TrapezoidIntegral(abs_tolerance, max_evaluations),
        )

    @classmethod
    def discrete_simpson(cls, evaluations: int = 1000) -> Integration:
        """# C++ parity: ``Integration::discreteSimpson`` (.cpp:953-958)."""
        return cls(
            _Algorithm.DiscreteSimpson,
            integrator=DiscreteSimpsonIntegrator(evaluations),
        )

    @classmethod
    def discrete_trapezoid(cls, evaluations: int = 1000) -> Integration:
        """# C++ parity: ``Integration::discreteTrapezoid`` (.cpp:960-965)."""
        return cls(
            _Algorithm.DiscreteTrapezoid,
            integrator=DiscreteTrapezoidIntegrator(evaluations),
        )

    @classmethod
    def exp_sinh(cls, rel_tolerance: float = 1e-8) -> Integration:
        """# C++ parity: ``Integration::expSinh`` (.cpp:967-972)."""
        return cls(_Algorithm.ExpSinh, integrator=ExpSinhIntegral(rel_tolerance))

    # --- inspectors ---------------------------------------------------------

    def number_of_evaluations(self) -> int:
        """# C++ parity: ``Integration::numberOfEvaluations`` (.cpp:974-982)."""
        if self._integrator is not None:
            return self._integrator.number_of_evaluations()
        if self._gaussian_quadrature is not None:
            return self._gaussian_quadrature.order()
        raise LibraryException("neither Integrator nor GaussianQuadrature given")

    def is_adaptive_integration(self) -> bool:
        """# C++ parity: ``Integration::isAdaptiveIntegration`` (.cpp:984-990)."""
        return self._int_algo in (
            _Algorithm.GaussLobatto,
            _Algorithm.GaussKronrod,
            _Algorithm.Simpson,
            _Algorithm.Trapezoid,
            _Algorithm.ExpSinh,
        )

    # --- the integral itself -------------------------------------------------

    def calculate(
        self,
        c_inf: float,
        f: Callable[[float], float],
        max_bound: Callable[[], float | None] | float | None = None,
        scaling: float = 1.0,
    ) -> float:
        """Integrate ``f`` over ``(0, inf)``.

        # C++ parity: ``Integration::calculate`` (.cpp:992-1044). ``max_bound``
        # may be a callable (C++'s ``std::function<Real()>``), a plain float
        # (the second overload) or ``None`` (C++'s empty ``std::function``);
        # it is consulted ONLY by the adaptive and discrete algorithms —
        # Gauss-Laguerre ignores it entirely.
        #
        # The callable is invoked LAZILY, exactly where C++ invokes it: inside
        # ``if (maxBound && maxBound() != Null<Real>())``, i.e. only on the
        # branches that can use a finite bound. Evaluating it eagerly is not a
        # harmless reordering — for the control-variate formulas ``maxBound``
        # is ``andersenPiterbargIntegrationLimit``, which Brent-solves
        # ``-log(epsilon)/c_inf`` and throws outright when
        # ``andersenPiterbargEpsilon`` is left at its ``Null<Real>()`` default.
        # C++ reaches Gauss-Laguerre without ever calling it, so that default
        # is perfectly usable there; a port that calls it first turns the
        # library's default-constructed engine into an exception.
        """

        def bound() -> float | None:
            # C++ ``maxBound && maxBound() != Null<Real>()``.
            value = max_bound() if callable(max_bound) else max_bound
            return None if value is None or value == _NULL_REAL else value

        if self._int_algo == _Algorithm.GaussLaguerre:
            assert self._gaussian_quadrature is not None
            return self._gaussian_quadrature(f)
        if self._int_algo in (
            _Algorithm.GaussLegendre,
            _Algorithm.GaussChebyshev,
            _Algorithm.GaussChebyshev2nd,
        ):
            assert self._gaussian_quadrature is not None
            return self._gaussian_quadrature(_integrand1(c_inf, f))
        if self._int_algo == _Algorithm.ExpSinh:
            assert self._integrator is not None
            return scaling * self._integrator(lambda x: f(scaling * x), 0.0, _MAX_REAL)
        if self._int_algo in (
            _Algorithm.Simpson,
            _Algorithm.Trapezoid,
            _Algorithm.GaussLobatto,
            _Algorithm.GaussKronrod,
        ):
            assert self._integrator is not None
            upper = bound()
            if upper is not None:
                return self._integrator(f, 0.0, upper)
            return self._integrator(_integrand2(c_inf, f), 0.0, 1.0)
        if self._int_algo in (_Algorithm.DiscreteTrapezoid, _Algorithm.DiscreteSimpson):
            assert self._integrator is not None
            upper = bound()
            if upper is not None:
                return self._integrator(f, 0.0, upper)
            return self._integrator(_integrand3(c_inf, f), 0.0, 1.0)
        raise LibraryException(f"unknwon integration algorithm: {self._int_algo}")

    @staticmethod
    def andersen_piterbarg_integration_limit(
        c_inf: float, epsilon: float, v0: float, t: float
    ) -> float:
        """Upper integration bound for the Andersen-Piterbarg forms.

        # C++ parity: ``Integration::andersenPiterbargIntegrationLimit``
        # (.cpp:1046-1065). Two Brent solves; the second is allowed to fail and
        # its failure falls back to the first root, exactly as in C++.
        """
        log_epsilon = math.log(epsilon)

        def u_max(u: float) -> float:
            # C++ parity: anonymous-namespace ``u_Max`` (.cpp:92-108).
            return c_inf * u + math.log(u) + log_epsilon

        u_max_guess = -math.log(epsilon) / c_inf
        u_max_step = 0.1 * u_max_guess
        root = Brent().solve(u_max, QL_EPSILON * u_max_guess, u_max_guess, u_max_step)

        v0t2 = 0.5 * v0 * t

        def u_hat_max(u: float) -> float:
            # C++ parity: anonymous-namespace ``uHat_Max`` (.cpp:111-127).
            return v0t2 * u * u + math.log(u) + log_epsilon

        try:
            u_hat_guess = math.sqrt(-math.log(epsilon) / v0t2)
            u_hat = Brent().solve(
                u_hat_max, QL_EPSILON * u_hat_guess, u_hat_guess, 0.001 * u_hat_guess
            )
        except (LibraryException, ValueError, ZeroDivisionError, OverflowError):
            return root
        return max(root, u_hat)


class AP_Helper:  # noqa: N801  (C++ class name, character for character)
    """Integrand for the Andersen-Piterbarg / angled-contour control variates.

    # C++ parity: ``class AnalyticHestonEngine::AP_Helper``
    # (analytichestonengine.hpp:257-275, .cpp:448-576, v1.43). Also reachable as
    # ``AnalyticHestonEngine.AP_Helper``.

    ``ExponentialFittingHestonEngine`` uses this class directly with its own
    fixed 64-node quadrature, which is why it is public in C++ and public here.
    """

    __slots__ = (
        "_alpha",
        "_cpx_log",
        "_engine",
        "_freq",
        "_fwd",
        "_phi",
        "_psi",
        "_s_alpha",
        "_strike",
        "_tan_phi",
        "_term",
        "_v_avg",
    )

    def __init__(
        self,
        term: float,
        fwd: float,
        strike: float,
        cpx_log: ComplexLogFormula,
        engine: AnalyticHestonEngine,
        alpha: float = -0.5,
    ) -> None:
        """# C++ parity: ``AP_Helper::AP_Helper`` (.cpp:448-505).

        C++ asserts ``enginePtr != nullptr``; the Python signature types
        ``engine`` as non-optional, so the check is the type annotation.
        """
        self._term: float = term
        self._fwd: float = fwd
        self._strike: float = strike
        self._freq: float = math.log(fwd / strike)
        self._cpx_log: ComplexLogFormula = cpx_log
        self._engine: AnalyticHestonEngine = engine
        self._alpha: float = alpha
        self._s_alpha: float = math.exp(alpha * self._freq)

        model = engine.model()
        v0 = model.v0()
        kappa = model.kappa()
        theta = model.theta()
        sigma = model.sigma()
        rho = model.rho()

        self._v_avg: float = 0.0
        self._tan_phi: float = 0.0
        self._phi: complex = 0j
        self._psi: complex = 0j

        if cpx_log == ComplexLogFormula.AndersenPiterbarg:
            self._v_avg = (1 - math.exp(-kappa * term)) * (v0 - theta) / (kappa * term) + theta
            return
        if cpx_log == ComplexLogFormula.AndersenPiterbargOptCV:
            self._v_avg = -8.0 * math.log(engine.ch_f(complex(0, alpha), term).real) / term
            return
        if cpx_log not in (
            ComplexLogFormula.AsymptoticChF,
            ComplexLogFormula.AngledContour,
            ComplexLogFormula.AngledContourNoCV,
        ):
            raise LibraryException(f"unknown control variate: {cpx_log}")

        # C++ parity: .cpp:478-501 — AsymptoticChF falls through into
        # AngledContour, which falls through into AngledContourNoCV, so the
        # later assignments run for the earlier cases too.
        if cpx_log == ComplexLogFormula.AsymptoticChF:
            rho_bar = math.sqrt(1 - rho * rho)
            self._phi = -(v0 + term * kappa * theta) / sigma * complex(rho_bar, rho)
            self._psi = (
                complex(
                    (kappa - 0.5 * rho * sigma) * (v0 + term * kappa * theta)
                    + kappa * theta * math.log(4 * (1 - rho * rho)),
                    -(
                        (0.5 * rho * rho * sigma - kappa * rho)
                        / rho_bar
                        * (v0 + kappa * theta * term)
                        - 2 * kappa * theta * math.atan(rho / rho_bar)
                    ),
                )
                / (sigma * sigma)
            )
        if cpx_log in (ComplexLogFormula.AsymptoticChF, ComplexLogFormula.AngledContour):
            self._v_avg = (1 - math.exp(-kappa * term)) * (v0 - theta) / (kappa * term) + theta

        r = rho - sigma * self._freq / (v0 + kappa * theta * term)
        # C++ parity: .cpp:497-499 — boost::math::sign(freq) is -1/0/+1.
        sign_freq = (self._freq > 0.0) - (self._freq < 0.0)
        self._tan_phi = math.tan((math.pi / 12 * sign_freq) if r * self._freq < 0.0 else 0.0)

    def __call__(self, u: float) -> float:
        """# C++ parity: ``AP_Helper::operator()`` (.cpp:507-548)."""
        engine = self._engine
        qassert.require(
            engine.add_on_term(u, self._term, 1) == 0
            and engine.add_on_term(u, self._term, 2) == 0,
            "only Heston model is supported",
        )
        i = 1j
        cpx_log = self._cpx_log

        if cpx_log in (
            ComplexLogFormula.AngledContour,
            ComplexLogFormula.AngledContourNoCV,
            ComplexLogFormula.AsymptoticChF,
        ):
            h_u = complex(u, u * self._tan_phi - self._alpha)
            h_prime = h_u - i

            phi_bs = 0j
            if cpx_log == ComplexLogFormula.AngledContour:
                phi_bs = cmath.exp(
                    -0.5
                    * self._v_avg
                    * self._term
                    * (h_prime * h_prime + complex(-h_prime.imag, h_prime.real))
                )
            elif cpx_log == ComplexLogFormula.AsymptoticChF:
                phi_bs = cmath.exp(u * complex(1, self._tan_phi) * self._phi + self._psi)

            return (
                math.exp(-u * self._tan_phi * self._freq)
                * (
                    cmath.exp(complex(0.0, u * self._freq))
                    * complex(1, self._tan_phi)
                    * (phi_bs - engine.ch_f(h_prime, self._term))
                    / (h_u * h_prime)
                ).real
                * self._s_alpha
            )

        if cpx_log in (
            ComplexLogFormula.AndersenPiterbarg,
            ComplexLogFormula.AndersenPiterbargOptCV,
        ):
            z = complex(u, -self._alpha)
            z_prime = complex(u, -self._alpha - 1)
            phi_bs = cmath.exp(
                -0.5
                * self._v_avg
                * self._term
                * (z_prime * z_prime + complex(-z_prime.imag, z_prime.real))
            )
            return (
                cmath.exp(complex(0.0, u * self._freq))
                * (phi_bs - engine.ch_f(z_prime, self._term))
                / (z * z_prime)
            ).real * self._s_alpha

        raise LibraryException(f"unknown control variate: {cpx_log}")

    def control_variate_value(self) -> float:
        """Closed-form price of the control variate.

        # C++ parity: ``AP_Helper::controlVariateValue`` (.cpp:550-576).
        """
        cpx_log = self._cpx_log
        if cpx_log in (
            ComplexLogFormula.AngledContour,
            ComplexLogFormula.AndersenPiterbarg,
            ComplexLogFormula.AndersenPiterbargOptCV,
        ):
            return BlackCalculator.from_type_strike(
                OptionType.Call,
                self._strike,
                self._fwd,
                math.sqrt(self._v_avg * self._term),
            ).value()

        if cpx_log == ComplexLogFormula.AsymptoticChF:
            qassert.require(self._alpha == -0.5, "alpha must be equal to -0.5")
            phi_freq = complex(self._phi.real, self._phi.imag + self._freq)
            return (
                self._fwd
                - math.sqrt(self._strike * self._fwd)
                / math.pi
                * (
                    cmath.exp(self._psi)
                    * (
                        -2.0 * ci(-0.5 * phi_freq) * cmath.sin(0.5 * phi_freq)
                        + cmath.cos(0.5 * phi_freq) * (math.pi + 2.0 * si(0.5 * phi_freq))
                    )
                ).real
            )

        if cpx_log == ComplexLogFormula.AngledContourNoCV:
            alpha = self._alpha
            return (
                (self._fwd if alpha <= 0.0 else 0.0)
                - (self._strike if alpha <= -1.0 else 0.0)
                - 0.5 * (self._fwd if alpha == 0.0 else 0.0)
                + 0.5 * (self._strike if alpha == -1.0 else 0.0)
            )

        raise LibraryException(f"unknown control variate: {cpx_log}")


class AnalyticHestonEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Analytic Heston engine via Fourier transform.

    # C++ parity: ``class AnalyticHestonEngine`` in
    # ql/pricingengines/vanilla/analytichestonengine.hpp:91-195 (v1.43).
    """

    # C++ spelling: AnalyticHestonEngine::Integration / ::AP_Helper /
    # ::Gatheral, ::OptimalCV, ...
    Integration = Integration
    AP_Helper = AP_Helper
    Gatheral = ComplexLogFormula.Gatheral
    BranchCorrection = ComplexLogFormula.BranchCorrection
    AndersenPiterbarg = ComplexLogFormula.AndersenPiterbarg
    AndersenPiterbargOptCV = ComplexLogFormula.AndersenPiterbargOptCV
    AsymptoticChF = ComplexLogFormula.AsymptoticChF
    AngledContour = ComplexLogFormula.AngledContour
    AngledContourNoCV = ComplexLogFormula.AngledContourNoCV
    OptimalCV = ComplexLogFormula.OptimalCV

    def __init__(
        self,
        model: HestonModel,
        integration_order: int = 144,
    ) -> None:
        """Construct with Gauss-Laguerre quadrature of ``integration_order``.

        # C++ parity: ``AnalyticHestonEngine(model, integrationOrder = 144)``
        # (analytichestonengine.cpp:659-671) — cpxLog = OptimalCV,
        # andersenPiterbargEpsilon = Null<Real>(), alpha = -0.5.
        """
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._model: HestonModel = model
        self._integration_order: int = integration_order
        self._evaluations: int = 0
        self._cpx_log: ComplexLogFormula = ComplexLogFormula.OptimalCV
        self._integration: Integration = Integration.gauss_laguerre(integration_order)
        # C++ leaves this at ``Null<Real>()`` here, and that is a usable value,
        # not a "not set" marker: the paired Gauss-Laguerre integration never
        # asks for an upper bound, so the epsilon is computed and discarded.
        self._andersen_piterbarg_epsilon: float = _NULL_REAL
        self._alpha: float = -0.5
        model.register_with(self)

    @classmethod
    def with_lobatto(
        cls, model: HestonModel, rel_tolerance: float, max_evaluations: int
    ) -> AnalyticHestonEngine:
        """Adaptive Gauss-Lobatto quadrature.

        # C++ parity: ``AnalyticHestonEngine(model, relTolerance,
        # maxEvaluations)`` (.cpp:673-685) — cpxLog = OptimalCV,
        # andersenPiterbargEpsilon = 1e-40.
        """
        engine = cls(model)
        engine._integration = Integration.gauss_lobatto(rel_tolerance, None, max_evaluations)
        engine._andersen_piterbarg_epsilon = 1e-40
        return engine

    @classmethod
    def with_integration(
        cls,
        model: HestonModel,
        cpx_log: ComplexLogFormula,
        integration: Integration,
        andersen_piterbarg_epsilon: float = 1e-25,
        alpha: float = -0.5,
    ) -> AnalyticHestonEngine:
        """Full control over the Fourier integration algorithm.

        # C++ parity: ``AnalyticHestonEngine(model, cpxLog, itg,
        # andersenPiterbargEpsilon = 1e-25, alpha = -0.5)`` (.cpp:687-705).
        """
        qassert.require(
            cpx_log != ComplexLogFormula.BranchCorrection
            or not integration.is_adaptive_integration(),
            "Branch correction does not work in conjunction with adaptive integration methods",
        )
        engine = cls(model)
        engine._cpx_log = cpx_log
        engine._integration = integration
        engine._andersen_piterbarg_epsilon = andersen_piterbarg_epsilon
        engine._alpha = alpha
        return engine

    # --- inspectors -----------------------------------------------------

    def model(self) -> HestonModel:
        """The underlying HestonModel."""
        return self._model

    def number_of_evaluations(self) -> int:
        """# C++ parity: ``AnalyticHestonEngine::numberOfEvaluations`` (.cpp:721-723)."""
        return self._evaluations

    @staticmethod
    def optimal_control_variate(
        t: float, v0: float, kappa: float, theta: float, sigma: float, rho: float
    ) -> ComplexLogFormula:
        """Pick the better control variate for these parameters.

        # C++ parity: ``AnalyticHestonEngine::optimalControlVariate``
        # (.cpp:707-719) — three conditions, ALL of which must hold for
        # ``AsymptoticChF``; otherwise ``AngledContour``.
        """
        if (
            t > 0.15
            and (v0 + t * kappa * theta) / sigma * math.sqrt(1 - rho * rho) < 0.15
            and (
                (kappa - 0.5 * rho * sigma) * (v0 + t * kappa * theta)
                + kappa * theta * math.log(4 * (1 - rho * rho))
            )
            / (sigma * sigma)
            < 0.1
        ):
            return ComplexLogFormula.AsymptoticChF
        return ComplexLogFormula.AngledContour

    # --- characteristic function ------------------------------------------

    def ch_f(self, z: complex, t: float) -> complex:
        """Normalized characteristic function.

        # C++ parity: ``AnalyticHestonEngine::chF`` (.cpp:578-619). For a small
        # ``sigma`` with a non-tiny ``kappa`` the exponential form is replaced
        # by a second-order expansion in ``sigma`` — a port that always calls
        # ``exp(lnChF(...))`` silently loses that branch.
        """
        model = self._model
        if model.sigma() > 1e-6 or model.kappa() < 1e-8:
            return cmath.exp(self.ln_ch_f(z, t))

        kappa = model.kappa()
        sigma = model.sigma()
        theta = model.theta()
        rho = model.rho()
        v0 = model.v0()
        sigma2 = sigma * sigma

        kt = kappa * t
        ekt = math.exp(kt)
        e2kt = math.exp(2 * kt)
        rho2 = rho * rho
        zpi = z + 1j

        return (
            cmath.exp(-(((theta - v0 + ekt * ((-1 + kt) * theta + v0)) * z * zpi) / ekt) / (2.0 * kappa))
            + (
                cmath.exp(
                    -kt
                    - ((theta - v0 + ekt * ((-1 + kt) * theta + v0)) * z * zpi) / (2.0 * ekt * kappa)
                )
                * rho
                * (2 * theta + kt * theta - v0 - kt * v0 + ekt * ((-2 + kt) * theta + v0))
                * (1.0 - complex(-z.imag, z.real))
                * z
                * z
            )
            / (2.0 * kappa * kappa)
            * sigma
            + (
                cmath.exp(
                    -2 * kt
                    - ((theta - v0 + ekt * ((-1 + kt) * theta + v0)) * z * zpi) / (2.0 * ekt * kappa)
                )
                * z
                * z
                * zpi
                * (
                    -2
                    * rho2
                    * (2 * theta + kt * theta - v0 - kt * v0 + ekt * ((-2 + kt) * theta + v0)) ** 2
                    * z
                    * z
                    * zpi
                    + 2
                    * kappa
                    * v0
                    * (
                        -zpi
                        + e2kt * (zpi + 4 * rho2 * z)
                        - 2 * ekt * (2 * rho2 * z + kt * (zpi + rho2 * (2 + kt) * z))
                    )
                    + kappa
                    * theta
                    * (
                        zpi
                        + e2kt * (-5.0 * zpi - 24 * rho2 * z + 2 * kt * (zpi + 4 * rho2 * z))
                        + 4 * ekt * (zpi + 6 * rho2 * z + kt * (zpi + rho2 * (4 + kt) * z))
                    )
                )
            )
            / (16.0 * (kappa * kappa) ** 2)
            * sigma2
        )

    def ln_ch_f(self, z: complex, t: float) -> complex:
        """Log of the normalized characteristic function.

        # C++ parity: ``AnalyticHestonEngine::lnChF`` (.cpp:621-657) — the
        # Andersen-Lake cancellation-avoiding rewrite of ``g - D``, plus the
        # complex ``expm1`` / ``log1p`` from ql/math/expm1.hpp.
        """
        model = self._model
        kappa = model.kappa()
        sigma = model.sigma()
        theta = model.theta()
        rho = model.rho()
        v0 = model.v0()
        sigma2 = sigma * sigma

        g = kappa + rho * sigma * complex(z.imag, -z.real)
        d = cmath.sqrt(g * g + (z * z + complex(-z.imag, z.real)) * sigma2)

        # C++ parity: .cpp:638-642 — reduce cancellation, see Andersen & Lake.
        r = g - d
        if g.real * d.real + g.imag * d.imag > 0.0:
            r = -sigma2 * z * complex(z.real, z.imag + 1) / (g + d)

        y = _cexpm1(-d * t) / (2.0 * d) if (d.real != 0.0 or d.imag != 0.0) else complex(-0.5 * t, 0.0)

        a = kappa * theta / sigma2 * (r * t - 2.0 * _clog1p(-r * y))
        b = z * complex(z.real, z.imag + 1) * y / (1.0 - r * y)
        return a + v0 * b

    # --- Fj integrand -------------------------------------------------------

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Sub-engine hook: extra term in the log characteristic function.

        # C++ parity: ``AnalyticHestonEngine::addOnTerm``
        # (analytichestonengine.hpp:307-310) — inline default returning
        # ``complex<Real>(0, 0)``. ``BatesEngine`` and friends override it.
        """
        del phi, t, j  # default is parameter-independent
        return 0 + 0j

    def _fj(
        self,
        phi: float,
        *,
        j: int,
        term: float,
        spot: float,
        strike: float,
        ratio: float,
    ) -> float:
        """Gatheral-form integrand for ``j = 1`` or ``j = 2``.

        # C++ parity: ``AnalyticHestonEngine::Fj_Helper::operator()(phi)``
        # (.cpp:181-243, Gatheral branch). Returns a real integrand on
        # ``(0, inf)``.
        """
        model = self._model
        kappa = model.kappa()
        theta = model.theta()
        sigma = model.sigma()
        v0 = model.v0()
        rho = model.rho()

        x = math.log(spot)
        sx = math.log(strike)
        dd = x - math.log(ratio)
        sigma2 = sigma * sigma
        rsigma = rho * sigma
        t0 = kappa - (rsigma if j == 1 else 0.0)

        rpsig = rsigma * phi
        t1 = complex(t0, -rpsig)
        sgn = 1.0 if j == 1 else -1.0
        d = cmath.sqrt(t1 * t1 - sigma2 * phi * complex(-phi, sgn))
        ex = cmath.exp(-d * term)

        add_on = self.add_on_term(phi, term, j)

        if phi != 0.0:
            if sigma > 1e-5:
                p = (t1 - d) / (t1 + d)
                g = cmath.log((1.0 - p * ex) / (1.0 - p))
                arg = (
                    v0 * (t1 - d) * (1.0 - ex) / (sigma2 * (1.0 - ex * p))
                    + (kappa * theta) / sigma2 * ((t1 - d) * term - 2.0 * g)
                    + complex(0.0, phi * (dd - sx))
                    + add_on
                )
                return cmath.exp(arg).imag / phi
            # C++ parity: .cpp:207-219 — sigma ~ 0, L'Hospital expansion.
            td = phi / (2.0 * t1) * complex(-phi, sgn)
            p = td * sigma2 / (t1 + d)
            g = p * (1.0 - ex)
            arg = (
                v0 * td * (1.0 - ex) / (1.0 - p * ex)
                + (kappa * theta) * (td * term - 2.0 * g / sigma2)
                + complex(0.0, phi * (dd - sx))
                + add_on
            )
            return cmath.exp(arg).imag / phi

        # C++ parity: .cpp:221-242 — l'Hospital limit at phi == 0.
        if j == 1:
            kmr = rsigma - kappa
            if abs(kmr) > 1e-7:
                return (
                    dd
                    - sx
                    + (math.exp(kmr * term) * kappa * theta - kappa * theta * (kmr * term + 1.0))
                    / (2 * kmr * kmr)
                    - v0 * (1.0 - math.exp(kmr * term)) / (2.0 * kmr)
                )
            # kappa == rho*sigma — series expansion.
            return dd - sx + 0.25 * kappa * theta * term * term + 0.5 * v0 * term
        return (
            dd
            - sx
            - (math.exp(-kappa * term) * kappa * theta + kappa * theta * (kappa * term - 1.0))
            / (2 * kappa * kappa)
            - v0 * (1.0 - math.exp(-kappa * term)) / (2 * kappa)
        )

    # --- pricing ---------------------------------------------------------

    def price_vanilla_payoff(
        self,
        *,
        payoff: PlainVanillaPayoff,
        maturity: float,
        fwd: float | None = None,
    ) -> float:
        """Closed-form price of a European vanilla under Heston.

        # C++ parity: ``AnalyticHestonEngine::priceVanillaPayoff`` (.cpp:725-859)
        # — a PUBLIC member there, and public here: ``HestonBlackVolSurface``
        # prices through it without going near the arguments/results protocol.
        """
        model = self._model
        process = model.process()
        spot = process.s0().value()
        qassert.require(spot > 0.0, f"negative or null underlying given: {spot}")

        dr = process.risk_free_rate().discount(maturity)
        if fwd is None:
            fwd = spot * process.dividend_yield().discount(maturity) / dr

        strike = payoff.strike()
        # C++ parity: .cpp:761-762 — df is the spot/forward ratio, dd the
        # dividend discount recovered from it.
        df = spot / fwd
        dd = dr / df

        kappa = model.kappa()
        sigma = model.sigma()
        theta = model.theta()
        rho = model.rho()
        v0 = model.v0()

        self._evaluations = 0

        if self._cpx_log in (ComplexLogFormula.Gatheral, ComplexLogFormula.BranchCorrection):
            c_inf = min(0.2, max(0.0001, math.sqrt(1.0 - rho * rho) / sigma)) * (
                v0 + kappa * theta * maturity
            )

            def make(j: int) -> Callable[[float], float]:
                def g(phi: float) -> float:
                    return self._fj(
                        phi, j=j, term=maturity, spot=spot, strike=strike, ratio=df
                    )

                return g

            p1 = self._integration.calculate(c_inf, make(1)) / math.pi
            self._evaluations += self._integration.number_of_evaluations()
            p2 = self._integration.calculate(c_inf, make(2)) / math.pi
            self._evaluations += self._integration.number_of_evaluations()

            if payoff.option_type() == OptionType.Call:
                return spot * dd * (p1 + 0.5) - strike * dr * (p2 + 0.5)
            if payoff.option_type() == OptionType.Put:
                return spot * dd * (p1 - 0.5) - strike * dr * (p2 - 0.5)
            raise LibraryException(f"unknown option type: {payoff.option_type()}")

        # C++ parity: .cpp:803-853 — the control-variate family.
        c_inf = math.sqrt(1.0 - rho * rho) * (v0 + kappa * theta * maturity) / sigma
        epsilon = self._andersen_piterbarg_epsilon * math.pi / (math.sqrt(strike * fwd) * dr)

        def u_m() -> float:
            return Integration.andersen_piterbarg_integration_limit(
                c_inf, epsilon, v0, maturity
            )

        final_log = (
            self.optimal_control_variate(maturity, v0, kappa, theta, sigma, rho)
            if self._cpx_log == ComplexLogFormula.OptimalCV
            else self._cpx_log
        )

        cv_helper = AP_Helper(maturity, fwd, strike, final_log, self, self._alpha)
        cv_value = cv_helper.control_variate_value()

        v_avg = (1 - math.exp(-kappa * maturity)) * (v0 - theta) / (kappa * maturity) + theta
        # C++ parity: .cpp:832-834 — note the test is on the ENGINE's cpxLog_,
        # not on the resolved `final_log`.
        scaling_factor = (
            max(0.25, min(1000.0, 0.25 / math.sqrt(0.5 * v_avg * maturity)))
            if self._cpx_log not in (ComplexLogFormula.OptimalCV, ComplexLogFormula.AsymptoticChF)
            else 1.0
        )

        h_cv = (
            fwd
            / math.pi
            * self._integration.calculate(c_inf, cv_helper, u_m, scaling_factor)
        )
        self._evaluations += self._integration.number_of_evaluations()

        if payoff.option_type() == OptionType.Call:
            return (cv_value + h_cv) * dr
        if payoff.option_type() == OptionType.Put:
            return (cv_value + h_cv - (fwd - strike)) * dr
        raise LibraryException(f"unknown option type: {payoff.option_type()}")

    def price_vanilla_payoff_at_date(
        self, payoff: PlainVanillaPayoff, maturity: Date
    ) -> float:
        """Price at an exercise ``Date`` rather than a year fraction.

        # C++ parity: the ``(payoff, const Date&)`` overload of
        # ``priceVanillaPayoff`` (.cpp:725-735).
        """
        process = self._model.process()
        fwd = (
            process.s0().value()
            * process.dividend_yield().discount(maturity)
            / process.risk_free_rate().discount(maturity)
        )
        return self.price_vanilla_payoff(
            payoff=payoff, maturity=process.time(maturity), fwd=fwd
        )

    def calculate(self) -> None:
        """Fill ``results.value`` for a European vanilla.

        # C++ parity: ``AnalyticHestonEngine::calculate`` (.cpp:861-875).
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not a European option",
        )
        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff),
            "non plain vanilla payoff given",
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        results.reset()
        results.value = self.price_vanilla_payoff_at_date(payoff, args.exercise.last_date())

    def update(self) -> None:
        """Observer.update — model parameters or curves changed."""
        self.notify_observers()


__all__ = ["AP_Helper", "AnalyticHestonEngine", "ComplexLogFormula", "Integration"]
