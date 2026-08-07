"""High-performance fixed-point American engine (Andersen-Lake-Offengenden).

# C++ parity: ql/pricingengines/vanilla/qdfpamericanengine.{hpp,cpp} (v1.43)
#             -- ``QdFpIterationScheme``, ``QdFpLegendreScheme``,
#             ``QdFpTanhSinhIterationScheme``, ``QdFpLegendreTanhSinhScheme``,
#             ``QdFpAmericanEngine`` and the fixed-point equations
#             ``DqFpEquation``, ``DqFpEquation_A``, ``DqFpEquation_B``.

The engine takes the QD+ exercise boundary
(:meth:`~pquantlib.pricingengines.vanilla.qd_plus_american_engine.QdPlusAmericanEngine.get_put_exercise_boundary`)
as a starting guess and refines it with one partial Jacobi-Newton step followed
by ``m - 1`` naive Richardson fixed-point steps, then integrates the refined
boundary into a price with
:class:`~pquantlib.pricingengines.vanilla.qd_plus_american_engine.QdPlusAddOnValue`.

References:
    Leif Andersen, Mark Lake and Dimitri Offengenden (2015), "High Performance
    American Option Pricing", https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2547027

    Leif Andersen, Mark Lake (2021), "Fast American Option Pricing: The
    Double-Boundary Case", https://onlinelibrary.wiley.com/doi/abs/10.1002/wilm.10969

Three knobs, all of them load-bearing
-------------------------------------
* **the iteration scheme** decides the number of Chebyshev nodes, the number of
  fixed-point steps, and the two quadrature rules (one inside every fixed-point
  step, one for the final boundary-to-price conversion). ``QdFpLegendreScheme``
  uses Gauss-Legendre for both; ``QdFpTanhSinhIterationScheme`` uses one
  tanh-sinh rule for both; ``QdFpLegendreTanhSinhScheme`` is Legendre inside and
  tanh-sinh outside. Note the C++ spelling of the accessors --
  ``getNumberOfNaiveFixedPointSteps`` / ``getNumberOfJacobiNewtonFixedPointSteps``,
  returning ``m - 1`` and ``1`` respectively.
* **the fixed-point equation**, ``FP_A`` or ``FP_B``. These are genuinely
  different integral equations for the same boundary and give different
  (both valid) approximations; ``Auto`` picks ``FP_A`` when ``|r - q| < 0.001``
  and ``FP_B`` otherwise.
* **whether the fixed-point integrator is a Gauss-Legendre rule.** ``DqFpEquation``
  reaches *into* the integrator: if it is a ``GaussLegendreIntegrator`` it pulls
  the nodes and weights out and sums them by hand (accumulating from the last
  index down); otherwise it calls the integrator with a closure that carries
  extra endpoint guards. The two paths are not numerically identical, so which
  one is taken is part of the contract.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import IntEnum
from typing import Final

import numpy as np

from pquantlib import qassert
from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.integrals.gaussian_quadrature import GaussLegendreIntegrator
from pquantlib.math.integrals.integrator import Integrator
from pquantlib.math.integrals.tanh_sinh_integral import TanhSinhIntegral
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.vanilla.qd_plus_american_engine import (
    M_1_SQRTPI,
    M_SQRT2,
    M_SQRT_2,
    M_SQRTPI,
    QdPlusAddOnValue,
    QdPlusAmericanEngine,
    QdPutCallParityEngine,
    ieee_div,
    ieee_exp,
    ieee_log,
    squared,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)

# C++ ``squared(QL_EPSILON)`` -- the threshold below which both fixed-point
# equations switch to their closed-form ``tau -> 0`` limits.
_TAU_FLOOR: Final[float] = QL_EPSILON * QL_EPSILON


class QdFpIterationScheme(ABC):
    """Iteration scheme for the fixed-point QD American engine.

    # C++ parity: ``QdFpIterationScheme`` (qdfpamericanengine.hpp:33-43).
    """

    @abstractmethod
    def get_number_of_chebyshev_interpolation_nodes(self) -> int: ...

    @abstractmethod
    def get_number_of_naive_fixed_point_steps(self) -> int: ...

    @abstractmethod
    def get_number_of_jacobi_newton_fixed_point_steps(self) -> int: ...

    @abstractmethod
    def get_fixed_point_integrator(self) -> Integrator: ...

    @abstractmethod
    def get_exercise_boundary_to_price_integrator(self) -> Integrator: ...


class QdFpLegendreScheme(QdFpIterationScheme):
    """Gauss-Legendre ``(l, m, n)-p`` scheme.

    # C++ parity: ``QdFpLegendreScheme`` (hpp:53-68, cpp:37-68).

    Args:
        l: order of the Gauss-Legendre rule used inside every fixed-point step.
        m: fixed-point steps; the first is a partial Jacobi-Newton, the rest are
            naive Richardson iterations (so ``get_number_of_naive_fixed_point_steps``
            reports ``m - 1``).
        n: number of Chebyshev nodes interpolating the exercise boundary.
        p: order of the Gauss-Legendre rule converting the boundary into a price.
    """

    def __init__(self, l: int, m: int, n: int, p: int) -> None:  # noqa: E741 -- C++ parameter names
        self._m: int = m
        self._n: int = n
        self._fp_integrator: Integrator = GaussLegendreIntegrator(l)
        self._exercise_boundary_integrator: Integrator = GaussLegendreIntegrator(p)
        qassert.require(m > 0, "at least one fixed point iteration step is needed")
        qassert.require(n > 0, "at least one interpolation point is needed")

    def get_number_of_chebyshev_interpolation_nodes(self) -> int:
        return self._n

    def get_number_of_naive_fixed_point_steps(self) -> int:
        return self._m - 1

    def get_number_of_jacobi_newton_fixed_point_steps(self) -> int:
        return 1

    def get_fixed_point_integrator(self) -> Integrator:
        return self._fp_integrator

    def get_exercise_boundary_to_price_integrator(self) -> Integrator:
        return self._exercise_boundary_integrator


class QdFpTanhSinhIterationScheme(QdFpIterationScheme):
    """tanh-sinh ``(m, n)-eps`` scheme -- one integrator used for both roles.

    # C++ parity: ``QdFpTanhSinhIterationScheme`` (hpp:94-107, cpp:70-101).
    """

    def __init__(self, m: int, n: int, eps: float) -> None:
        self._m: int = m
        self._n: int = n
        # C++ builds a boost tanh_sinh when QL_BOOST_HAS_TANH_SINH is defined
        # (boost >= 1.69, i.e. always in practice); the GaussLobattoIntegral
        # fallback is dead code upstream.
        self._integrator: Integrator = TanhSinhIntegral(eps)

    def get_number_of_chebyshev_interpolation_nodes(self) -> int:
        return self._n

    def get_number_of_naive_fixed_point_steps(self) -> int:
        return self._m - 1

    def get_number_of_jacobi_newton_fixed_point_steps(self) -> int:
        return 1

    def get_fixed_point_integrator(self) -> Integrator:
        return self._integrator

    def get_exercise_boundary_to_price_integrator(self) -> Integrator:
        return self._integrator


class QdFpLegendreTanhSinhScheme(QdFpLegendreScheme):
    """Legendre inside, tanh-sinh outside: ``(l, m, n)-eps``.

    # C++ parity: ``QdFpLegendreTanhSinhScheme`` (hpp:78-86, cpp:103-116).

    The base is constructed with ``p = 1`` -- that 1-point Legendre rule is
    built and then never used, because
    :meth:`get_exercise_boundary_to_price_integrator` is overridden. C++ does
    exactly the same, and a fresh ``TanhSinhIntegral`` is returned on *every*
    call rather than cached.
    """

    def __init__(self, l: int, m: int, n: int, eps: float) -> None:  # noqa: E741 -- C++ parameter names
        super().__init__(l, m, n, 1)
        self._eps: float = eps

    def get_exercise_boundary_to_price_integrator(self) -> Integrator:
        return TanhSinhIntegral(self._eps)


class DqFpEquation(ABC):
    """Base for the two fixed-point equations for the exercise boundary.

    # C++ parity: ``DqFpEquation`` (qdfpamericanengine.cpp:119-158).

    If the fixed-point integrator is a Gauss-Legendre rule, its nodes and
    weights are lifted out and summed directly (from the last index down, as
    C++ does); otherwise the integrator is called with a closure. The two paths
    differ in their endpoint handling and are therefore not interchangeable.
    """

    def __init__(
        self,
        r: float,
        q: float,
        vol: float,
        b: Callable[[float], float],
        integrator: Integrator,
    ) -> None:
        self.r: float = r
        self.q: float = q
        self.vol: float = vol
        self.b: Callable[[float], float] = b
        self.integrator: Integrator = integrator

        self.x_i: list[float] = []
        self.w_i: list[float] = []
        if isinstance(integrator, GaussLegendreIntegrator):
            integration = integrator.get_integration()
            self.x_i = [float(v) for v in integration.x()]
            self.w_i = [float(v) for v in integration.weights()]

        self.phi: NormalDistribution = NormalDistribution()
        self.big_phi: CumulativeNormalDistribution = CumulativeNormalDistribution()

    def d(self, t: float, z: float) -> tuple[float, float]:
        """# C++ parity: ``DqFpEquation::d`` (cpp:142-147)."""
        v = self.vol * math.sqrt(t)
        m = ieee_div(ieee_log(z) + (self.r - self.q) * t, v) + 0.5 * v
        return (m, m - v)

    @abstractmethod
    def nd_d(self, tau: float, b: float) -> tuple[float, float]:
        """``(N', D')`` w.r.t. the boundary. # C++ parity: ``NDd``."""

    @abstractmethod
    def f(self, tau: float, b: float) -> tuple[float, float, float]:
        """``(N, D, fv)``. # C++ parity: ``f``."""

    # --- shared tau -> 0 limit of fv -------------------------------------

    def _degenerate_fv(self, alpha: float, at_or_above_strike: bool) -> float:
        """The ``tau < QL_EPSILON^2`` limit of ``fv`` shared by FP-A and FP-B.

        C++ writes it out twice (cpp:268-279 and cpp:378-387); the only
        difference is which ``b`` vs ``K`` comparison selects the first branch,
        which the caller has already resolved.
        """
        if at_or_above_strike:
            return alpha
        if close_enough(self.q, 0.0):
            return alpha * self.r * (-1.0 if self.q < 0 else 1.0) / QL_EPSILON
        return ieee_div(alpha * self.r, self.q)


class DqFpEquation_A(DqFpEquation):  # noqa: N801 -- C++ class name is DqFpEquation_A
    """Fixed-point equation A.

    # C++ parity: ``DqFpEquation_A`` (cpp:176-309).
    """

    def __init__(
        self,
        k: float,
        r: float,
        q: float,
        vol: float,
        b: Callable[[float], float],
        integrator: Integrator,
    ) -> None:
        super().__init__(r, q, vol, b, integrator)
        self.k: float = k

    def f(self, tau: float, b: float) -> tuple[float, float, float]:
        # C++ parity: cpp:200-284.
        v = self.vol * math.sqrt(tau)

        if tau < _TAU_FLOOR:
            if close_enough(b, self.k):
                n_val = ieee_div(1.0, M_SQRT2 * M_SQRTPI * v)
                d_val = n_val + 0.5
            else:
                n_val = 0.0
                d_val = 1.0 if b > self.k else 0.0
        else:
            stv = math.sqrt(tau) / self.vol

            if self.x_i:
                k12 = 0.0
                k3 = 0.0
                for i in range(len(self.x_i) - 1, -1, -1):
                    y = self.x_i[i]
                    m = 0.25 * tau * squared(1.0 + y)
                    dp, dm = self.d(m, ieee_div(b, self.b(tau - m)))
                    k12 += (
                        self.w_i[i]
                        * ieee_exp(self.q * tau - self.q * m)
                        * (0.5 * tau * (y + 1.0) * self.big_phi(dp) + stv * self.phi(dp))
                    )
                    k3 += self.w_i[i] * stv * ieee_exp(self.r * tau - self.r * m) * self.phi(dm)
            else:

                def k12_integrand(y: float) -> float:
                    m = 0.25 * tau * squared(1.0 + y)
                    df = ieee_exp(self.q * tau - self.q * m)
                    if y <= 5.0 * QL_EPSILON - 1.0:
                        if close_enough(b, self.b(tau - m)):
                            return df * stv / (M_SQRT2 * M_SQRTPI)
                        return 0.0
                    dp = self.d(m, ieee_div(b, self.b(tau - m)))[0]
                    return df * (
                        0.5 * tau * (y + 1.0) * self.big_phi(dp) + stv * self.phi(dp)
                    )

                def k3_integrand(y: float) -> float:
                    m = 0.25 * tau * squared(1.0 + y)
                    df = ieee_exp(self.r * tau - self.r * m)
                    if y <= 5.0 * QL_EPSILON - 1.0:
                        if close_enough(b, self.b(tau - m)):
                            return df * stv / (M_SQRT2 * M_SQRTPI)
                        return 0.0
                    return df * stv * self.phi(self.d(m, ieee_div(b, self.b(tau - m)))[1])

                k12 = self.integrator(k12_integrand, -1.0, 1.0)
                k3 = self.integrator(k3_integrand, -1.0, 1.0)

            dp, dm = self.d(tau, ieee_div(b, self.k))
            n_val = ieee_div(self.phi(dm), v) + self.r * k3
            d_val = ieee_div(self.phi(dp), v) + self.big_phi(dp) + self.q * k12

        alpha = self.k * ieee_exp(-(self.r - self.q) * tau)
        if tau < _TAU_FLOOR:
            if close_enough(b, self.k):
                fv = alpha
            elif b > self.k:
                fv = 0.0
            else:
                fv = self._degenerate_fv(alpha, at_or_above_strike=False)
        else:
            fv = ieee_div(alpha * n_val, d_val)

        return (n_val, d_val, fv)

    def nd_d(self, tau: float, b: float) -> tuple[float, float]:
        # C++ parity: cpp:286-309.
        if tau < _TAU_FLOOR:
            if close_enough(b, self.k):
                sq_tau = math.sqrt(tau)
                vol2 = self.vol * self.vol
                dd = (
                    M_1_SQRTPI
                    * M_SQRT_2
                    * (
                        -ieee_div(0.5 * vol2 + self.r - self.q, b * self.vol * vol2 * sq_tau)
                        + ieee_div(1.0, b * self.vol * sq_tau)
                    )
                )
                nd = (
                    M_1_SQRTPI
                    * M_SQRT_2
                    * ieee_div(-0.5 * vol2 + self.r - self.q, b * self.vol * vol2 * sq_tau)
                )
            else:
                dd = 0.0
                nd = 0.0
        else:
            dp, dm = self.d(tau, ieee_div(b, self.k))
            dd = ieee_div(
                -self.phi(dp) * dp, b * self.vol * self.vol * tau
            ) + ieee_div(self.phi(dp), b * self.vol * math.sqrt(tau))
            nd = ieee_div(-self.phi(dm) * dm, b * self.vol * self.vol * tau)

        return (nd, dd)


class DqFpEquation_B(DqFpEquation):  # noqa: N801 -- C++ class name is DqFpEquation_B
    """Fixed-point equation B.

    # C++ parity: ``DqFpEquation_B`` (cpp:160-174, 312-400).
    """

    def __init__(
        self,
        k: float,
        r: float,
        q: float,
        vol: float,
        b: Callable[[float], float],
        integrator: Integrator,
    ) -> None:
        super().__init__(r, q, vol, b, integrator)
        self.k: float = k

    def f(self, tau: float, b: float) -> tuple[float, float, float]:
        # C++ parity: cpp:321-392.
        if tau < _TAU_FLOOR:
            if close_enough(b, self.k):
                n_val = d_val = 0.5
            elif b < self.k:
                n_val = d_val = 0.0
            else:
                n_val = d_val = 1.0
        else:
            if self.x_i:
                c = 0.5 * tau
                ni = 0.0
                di = 0.0
                for i in range(len(self.x_i) - 1, -1, -1):
                    u = c * self.x_i[i] + c
                    dp, dm = self.d(tau - u, ieee_div(b, self.b(u)))
                    ni += self.w_i[i] * ieee_exp(self.r * u) * self.big_phi(dm)
                    di += self.w_i[i] * ieee_exp(self.q * u) * self.big_phi(dp)
                ni *= c
                di *= c
            else:

                def ni_integrand(u: float) -> float:
                    df = ieee_exp(self.r * u)
                    if u >= tau * (1.0 - 5.0 * QL_EPSILON):
                        if close_enough(b, self.b(u)):
                            return 0.5 * df
                        return df * (0.0 if b < self.b(u) else 1.0)
                    return df * self.big_phi(self.d(tau - u, ieee_div(b, self.b(u)))[1])

                def di_integrand(u: float) -> float:
                    df = ieee_exp(self.q * u)
                    if u >= tau * (1.0 - 5.0 * QL_EPSILON):
                        if close_enough(b, self.b(u)):
                            return 0.5 * df
                        return df * (0.0 if b < self.b(u) else 1.0)
                    return df * self.big_phi(self.d(tau - u, ieee_div(b, self.b(u)))[0])

                ni = self.integrator(ni_integrand, 0.0, tau)
                di = self.integrator(di_integrand, 0.0, tau)

            dp, dm = self.d(tau, ieee_div(b, self.k))
            n_val = self.big_phi(dm) + self.r * ni
            d_val = self.big_phi(dp) + self.q * di

        alpha = self.k * ieee_exp(-(self.r - self.q) * tau)
        if tau < _TAU_FLOOR:
            fv = self._degenerate_fv(
                alpha, at_or_above_strike=close_enough(b, self.k) or b > self.k
            )
        else:
            fv = ieee_div(alpha * n_val, d_val)

        return (n_val, d_val, fv)

    def nd_d(self, tau: float, b: float) -> tuple[float, float]:
        # C++ parity: cpp:394-400.
        dp, dm = self.d(tau, ieee_div(b, self.k))
        denom = b * self.vol * math.sqrt(tau)
        return (ieee_div(self.phi(dm), denom), ieee_div(self.phi(dp), denom))


# C++ uses function-local statics inside fastScheme() / accurateScheme() /
# highPrecisionScheme(), i.e. one shared instance per process. Mirrored here.
_fast_scheme_instance: QdFpIterationScheme | None = None
_accurate_scheme_instance: QdFpIterationScheme | None = None
_high_precision_scheme_instance: QdFpIterationScheme | None = None


class QdFpAmericanEngine(QdPutCallParityEngine):
    """High performance/precision American engine via fixed-point iteration.

    # C++ parity: ``QdFpAmericanEngine`` (hpp:121-141, cpp:402-516).

    Args:
        bs_process: the Black-Scholes process.
        iteration_scheme: defaults to :meth:`accurate_scheme`, as in C++.
        fp_equation: ``FP_A``, ``FP_B`` or ``Auto`` (the default).
    """

    class FixedPointEquation(IntEnum):
        """# C++ parity: ``QdFpAmericanEngine::FixedPointEquation`` (hpp:123)."""

        FP_A = 0
        FP_B = 1
        Auto = 2

    def __init__(
        self,
        bs_process: GeneralizedBlackScholesProcess | None,
        iteration_scheme: QdFpIterationScheme | None = None,
        fp_equation: QdFpAmericanEngine.FixedPointEquation = FixedPointEquation.Auto,
    ) -> None:
        super().__init__(bs_process)
        self._iteration_scheme: QdFpIterationScheme = (
            QdFpAmericanEngine.accurate_scheme()
            if iteration_scheme is None
            else iteration_scheme
        )
        self._fp_equation: QdFpAmericanEngine.FixedPointEquation = fp_equation

    # --- the three canonical schemes --------------------------------------

    @staticmethod
    def fast_scheme() -> QdFpIterationScheme:
        """``QdFpLegendreScheme(7, 2, 7, 27)``. # C++ parity: cpp:411-415."""
        global _fast_scheme_instance  # noqa: PLW0603 -- mirrors the C++ function-local static
        if _fast_scheme_instance is None:
            _fast_scheme_instance = QdFpLegendreScheme(7, 2, 7, 27)
        return _fast_scheme_instance

    @staticmethod
    def accurate_scheme() -> QdFpIterationScheme:
        """``QdFpLegendreTanhSinhScheme(25, 5, 13, 1e-8)``. # C++ parity: cpp:417-421."""
        global _accurate_scheme_instance  # noqa: PLW0603 -- mirrors the C++ function-local static
        if _accurate_scheme_instance is None:
            _accurate_scheme_instance = QdFpLegendreTanhSinhScheme(25, 5, 13, 1e-8)
        return _accurate_scheme_instance

    @staticmethod
    def high_precision_scheme() -> QdFpIterationScheme:
        """``QdFpTanhSinhIterationScheme(10, 30, 1e-10)``. # C++ parity: cpp:423-427."""
        global _high_precision_scheme_instance  # noqa: PLW0603 -- mirrors the C++ function-local static
        if _high_precision_scheme_instance is None:
            _high_precision_scheme_instance = QdFpTanhSinhIterationScheme(10, 30, 1e-10)
        return _high_precision_scheme_instance

    # --- pricing ----------------------------------------------------------

    def calculate_put(
        self, s: float, k: float, r: float, q: float, vol: float, t: float
    ) -> float:
        """# C++ parity: ``QdFpAmericanEngine::calculatePut`` (cpp:429-516)."""
        qassert.require(
            not (r < 0.0 and q < r),
            "double-boundary case q<r<0 for a put option is given",
        )

        xmax = QdPlusAmericanEngine.x_max(k, r, q)
        n = self._iteration_scheme.get_number_of_chebyshev_interpolation_nodes()

        # The QD+ boundary is the initial guess. Note the n + 1 (not n) nodes
        # and the hardcoded (Halley, 1e-8) -- both are C++'s.
        interp = QdPlusAmericanEngine(
            self._process, n + 1, QdPlusAmericanEngine.SolverType.Halley, 1e-8
        ).get_put_exercise_boundary(s, k, r, q, vol, t)

        z_nodes = interp.nodes()
        x_nodes = 0.5 * math.sqrt(t) * (1.0 + z_nodes)

        def boundary(tau: float) -> float:
            z = 2.0 * math.sqrt(abs(tau) / t) - 1.0
            return xmax * ieee_exp(
                -math.sqrt(max(0.0, interp(z, allow_extrapolation=True)))
            )

        def h(fv: float) -> float:
            return squared(ieee_log(ieee_div(fv, xmax)))

        eqn: DqFpEquation
        if self._fp_equation == QdFpAmericanEngine.FixedPointEquation.FP_A or (
            self._fp_equation == QdFpAmericanEngine.FixedPointEquation.Auto
            and abs(r - q) < 0.001
        ):
            eqn = DqFpEquation_A(
                k, r, q, vol, boundary, self._iteration_scheme.get_fixed_point_integrator()
            )
        else:
            eqn = DqFpEquation_B(
                k, r, q, vol, boundary, self._iteration_scheme.get_fixed_point_integrator()
            )

        size = int(x_nodes.shape[0])
        y = np.zeros(size, dtype=np.float64)
        y[0] = 0.0

        n_newton = self._iteration_scheme.get_number_of_jacobi_newton_fixed_point_steps()
        for _ in range(n_newton):
            for i in range(1, size):
                tau = squared(float(x_nodes[i]))
                b = boundary(tau)

                n_val, d_val, fv = eqn.f(tau, b)

                if tau < QL_EPSILON:
                    y[i] = h(fv)
                else:
                    nd, dd = eqn.nd_d(tau, b)
                    fd = (
                        k
                        * ieee_exp(-(r - q) * tau)
                        * (ieee_div(nd, d_val) - ieee_div(dd * n_val, d_val * d_val))
                    )
                    y[i] = h(b - ieee_div(fv - b, fd - 1.0))
            interp.update_y(y)

        n_fp = self._iteration_scheme.get_number_of_naive_fixed_point_steps()
        for _ in range(n_fp):
            for i in range(1, size):
                tau = squared(float(x_nodes[i]))
                fv = eqn.f(tau, boundary(tau))[2]
                y[i] = h(fv)
            interp.update_y(y)

        aov = QdPlusAddOnValue(t, s, k, r, q, vol, xmax, interp)
        add_on = self._iteration_scheme.get_exercise_boundary_to_price_integrator()(
            aov, 0.0, math.sqrt(t)
        )

        european_value = BlackCalculator.from_type_strike(
            OptionType.Put,
            k,
            s * ieee_exp((r - q) * t),
            vol * math.sqrt(t),
            ieee_exp(-r * t),
        ).value()

        return max(european_value, 0.0) + max(0.0, add_on)


__all__ = [
    "DqFpEquation",
    "DqFpEquation_A",
    "DqFpEquation_B",
    "QdFpAmericanEngine",
    "QdFpIterationScheme",
    "QdFpLegendreScheme",
    "QdFpLegendreTanhSinhScheme",
    "QdFpTanhSinhIterationScheme",
]
