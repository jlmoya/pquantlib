"""ZabrModel — the ZABR smile model and its five evaluation surfaces.

# C++ parity: ql/termstructures/volatility/zabr.{hpp,cpp} (v1.43).

Reference: Andreasen, Huge, *ZABR — Expansions for the Masses*,
Preliminary Version, December 2011, http://ssrn.com/abstract=1980726.

ZABR generalises SABR with one extra parameter ``gamma`` controlling the
elasticity of the variance dynamics:

.. math::

   dF_t &= \\alpha_t F_t^{\\beta}\\, dW_t \\\\
   d\\alpha_t &= \\nu\\, \\alpha_t^{\\gamma}\\, dZ_t \\\\
   d\\langle W, Z\\rangle_t &= \\rho\\, dt

and collapses to SABR at ``gamma = 1``.

Three facts about this class are easy to miss and expensive to get wrong:

1. **``nu`` is transformed in the constructor** to
   ``nu * alpha**(1 - gamma)``, and every internal formula — plus the
   public :meth:`nu` inspector — uses the transformed value. At
   ``gamma = 1`` the factor is 1, so a port that skips the transform
   agrees exactly there and nowhere else.

2. **``x(K)`` for a vector of strikes is not N scalar calls.** The
   strikes are ordered by their ``y`` value, the walk starts at the
   strike whose ``y`` is closest to zero and marches outward in both
   directions, carrying the integrator's ``(y0, u0)`` state across
   strikes. The two therefore differ at the integrator's tolerance
   (order 1e-9 relative in practice), and both spellings are reachable
   from the public API.

3. **The ODE integrator is QuantLib's own**
   :class:`~pquantlib.math.ode.adaptive_runge_kutta.AdaptiveRungeKutta`
   with ``(eps=1e-8, h1=1e-5, hmin=0.0)`` — a Cash-Karp pair with
   Numerical-Recipes step control. ``h1`` is the *initial step size*,
   not a relative tolerance.

Math-symbol names (``alpha``, ``beta``, ``nu``, ``rho``, ``gamma``,
``F``, ``x``, ``y``, ``u``) are deliberate carryovers from the paper and
the C++ source.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.fdm_dupire_1d_op import FdmDupire1dOp
from pquantlib.experimental.finitedifferences.fdm_zabr_op import FdmZabrOp
from pquantlib.experimental.finitedifferences.glued_1d_mesher import Glued1dMesher
from pquantlib.math.array import Array
from pquantlib.math.closeness import close
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.interpolations.bicubic_spline import BicubicSpline
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    CubicInterpolation,
    DerivativeApprox,
)
from pquantlib.math.interpolations.sabr_formula import validate_sabr_parameters
from pquantlib.math.matrix import Matrix
from pquantlib.math.ode.adaptive_runge_kutta import AdaptiveRungeKutta
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_backward_solver import (
    FdmBackwardSolver,
)

_INV_CUM_NORMAL = InverseCumulativeNormal()


class ZabrModel:
    """ZABR model at a single expiry.

    # C++ parity: ``class ZabrModel``.

    Args:
        expiry_time: option expiry in years; must be positive.
        forward: ATM forward; must be non-negative.
        alpha, beta, nu, rho: SABR parameters, validated by
            ``validateSabrParameters``.
        gamma: variance elasticity; must be non-negative.
    """

    def __init__(
        self,
        expiry_time: float,
        forward: float,
        alpha: float,
        beta: float,
        nu: float,
        rho: float,
        gamma: float,
    ) -> None:
        self._expiry_time: float = expiry_time
        self._forward: float = forward
        self._alpha: float = alpha
        self._beta: float = beta
        # C++ stores the TRANSFORMED nu in nu_ (zabr.cpp ctor init list).
        self._nu: float = nu * (alpha ** (1.0 - gamma))
        self._rho: float = rho
        self._gamma: float = gamma

        validate_sabr_parameters(alpha, beta, nu, rho)
        qassert.require(gamma >= 0.0, f"gamma must be non negative: {gamma} not allowed")
        qassert.require(forward >= 0.0, f"forward must be non negative: {forward} not allowed")
        qassert.require(
            expiry_time > 0.0, f"expiry time must be positive: {expiry_time} not allowed"
        )

    # --- inspectors -------------------------------------------------------

    def forward(self) -> float:
        return self._forward

    def expiry_time(self) -> float:
        return self._expiry_time

    def alpha(self) -> float:
        return self._alpha

    def beta(self) -> float:
        return self._beta

    def nu(self) -> float:
        """The **transformed** ``nu = nu_input * alpha**(1 - gamma)``.

        # C++ parity: ``ZabrModel::nu`` returns ``nu_``, which the ctor
        # already transformed.
        """
        return self._nu

    def rho(self) -> float:
        return self._rho

    def gamma(self) -> float:
        return self._gamma

    # --- volatility surfaces ----------------------------------------------

    def lognormal_volatility(self, strike: float) -> float:
        """# C++ parity: ``ZabrModel::lognormalVolatility(Real)``."""
        return self.lognormal_volatility_vector([strike])[0]

    def lognormal_volatility_vector(self, strikes: Sequence[float]) -> list[float]:
        """# C++ parity: ``ZabrModel::lognormalVolatility(const std::vector<Real>&)``."""
        xs = self._x_vector(strikes)
        return [self._lognormal_volatility_helper(k, x) for k, x in zip(strikes, xs, strict=True)]

    def normal_volatility(self, strike: float) -> float:
        """# C++ parity: ``ZabrModel::normalVolatility(Real)``."""
        return self.normal_volatility_vector([strike])[0]

    def normal_volatility_vector(self, strikes: Sequence[float]) -> list[float]:
        """# C++ parity: ``ZabrModel::normalVolatility(const std::vector<Real>&)``."""
        xs = self._x_vector(strikes)
        return [self._normal_volatility_helper(k, x) for k, x in zip(strikes, xs, strict=True)]

    def local_volatility(self, f: float) -> float:
        """# C++ parity: ``ZabrModel::localVolatility(Real)``."""
        return self.local_volatility_vector([f])[0]

    def local_volatility_vector(self, f: Sequence[float]) -> list[float]:
        """# C++ parity: ``ZabrModel::localVolatility(const std::vector<Real>&)``."""
        xs = self._x_vector(f)
        return [self._local_volatility_helper(fi, x) for fi, x in zip(f, xs, strict=True)]

    def _lognormal_volatility_helper(self, strike: float, x: float) -> float:
        """# C++ parity: ``ZabrModel::lognormalVolatilityHelper``."""
        if close(strike, self._forward):
            return (self._forward ** (self._beta - 1.0)) * self._alpha
        return math.log(self._forward / strike) / x

    def _normal_volatility_helper(self, strike: float, x: float) -> float:
        """# C++ parity: ``ZabrModel::normalVolatilityHelper``."""
        if close(strike, self._forward):
            return (self._forward**self._beta) * self._alpha
        return (self._forward - strike) / x

    def _local_volatility_helper(self, f: float, x: float) -> float:
        """# C++ parity: ``ZabrModel::localVolatilityHelper``."""
        return (
            self._alpha
            * (math.fabs(f) ** self._beta)
            / self._f(self._y(f), (self._alpha ** (self._gamma - 1.0)) * x)
        )

    # --- the x / y / F transforms -----------------------------------------

    def _y(self, strike: float) -> float:
        """# C++ parity: ``ZabrModel::y``."""
        if close(self._beta, 1.0):
            return math.log(self._forward / strike) * (self._alpha ** (self._gamma - 2.0))
        if strike < 0.0:
            base = (self._forward ** (1.0 - self._beta)) + ((-strike) ** (1.0 - self._beta))
        else:
            base = (self._forward ** (1.0 - self._beta)) - (strike ** (1.0 - self._beta))
        return base * (self._alpha ** (self._gamma - 2.0)) / (1.0 - self._beta)

    def _f(self, y: float, u: float) -> float:
        """The ODE right-hand side ``du/dy``.

        # C++ parity: ``ZabrModel::F``.
        """
        g2 = self._gamma - 2.0
        nu = self._nu
        a = 1.0 + g2 * g2 * nu * nu * y * y + 2.0 * self._rho * g2 * nu * y
        b = (
            2.0 * self._rho * (1.0 - self._gamma) * nu
            + 2.0 * (1.0 - self._gamma) * g2 * nu * nu * y
        )
        c = (1.0 - self._gamma) * (1.0 - self._gamma) * nu * nu
        return (-b * u + math.sqrt(b * b * u * u - 4.0 * a * (c * u * u - 1.0))) / (2.0 * a)

    def _x_vector(self, strikes: Sequence[float]) -> list[float]:
        """The Andreasen-Huge ``x(K)`` transform for a strike vector.

        # C++ parity: ``ZabrModel::x(const std::vector<Real>&)``.

        For ``gamma == 1`` there is a closed form. Otherwise the ODE
        ``du/dy = F(y, u)`` is integrated INCREMENTALLY: ``y`` is built in
        reverse strike order (so it ascends), the walk starts at the entry
        nearest zero and runs outward in both directions, each leg
        continuing from the previous leg's ``(y0, u0)``.
        """
        n = len(strikes)
        qassert.require(
            strikes[0] > 0.0 or self._beta < 1.0,
            f"strikes must be positive ({strikes[0]}) if beta = 1",
        )
        for i in range(1, n):
            qassert.require(
                strikes[i] > strikes[i - 1],
                f"strikes must be strictly ascending ({strikes[i - 1]},{strikes[i]})",
            )

        rk = AdaptiveRungeKutta(1.0e-8, 1.0e-5, 0.0)
        # C++ transforms strikes.rbegin()..rend() into y, i.e. y is in
        # DESCENDING strike order and therefore ascending in value.
        y = [self._y(k) for k in reversed(strikes)]
        result = [0.0] * n

        if close(self._gamma, 1.0):
            for m in range(n):
                j = math.sqrt(
                    1.0 + self._nu * self._nu * y[m] * y[m] - 2.0 * self._rho * self._nu * y[m]
                )
                result[n - 1 - m] = (
                    math.log((j + self._nu * y[m] - self._rho) / (1.0 - self._rho)) / self._nu
                )
            return result

        # First index with y > 0 (std::upper_bound on a sorted range).
        ynz = int(np.searchsorted(np.asarray(y, dtype=np.float64), 0.0, side="right"))
        if ynz > 0 and close(y[ynz - 1], 0.0):
            ynz -= 1
        if ynz == n:
            ynz -= 1

        for direction in (1, -1):
            y0 = 0.0
            u0 = 0.0
            m = ynz + (-1 if direction == -1 else 0)
            while (m >= 0) if direction == -1 else (m < n):
                u = rk.solve_1d(self._f, u0, y0, y[m])
                result[n - 1 - m] = u * (self._alpha ** (1.0 - self._gamma))
                u0 = u
                y0 = y[m]
                m += direction

        return result

    # --- finite-difference prices -----------------------------------------

    def fd_price(self, strike: float) -> float:
        """# C++ parity: ``ZabrModel::fdPrice(Real)``."""
        return self.fd_price_vector([strike])[0]

    def fd_price_vector(self, strikes: Sequence[float]) -> list[float]:
        """Call prices from the 1-D Dupire local-volatility PDE.

        # C++ parity: ``ZabrModel::fdPrice(const std::vector<Real>&)``.

        NOTE the grid depends on the strike vector: ``start`` and ``end``
        are derived from ``strikes.front()`` and ``strikes.back()``, so
        pricing a strike alone and pricing it inside a wider vector give
        different numbers. That is C++ behaviour, not an artefact.
        """
        # C++ marks these "TODO put these parameters somewhere"; they are
        # hard-coded and load-bearing.
        start = min(0.00001, strikes[0] * 0.5)
        end = max(0.10, strikes[-1] * 1.5)
        size = 500
        density = 0.1
        steps = math.ceil(self._expiry_time * 24)
        damping_steps = 5

        m1: Fdm1dMesher = Concentrating1dMesher(
            start, end, size, (self._forward, density), True
        )
        mesher = FdmMesherComposite(m1)

        # initial values: the call payoff read backwards, max(F - k, 0).
        k: Array = mesher.locations(0)
        rhs: Array = np.maximum(self._forward - k, 0.0)

        loc_vol: Array = np.asarray(
            self.local_volatility_vector([float(v) for v in k]), dtype=np.float64
        )

        op = FdmDupire1dOp(mesher, loc_vol)
        solver = FdmBackwardSolver(op, None, FdmSchemeDesc.douglas())
        rhs = solver.rollback(rhs, self._expiry_time, 0.0, steps, damping_steps)

        solution = CubicInterpolation(
            k,
            rhs,
            DerivativeApprox.Spline,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )
        solution.enable_extrapolation(False)
        return [solution(s) for s in strikes]

    def full_fd_price(self, strike: float) -> float:
        """Call price from the full 2-D (forward, vol) ZABR PDE.

        # C++ parity: ``ZabrModel::fullFdPrice``.

        The grid is sized from a 1% tail quantile of the lognormal
        approximations to F and alpha, widened by ``scaleFactor = 1.5``
        and clamped around the strike. C++ flags these constants with
        "TODO what are good values here, still experimenting with them";
        they are reproduced verbatim because they determine the answer.
        """
        eps = 0.01
        scale_factor = 1.5
        norm_inv_eps = _INV_CUM_NORMAL(1.0 - eps)
        alpha_i = self._alpha * (self._forward ** (self._beta - 1.0))
        sqrt_t = math.sqrt(self._expiry_time)
        # nu is already standardized within this class.
        v0 = self._alpha * math.exp(-scale_factor * norm_inv_eps * sqrt_t * self._nu)
        v1 = self._alpha * math.exp(scale_factor * norm_inv_eps * sqrt_t * self._nu)
        f0 = self._forward * math.exp(-scale_factor * norm_inv_eps * sqrt_t * alpha_i)
        f1 = self._forward * math.exp(scale_factor * norm_inv_eps * sqrt_t * alpha_i)
        v1 = min(v1, 2.0)
        f0 = min(strike / 2.0, f0)
        f1 = max(strike * 1.5, min(f1, max(2.0, strike * 1.5)))

        size_f = 100
        size_v = 100
        steps = int(24 * self._expiry_time + 1)
        damping_steps = 5
        density_f = 0.1
        density_v = 0.1

        qassert.require(
            f0 <= strike <= f1,
            f"strike ({strike}) must be inside pde grid [{f0};{f1}]",
        )

        # Two concentrating meshers glued at the midpoint, each with a
        # required point at one of min/max(forward, strike).
        x0 = min(self._forward, strike)
        x1 = max(self._forward, strike)
        size_fa = max(4, math.ceil(((x0 + x1) / 2.0 - f0) / (f1 - f0) * size_f))
        size_fb = size_f - size_fa + 1  # common point, so we can spend one more here
        mfa = Concentrating1dMesher(f0, (x0 + x1) / 2.0, size_fa, (x0, density_f), True)
        mfb = Concentrating1dMesher((x0 + x1) / 2.0, f1, size_fb, (x1, density_f), True)
        mf: Fdm1dMesher = Glued1dMesher(mfa, mfb)
        mv: Fdm1dMesher = Concentrating1dMesher(v0, v1, size_v, (self._alpha, density_v), True)
        mesher = FdmMesherComposite(mf, mv)

        f_axis = mf.locations()
        v_axis = mv.locations()

        rhs: Array = np.maximum(mesher.locations(0) - strike, 0.0)

        op = FdmZabrOp(mesher, self._beta, self._nu, self._rho, self._gamma)
        solver = FdmBackwardSolver(op, None, FdmSchemeDesc.hundsdorfer())
        rhs = solver.rollback(rhs, self._expiry_time, 0.0, steps, damping_steps)

        # rhs is laid out with the forward axis varying fastest, so the
        # (f, v) matrix is a plain row-major reshape with v as the row index.
        result: Matrix = rhs.reshape(v_axis.size, f_axis.size)
        interpolation = BicubicSpline(f_axis, v_axis, result)
        interpolation.enable_extrapolation(False)
        return interpolation(self._forward, self._alpha)


__all__ = ["ZabrModel"]
