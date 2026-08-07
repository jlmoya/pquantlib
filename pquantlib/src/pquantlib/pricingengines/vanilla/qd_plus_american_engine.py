"""QD+ American option engine and the put/call-parity base it shares with QD-FP.

# C++ parity: ql/pricingengines/vanilla/qdplusamericanengine.{hpp,cpp} (v1.43)
#             -- ``detail::QdPutCallParityEngine``, ``detail::QdPlusAddOnValue``,
#             ``QdPlusBoundaryEvaluator`` and ``QdPlusAmericanEngine``.

The QD+ approximation (Li 2009) writes an American put as the European put plus
an early-exercise add-on integral over the exercise boundary ``B(t)``, and
approximates ``B`` by solving, at each ``tau``, a one-dimensional equation whose
root is the critical stock price. QuantLib parametrises the boundary as

    B(tau) = xMax * exp(-sqrt(max(0, q(z)))),   z = 2 sqrt(tau/T) - 1

and interpolates ``q`` with a Chebyshev (second-kind) interpolation of the
squared log-ratio, which is what :meth:`QdPlusAmericanEngine.get_put_exercise_boundary`
returns. That interpolation is *also* the initial guess consumed by
:class:`~pquantlib.pricingengines.vanilla.qd_fp_american_engine.QdFpAmericanEngine`,
so this module is a dependency of that one rather than an alternative to it.

References:
    Li, M. (2009), "Analytical Approximations for the Critical Stock Prices of
    American Options: A Performance Comparison", Working paper, Georgia
    Institute of Technology. https://mpra.ub.uni-muenchen.de/15018/

    Leif Andersen, Mark Lake (2021), "Fast American Option Pricing: The
    Double-Boundary Case" -- Table 2 is :meth:`QdPlusAmericanEngine.x_max`.

Structure notes
---------------
* ``QdPutCallParityEngine`` is C++ ``detail::QdPutCallParityEngine``: it reads
  the market off the process, maps a CALL onto a PUT by the McDonald-Schroder
  parity ``(S, K, r, q) -> (K, S, q, r)``, and owns every degenerate-input
  early return. It fills ``results_.value`` and *nothing else* -- neither this
  engine nor QD-FP produces a single greek.
* ``QdPlusBoundaryEvaluator`` is a genuine top-level class in the C++ .cpp
  (forward-declared in the .hpp), not a lambda, and is ported as one. It is the
  functor handed to the 1-D solvers: ``__call__`` is ``f``, ``derivative`` is
  ``f'`` and ``fprime2`` is ``f''`` (Newton needs the first, the hand-rolled
  Halley / super-Halley iteration needs both).
* ``QdPlusAddOnValue`` is the add-on integrand in the ``z = sqrt(t)`` variable.

Divergences from C++, all deliberate and local
----------------------------------------------
Python raises where C++ returns an IEEE special: ``1/0``, ``log`` of a
non-positive number and ``exp`` of a large argument. C++ relies on the
resulting ``inf`` / ``nan`` flowing into a comparison that then rejects the
branch (``extremT > 0 && extremT < T`` is false for a NaN, for instance).
:func:`ieee_div`, :func:`ieee_log` and :func:`ieee_exp` reproduce the IEEE result
instead of raising, so the control flow matches. They are shared with
``qd_fp_american_engine``, which is why they are exported rather than private.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Final

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.closeness import close, close_enough
from pquantlib.math.constants import QL_EPSILON, QL_MAX_REAL
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.integrals.tanh_sinh_integral import TanhSinhIntegral
from pquantlib.math.interpolations.chebyshev_interpolation import (
    ChebyshevInterpolation,
    PointsType,
)
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.math.solvers1d.newton import Newton
from pquantlib.math.solvers1d.ridder import Ridder
from pquantlib.math.solvers1d.solver_1d import Solver1D
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)

# C++ ``Null<Real>()`` -- ql/utilities/null.hpp yields
# ``std::numeric_limits<float>::max()`` for any floating-point type. The
# boundary evaluator seeds its cache key with it, and ``buildInSolver`` tests
# its ``guess`` argument against it.
_NULL_REAL: Final[float] = 3.4028234663852886e38

# C++ ``Null<Size>()`` -- ``std::numeric_limits<int>::max()``.
_NULL_SIZE: Final[int] = 2147483647

# ql/mathconstants.hpp spells these out as decimal literals; reproduced exactly
# so the degenerate ``tau < QL_EPSILON^2`` branches of the fixed-point equations
# (which are the only consumers) agree bit-for-bit.
M_SQRTPI: Final[float] = 1.77245385090551602792981
M_1_SQRTPI: Final[float] = 0.564189583547756286948
M_SQRT2: Final[float] = 1.41421356237309504880
M_SQRT_2: Final[float] = 0.7071067811865475244008443621048490392848359376887


def ieee_div(a: float, b: float) -> float:
    """``a / b`` with IEEE-754 semantics for ``b == 0``.

    C++ never raises here: ``x/0`` is ``+-inf`` (or NaN for ``0/0``) and the
    value then flows into a comparison that rejects the branch. Python raises
    ``ZeroDivisionError`` instead, which would turn a rejected branch into a
    crash, so the IEEE result is produced explicitly.
    """
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def ieee_log(x: float) -> float:
    """``std::log(x)`` with IEEE-754 semantics outside the domain."""
    if math.isnan(x) or x < 0.0:
        return math.nan
    if x == 0.0:
        return -math.inf
    return math.log(x)


def ieee_exp(x: float) -> float:
    """``std::exp(x)`` with IEEE-754 overflow to ``+inf`` instead of raising."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def squared(x: float) -> float:
    """# C++ parity: ``QuantLib::squared`` (ql/math/functional.hpp)."""
    return x * x


class QdPlusBoundaryEvaluator:
    """The QD+ boundary equation ``f(S) = 0`` and its first two derivatives.

    # C++ parity: ``QdPlusBoundaryEvaluator``
    #             (qdplusamericanengine.cpp:41-122). Declared in the header as
    #             ``class QdPlusBoundaryEvaluator;`` -- a real top-level class,
    #             not a nested helper.

    ``__call__`` counts its own invocations, which is what
    :meth:`QdPlusAmericanEngine.put_exercise_boundary_at_tau` reports back as
    the evaluation count; ``derivative`` and ``fprime2`` deliberately do *not*
    count, matching C++ (``++nrEvaluations`` appears only in ``operator()``).

    The intermediate quantities are cached against the last argument ``sc``.
    The cache key is compared against the *unclamped* argument while the
    quantities are computed from ``max(QL_EPSILON, S)`` -- so an argument below
    ``QL_EPSILON`` recomputes on every call. That asymmetry is C++'s and is
    reproduced rather than tidied, because it is observable through the
    evaluation count.
    """

    def __init__(
        self,
        s: float,
        strike: float,
        rf: float,
        dy: float,
        vol: float,
        t: float,
        big_t: float,
    ) -> None:
        # ``T`` is a constructor parameter in C++ too, and is likewise unused.
        del big_t
        self._phi_cdf: CumulativeNormalDistribution = CumulativeNormalDistribution()
        self._phi_pdf: NormalDistribution = NormalDistribution()
        self.tau: float = t
        self.k: float = strike
        self.sigma: float = vol
        self.sigma2: float = vol * vol
        self.v: float = vol * math.sqrt(t)
        self.r: float = rf
        self.q: float = dy
        self.dr: float = ieee_exp(-rf * t)
        self.dq: float = ieee_exp(-dy * t)
        # r/(1 - exp(-r tau)) loses all its digits as r tau -> 0; C++ switches
        # to the series 1/(tau (1 - r tau/2 (1 - r tau/3))) below |r tau| = 1e-5.
        self.ddr: float = (
            ieee_div(rf, 1.0 - self.dr)
            if abs(rf * t) > 1e-5
            else ieee_div(1.0, t * (1.0 - 0.5 * rf * t * (1.0 - rf * t / 3.0)))
        )
        self.omega: float = ieee_div(2.0 * (rf - dy), self.sigma2)
        disc = math.sqrt(squared(self.omega - 1.0) + 8.0 * ieee_div(self.ddr, self.sigma2))
        self.lambda_: float = 0.5 * (-(self.omega - 1.0) - disc)
        self.lambda_prime: float = ieee_div(2.0 * self.ddr * self.ddr, self.sigma2 * disc)
        self.alpha: float = ieee_div(
            2.0 * self.dr, self.sigma2 * (2.0 * self.lambda_ + self.omega - 1.0)
        )
        self.beta: float = (
            self.alpha
            * (
                self.ddr
                + ieee_div(self.lambda_prime, 2.0 * self.lambda_ + self.omega - 1.0)
            )
            - self.lambda_
        )
        self._x_max: float = QdPlusAmericanEngine.x_max(strike, rf, dy)
        self._x_min: float = QL_EPSILON * 1e4 * min(0.5 * (strike + s), self._x_max)

        self._nr_evaluations: int = 0
        self._sc: float = _NULL_REAL
        self._dp: float = 0.0
        self._dm: float = 0.0
        self._phi_dp: float = 0.0
        self._phi_dm: float = 0.0
        self._pdf_dp: float = 0.0
        self._npv: float = 0.0
        self._theta: float = 0.0
        self._charm: float = 0.0

    # --- cached intermediates -------------------------------------------

    def _pre_calculate(self, s: float) -> None:
        # C++ parity: qdplusamericanengine.cpp:97-109.
        s = max(QL_EPSILON, s)
        self._sc = s
        self._dp = ieee_div(ieee_log(ieee_div(s * self.dq, self.k * self.dr)), self.v) + 0.5 * self.v
        self._dm = self._dp - self.v
        self._phi_dp = self._phi_cdf(-self._dp)
        self._phi_dm = self._phi_cdf(-self._dm)
        self._pdf_dp = self._phi_pdf(self._dp)

        self._npv = self.dr * self.k * self._phi_dm - s * self.dq * self._phi_dp
        self._theta = (
            self.r * self.k * self.dr * self._phi_dm
            - self.q * s * self.dq * self._phi_dp
            - ieee_div(self.sigma2 * s, 2.0 * self.v) * self.dq * self._pdf_dp
        )
        self._charm = -self.dq * (
            self._pdf_dp * (ieee_div(self.r - self.q, self.v) - ieee_div(self._dm, 2.0 * self.tau))
            + self.q * self._phi_dp
        )

    # --- functor surface --------------------------------------------------

    def __call__(self, s: float) -> float:
        # C++ parity: qdplusamericanengine.cpp:60-73.
        self._nr_evaluations += 1
        if s != self._sc:
            self._pre_calculate(s)

        if close_enough(self.k - s, self._npv):
            return (1.0 - self.dq * self._phi_dp) * s + ieee_div(
                self.alpha * self._theta, self.dr
            )
        c0 = (
            -self.beta
            - self.lambda_
            + ieee_div(self.alpha * self._theta, self.dr * (self.k - s - self._npv))
        )
        return (1.0 - self.dq * self._phi_dp) * s + (self.lambda_ + c0) * (
            self.k - s - self._npv
        )

    def derivative(self, s: float) -> float:
        # C++ parity: qdplusamericanengine.cpp:74-80.
        if s != self._sc:
            self._pre_calculate(s)
        return (
            1.0
            - self.dq * self._phi_dp
            + ieee_div(self.dq, self.v) * self._pdf_dp
            + self.beta * (1.0 - self.dq * self._phi_dp)
            + ieee_div(self.alpha, self.dr) * self._charm
        )

    def fprime2(self, s: float) -> float:
        # C++ parity: qdplusamericanengine.cpp:81-90. Note the raw (unclamped)
        # ``S`` in the gamma / colour expressions -- C++ does the same.
        if s != self._sc:
            self._pre_calculate(s)
        gamma = ieee_div(self._pdf_dp * self.dq, self.v * s)
        colour = gamma * (
            self.q
            + ieee_div((self.r - self.q) * self._dp, self.v)
            + ieee_div(1.0 - self._dp * self._dm, 2.0 * self.tau)
        )
        return (
            self.dq
            * (
                ieee_div(self._pdf_dp, s * self.v)
                - ieee_div(self._pdf_dp * self._dp, s * self.v * self.v)
            )
            + self.beta * gamma
            + ieee_div(self.alpha, self.dr) * colour
        )

    # --- inspectors -------------------------------------------------------

    def xmin(self) -> float:
        return self._x_min

    def xmax(self) -> float:
        return self._x_max

    def evaluations(self) -> int:
        return self._nr_evaluations


class QdPlusAddOnValue:
    """Early-exercise add-on integrand in the ``z = sqrt(t)`` variable.

    # C++ parity: ``detail::QdPlusAddOnValue``
    #             (qdplusamericanengine.hpp:58-77, .cpp:125-158).

    ``q_z`` is the Chebyshev interpolation of ``squared(log(B(tau)/xmax))`` on
    ``z in [-1, 1]``; it is read with extrapolation enabled because the caller's
    ``z`` can drift a few ulps outside the node range.
    """

    def __init__(
        self,
        t: float,
        s: float,
        k: float,
        r: float,
        q: float,
        vol: float,
        xmax: float,
        q_z: Interpolation,
    ) -> None:
        self._t: float = t
        self._s: float = s
        self._k: float = k
        self._xmax: float = xmax
        self._r: float = r
        self._q: float = q
        self._vol: float = vol
        self._q_z: Interpolation = q_z
        self._phi: CumulativeNormalDistribution = CumulativeNormalDistribution()

    def __call__(self, z: float) -> float:
        # C++ parity: qdplusamericanengine.cpp:132-158.
        t = z * z
        # NB: C++ shadows the member ``q_`` with a local ``q`` here.
        q_at = self._q_z(
            2.0 * math.sqrt(max(0.0, self._t - t) / self._t) - 1.0,
            allow_extrapolation=True,
        )
        b_t = self._xmax * ieee_exp(-math.sqrt(max(0.0, q_at)))

        dr = ieee_exp(-self._r * t)
        dq = ieee_exp(-self._q * t)
        v = self._vol * math.sqrt(t)

        if v >= QL_EPSILON:
            if b_t > QL_EPSILON:
                dp = ieee_div(ieee_log(ieee_div(self._s * dq, b_t * dr)), v) + 0.5 * v
                return 2.0 * z * (
                    self._r * self._k * dr * self._phi(-dp + v)
                    - self._q * self._s * dq * self._phi(-dp)
                )
            return 0.0
        if close_enough(self._s * dq, b_t * dr):
            return z * (self._r * self._k * dr - self._q * self._s * dq)
        if b_t * dr > self._s * dq:
            return 2.0 * z * (self._r * self._k * dr - self._q * self._s * dq)
        return 0.0


class QdPutCallParityEngine(
    GenericEngine[OptionArguments, OneAssetOptionResults], ABC
):
    """American vanilla engine that prices a call as a put by parity.

    # C++ parity: ``detail::QdPutCallParityEngine``
    #             (qdplusamericanengine.hpp:40-56, .cpp:161-228).

    Subclasses implement :meth:`calculate_put` for the non-degenerate case;
    everything degenerate is handled here, in
    :meth:`_calculate_put_with_edge_cases`.

    ``process`` may be ``None``: C++ takes an ``ext::shared_ptr`` and
    ``registerWith`` is a no-op on a null pointer, which is exactly how the
    upstream test-suite drives
    :meth:`QdPlusAmericanEngine.put_exercise_boundary_at_tau` without a market.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess | None) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess | None = process
        if process is not None:
            process.register_with(self)

    @abstractmethod
    def calculate_put(
        self, s: float, k: float, r: float, q: float, vol: float, t: float
    ) -> float:
        """Price an American put with no degenerate inputs.

        # C++ parity: ``detail::QdPutCallParityEngine::calculatePut`` (pure).
        """

    def calculate(self) -> None:
        """# C++ parity: ``detail::QdPutCallParityEngine::calculate``."""
        args = self._arguments
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.American, "not an American option"
        )

        payoff = args.payoff
        qassert.require(isinstance(payoff, StrikedTypePayoff), "non-striked payoff given")
        assert isinstance(payoff, StrikedTypePayoff)

        process = self._process
        qassert.require(process is not None, "no Black-Scholes process given")
        assert process is not None

        spot = process.x0()
        qassert.require(spot >= 0.0, "negative underlying given")

        maturity = args.exercise.last_date()
        t = process.time(maturity)
        s = process.x0()
        k = payoff.strike()
        r = ieee_div(-ieee_log(process.risk_free_rate().discount(maturity)), t)
        q = ieee_div(-ieee_log(process.dividend_yield().discount(maturity)), t)
        vol = process.black_volatility().black_vol_at_time(t, k)

        qassert.require(s >= 0, "zero or positive underlying value is required")
        qassert.require(k >= 0, "zero or positive strike is required")
        qassert.require(vol >= 0, "zero or positive volatility is required")

        option_type = payoff.option_type()
        if option_type == OptionType.Put:
            self._results.value = self._calculate_put_with_edge_cases(s, k, r, q, vol, t)
        elif option_type == OptionType.Call:
            # McDonald-Schroder parity: a call on (S, K, r, q) is a put on
            # (K, S, q, r).
            self._results.value = self._calculate_put_with_edge_cases(k, s, q, r, vol, t)
        else:
            qassert.fail("unknown option type")

    def _calculate_put_with_edge_cases(
        self, s: float, k: float, r: float, q: float, vol: float, t: float
    ) -> float:
        """# C++ parity: ``calculatePutWithEdgeCases`` (.cpp:198-228)."""
        if close(k, 0.0):
            return 0.0

        if close(s, 0.0):
            return max(k, k * ieee_exp(-r * t))

        if r <= 0.0 and r <= q:
            # No early exercise is ever optimal -- the European value is exact.
            return max(
                0.0,
                BlackCalculator.from_type_strike(
                    OptionType.Put,
                    k,
                    s * ieee_exp((r - q) * t),
                    vol * math.sqrt(t),
                    ieee_exp(-r * t),
                ).value(),
            )

        if close(vol, 0.0):
            # Deterministic underlying: the value is the best discounted
            # intrinsic over t in [0, T], and the interior optimum (if any) is
            # at extremT = log(rK/(qS))/(r-q).
            def intrinsic(tt: float) -> float:
                return max(0.0, k * ieee_exp(-r * tt) - s * ieee_exp(-q * tt))

            npv0 = intrinsic(0.0)
            npv_t = intrinsic(t)
            extrem_t = QL_MAX_REAL if close_enough(r, q) else _extreme_time(r, k, q, s)

            if extrem_t > 0.0 and extrem_t < t:
                return max(npv0, npv_t, intrinsic(extrem_t))
            return max(npv0, npv_t)

        return self.calculate_put(s, k, r, q, vol, t)


def _extreme_time(r: float, k: float, q: float, s: float) -> float:
    """``std::log(r*K/(q*S))/(r-q)`` under IEEE-754 semantics.

    Called only from the zero-volatility branch, where ``r != q`` (the
    ``close_enough(r, q)`` case short-circuits to ``QL_MAX_REAL`` first). A NaN
    or infinite result is rejected by the ``0 < extremT < T`` guard, which is
    exactly what C++ relies on.
    """
    return ieee_div(ieee_log(ieee_div(r * k, q * s)), r - q)


class QdPlusAmericanEngine(QdPutCallParityEngine):
    """American engine based on the QD+ approximation to the exercise boundary.

    # C++ parity: ``QdPlusAmericanEngine``
    #             (qdplusamericanengine.hpp:92-127, .cpp:231-395).

    Its main purpose upstream is to seed
    :class:`~pquantlib.pricingengines.vanilla.qd_fp_american_engine.QdFpAmericanEngine`
    with a good initial boundary, but it is a usable engine in its own right.

    Args:
        process: the Black-Scholes process; may be ``None`` when only
            :meth:`put_exercise_boundary_at_tau` / :meth:`get_put_exercise_boundary`
            are used.
        interpolation_points: Chebyshev nodes used for the boundary.
        solver_type: which root finder solves the boundary equation. Brent,
            Newton and Ridder go through :class:`Solver1D`; Halley and
            SuperHalley are a hand-rolled iteration with a Brent fallback.
        eps: root accuracy.
        max_iter: evaluation budget. ``None`` (C++ ``Null<Size>()``) means 100
            for Brent / Newton / Ridder and 10 for the Halley pair.
    """

    class SolverType(IntEnum):
        """# C++ parity: ``QdPlusAmericanEngine::SolverType`` (hpp:94)."""

        Brent = 0
        Newton = 1
        Ridder = 2
        Halley = 3
        SuperHalley = 4

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess | None,
        interpolation_points: int = 8,
        solver_type: QdPlusAmericanEngine.SolverType = SolverType.Halley,
        eps: float = 1e-6,
        max_iter: int | None = None,
    ) -> None:
        super().__init__(process)
        self._interpolation_points: int = interpolation_points
        self._solver_type: QdPlusAmericanEngine.SolverType = solver_type
        self._eps: float = eps
        if max_iter is None or max_iter == _NULL_SIZE:
            self._max_iter: int = (
                100
                if solver_type
                in (
                    QdPlusAmericanEngine.SolverType.Newton,
                    QdPlusAmericanEngine.SolverType.Brent,
                    QdPlusAmericanEngine.SolverType.Ridder,
                )
                else 10
            )
        else:
            self._max_iter = max_iter

    # --- static helpers ---------------------------------------------------

    @staticmethod
    def x_max(k: float, r: float, q: float) -> float:
        """Upper bound of the exercise boundary.

        # C++ parity: ``QdPlusAmericanEngine::xMax`` (.cpp:231-251) --
        # Table 2 of Leif Andersen, Mark Lake (2021), "Fast American Option
        # Pricing: The Double-Boundary Case".
        """
        if r > 0.0 and q > 0.0:
            return k * min(1.0, r / q)
        if r > 0.0 and q <= 0.0:
            return k
        if r == 0.0 and q < 0.0:
            return k
        if r == 0.0 and q >= 0.0:
            return 0.0  # European case
        if r < 0.0 and q >= 0.0:
            return 0.0  # European case
        if r < 0.0 and q < r:
            return k  # double boundary case
        if r < 0.0 and r <= q < 0.0:
            return 0.0  # European case
        qassert.fail("internal error")

    # --- boundary ---------------------------------------------------------

    def _build_in_solver(
        self,
        evaluator: QdPlusBoundaryEvaluator,
        solver: Solver1D,
        s: float,
        strike: float,
        max_iter: int,
        guess: float = _NULL_REAL,
    ) -> float:
        """# C++ parity: ``QdPlusAmericanEngine::buildInSolver`` (.cpp:267-289)."""
        del strike  # C++ takes it and does not use it either.
        solver.set_max_evaluations(max_iter)
        solver.set_lower_bound(evaluator.xmin())

        fxmin = evaluator(evaluator.xmin())
        xmax = max(0.5 * (evaluator.xmax() + s), evaluator.xmax())
        # NB the bracket-widening loop is capped by ``self._max_iter``, not by
        # the ``max_iter`` argument -- C++ does the same, and it matters for the
        # Halley fallback which passes 10x the member value.
        while evaluator(xmax) * fxmin > 0.0 and evaluator.evaluations() < self._max_iter:
            xmax *= 2.0

        if guess == _NULL_REAL:
            guess = 0.5 * (xmax + s)

        if guess >= xmax:
            guess = math.nextafter(xmax, -1.0)
        elif guess <= evaluator.xmin():
            guess = math.nextafter(evaluator.xmin(), sys.float_info.max)

        return solver.solve(evaluator, self._eps, guess, evaluator.xmin(), xmax)

    def put_exercise_boundary_at_tau(
        self,
        s: float,
        k: float,
        r: float,
        q: float,
        vol: float,
        t: float,
        tau: float,
    ) -> tuple[int, float]:
        """Critical stock price at time-to-maturity ``tau``.

        # C++ parity: ``QdPlusAmericanEngine::putExerciseBoundaryAtTau``
        #             (.cpp:291-344).

        Returns ``(evaluations, boundary)`` -- the same ``(Size, Real)`` pair as
        C++, in the same order.
        """
        if tau < QL_EPSILON:
            return (0, QdPlusAmericanEngine.x_max(k, r, q))

        evaluator = QdPlusBoundaryEvaluator(s, k, r, q, vol, tau, t)

        solver_type = self._solver_type
        if solver_type == QdPlusAmericanEngine.SolverType.Brent:
            x = self._build_in_solver(evaluator, Brent(), s, k, self._max_iter)
        elif solver_type == QdPlusAmericanEngine.SolverType.Newton:
            x = self._build_in_solver(evaluator, Newton(), s, k, self._max_iter)
        elif solver_type == QdPlusAmericanEngine.SolverType.Ridder:
            x = self._build_in_solver(evaluator, Ridder(), s, k, self._max_iter)
        elif solver_type in (
            QdPlusAmericanEngine.SolverType.Halley,
            QdPlusAmericanEngine.SolverType.SuperHalley,
        ):
            is_halley = solver_type == QdPlusAmericanEngine.SolverType.Halley
            x = evaluator.xmax()
            x_min = evaluator.xmin()
            fx = 0.0
            result_close_enough = False
            # do { ... } while (!closeEnough && evaluations < maxIter)
            while True:
                x_old = x
                fx = evaluator(x)
                f_prime = evaluator.derivative(x)
                lf = ieee_div(fx * evaluator.fprime2(x), f_prime * f_prime)
                step = (
                    ieee_div(ieee_div(1.0, 1.0 - 0.5 * lf) * fx, f_prime)
                    if is_halley
                    else ieee_div((1.0 + ieee_div(0.5 * lf, 1.0 - lf)) * fx, f_prime)
                )
                x = max(x_min, x - step)
                result_close_enough = abs(x - x_old) < 0.5 * self._eps
                if result_close_enough or evaluator.evaluations() >= self._max_iter:
                    break

            if not result_close_enough and not close(abs(fx), 0.0):
                x = self._build_in_solver(
                    evaluator, Brent(), s, k, 10 * self._max_iter, x
                )
        else:
            qassert.fail("unknown solver type")

        return (evaluator.evaluations(), x)

    def get_put_exercise_boundary(
        self, s: float, k: float, r: float, q: float, vol: float, t: float
    ) -> ChebyshevInterpolation:
        """Chebyshev interpolation of ``squared(log(B(tau)/xMax))`` over ``z``.

        # C++ parity: ``QdPlusAmericanEngine::getPutExerciseBoundary``
        #             (.cpp:346-362).

        The abscissa is ``z in [-1, 1]`` with ``tau = T (1 + z)^2 / 4``.
        """
        xmax = QdPlusAmericanEngine.x_max(k, r, q)

        def f(z: float) -> float:
            x_sq = 0.25 * t * squared(1.0 + z)
            boundary = self.put_exercise_boundary_at_tau(s, k, r, q, vol, t, x_sq)[1]
            return squared(ieee_log(ieee_div(boundary, xmax)))

        return ChebyshevInterpolation(
            n=self._interpolation_points, f=f, points_type=PointsType.SecondKind
        )

    # --- pricing ----------------------------------------------------------

    def calculate_put(
        self, s: float, k: float, r: float, q: float, vol: float, t: float
    ) -> float:
        """# C++ parity: ``QdPlusAmericanEngine::calculatePut`` (.cpp:364-395)."""
        qassert.require(
            not (r < 0.0 and q < r),
            "double-boundary case q<r<0 for a put option is given",
        )

        q_z = self.get_put_exercise_boundary(s, k, r, q, vol, t)

        xmax = QdPlusAmericanEngine.x_max(k, r, q)
        aov = QdPlusAddOnValue(t, s, k, r, q, vol, xmax, q_z)

        # C++ uses boost's tanh-sinh when QL_BOOST_HAS_TANH_SINH is defined
        # (boost >= 1.69), which it is for every supported build; the
        # GaussLobattoIntegral fallback is dead code upstream.
        add_on = TanhSinhIntegral(self._eps)(aov, 0.0, math.sqrt(t))

        qassert.require(
            add_on > -10.0 * self._eps, f"negative early exercise value {add_on}"
        )

        european_value = max(
            0.0,
            BlackCalculator.from_type_strike(
                OptionType.Put,
                k,
                s * ieee_exp((r - q) * t),
                vol * math.sqrt(t),
                ieee_exp(-r * t),
            ).value(),
        )

        return european_value + max(0.0, add_on)


__all__ = [
    "M_1_SQRTPI",
    "M_SQRT2",
    "M_SQRTPI",
    "M_SQRT_2",
    "QdPlusAddOnValue",
    "QdPlusAmericanEngine",
    "QdPlusBoundaryEvaluator",
    "QdPutCallParityEngine",
    "ieee_div",
    "ieee_exp",
    "ieee_log",
    "squared",
]
