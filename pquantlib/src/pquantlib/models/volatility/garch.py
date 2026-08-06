"""GARCH(1,1) volatility model and its calibration.

# C++ parity: ql/models/volatility/garch.{hpp,cpp} @ v1.43.

``sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}``, fitted by
maximum likelihood. Volatilities are expressed on an annual basis.

What the constructor arguments mean
-----------------------------------
``Garch11(a, b, vl)`` takes the long-term VARIANCE ``vl``, not ``omega``:
it sets ``gamma = 1 - a - b`` and ``omega() == vl * gamma``
(garch.hpp:56-70). A port that read the third argument as ``omega`` would
reproduce ``alpha``/``beta`` and get every forecast wrong.

The calibration path
--------------------
``calibrate`` squares the series (``to_r2``), then hands ``r2`` to
``calibrate_r2``, which:

1. builds the autocovariance function of the centered ``r2`` out to
   ``maxLag = (Size)sqrt(n)`` lags (garch.cpp:416-421);
2. derives up to two initial guesses from it — a moment-matching guess
   (``initialGuess1``, garch.cpp:230-313) and a guess built from the
   ``acf(i+1) = gamma*acf(i)`` decay ratio (``initialGuess2``,
   garch.cpp:318-369) — each refined by a constrained non-linear
   least-squares fit of the ACF that is allowed to FAIL silently
   (``catch (const std::exception&)``, garch.cpp:309/365);
3. runs ``Simplex(0.001)`` on the negative log-likelihood from whichever
   guess is cheaper, or from both when the mode is ``DoubleOptimization``.

Which of those steps happen is what :class:`Garch11Mode` selects.

Two upstream quirks are reproduced deliberately
-----------------------------------------------
* ``initialGuess2`` (garch.cpp:318-369) assigns ``beta`` and ``omega`` but
  NEVER ``alpha`` outside the least-squares block. Its caller passes
  ``opt2[1]`` of an ``Array opt2(3)``, and QuantLib's ``Array(Size)``
  does not value-initialise, so when that block throws, C++ reads an
  UNINITIALISED double and feeds it to the cost function. This port
  starts ``alpha`` at 0.0 instead — the only reachable difference is on
  the path where C++ has undefined behaviour.
* In the ``DoubleOptimization`` branch (garch.cpp:461-498) ``opt1`` is
  overwritten with the OPTIMISED point before ``constraints.test(opt1)``
  is consulted, but when that test fails ``fCost1`` keeps the cost of the
  original GUESS. So the comparison that picks a winner can score one
  point while returning another. Transcribed as-is.

``calibrate_r2`` and Python's lack of overloads
-----------------------------------------------
C++ declares six ``calibrate_r2`` overloads (garch.hpp:171-235) that differ
by which of ``mode`` / ``mean_r2`` / ``method`` / ``endCriteria`` /
``constraints`` / ``initialGuess`` they take, and returns three results
through ``Real&`` out-parameters. Python has neither, so there is ONE
``calibrate_r2`` with keyword-only arguments returning a
:class:`Garch11CalibrationResult`:

===================================================== =========================================================
C++ overload                                          Python call
===================================================== =========================================================
``(mode, r2, mean_r2, &a,&b,&o)``                     ``calibrate_r2(r2, mode=, mean_r2=)``
``(mode, r2, mean_r2, method, ec, &a,&b,&o)``         ``calibrate_r2(r2, mode=, mean_r2=, method=, end_criteria=)``
``(r2, mean_r2, method, ec, guess, &a,&b,&o)``        ``calibrate_r2(r2, mean_r2=, method=, end_criteria=, initial_guess=)``
``(r2, method, ec, guess, &a,&b,&o)``                 ``calibrate_r2(r2, method=, end_criteria=, initial_guess=)``
``(r2, mean_r2, method, cons, ec, guess, &a,&b,&o)``  ``... + constraints=``
``(r2, method, cons, ec, guess, &a,&b,&o)``           ``... + constraints=`` (no ``mean_r2``)
===================================================== =========================================================

Passing ``mean_r2`` with an ``initial_guess`` centers ``r2`` first, which is
exactly what the C++ overloads that take both do (garch.cpp:512-523,
:544-556).

``autocovariances``
-------------------
``ql/math/autocovariance.hpp`` has no pquantlib module yet, so the two
functions ``calibrate_r2`` needs are ported here as module-private helpers
(:func:`_double_ft`, :func:`_autocovariances`) on top of the already-ported
:class:`~pquantlib.math.fast_fourier_transform.FastFourierTransform`. If
that header is ported later these should be deleted in favour of it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from typing import ClassVar, NamedTuple, overload

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON, QL_MAX_REAL
from pquantlib.math.fast_fourier_transform import FastFourierTransform
from pquantlib.math.optimization.constraint import Constraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.least_square import LeastSquareProblem, NonLinearLeastSquare
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.math.optimization.simplex import Simplex
from pquantlib.time.time_series import TimeSeries
from pquantlib.volatility_model import VolatilityCompositor

# C++ parity: garch.cpp:31 — the anonymous-namespace ``tol_level``. It is
# simultaneously the constraint slack, the ACF-fit accuracy and all three
# EndCriteria epsilons.
_TOL_LEVEL: float = 1.0e-8


class Garch11Mode(IntEnum):
    """Which initial guess (or guesses) the calibration starts from.

    # C++ parity: nested ``Garch11::Mode`` (garch.hpp:42-52), flattened to
    # module scope as this port does for every nested enum (cf.
    # ``IntervalPriceType`` for ``IntervalPrice::Type``). Integer values
    # follow C++ declaration order. Also reachable as ``Garch11.Mode``.
    """

    MomentMatchingGuess = 0
    """Moment-matching estimates for mean(r2), acf(0) and acf(1)."""
    GammaGuess = 1
    """Gamma estimated from the property ``acf(i+1) = gamma*acf(i)`` for ``i > 1``."""
    BestOfTwo = 2
    """Optimize once, from whichever of the two guesses has the lower cost."""
    DoubleOptimization = 3
    """Optimize from BOTH guesses and keep the better result."""


class Garch11CalibrationResult(NamedTuple):
    """What C++ ``calibrate_r2`` returns plus its three out-parameters.

    ``problem`` is ``None`` when C++ would have returned a null
    ``shared_ptr<Problem>``, i.e. when every optimization attempt threw and
    the calibration fell back to an unoptimized initial guess.
    """

    problem: Problem | None
    alpha: float
    beta: float
    omega: float


# --- ql/math/autocovariance.hpp (ported here; see the module docstring) ------


def _double_ft(data: Sequence[float]) -> list[complex]:
    """``input -> FFT -> |.|^2 -> FFT``.

    # C++ parity: ``detail::double_ft`` (ql/math/autocovariance.hpp:40-57).

    The order is ``min_order(n) + 1``, i.e. the transform is over at least
    twice as many points as there are data — that zero padding is what makes
    the circular convolution the FFT computes agree with the linear one.
    """
    order = FastFourierTransform.min_order(len(data)) + 1
    fft = FastFourierTransform(order)
    ft = fft.transform(data)
    # C++ ``std::norm``: for finite arguments, re*re + im*im — NOT abs(z)**2,
    # which would round twice.
    tmp = [z.real * z.real + z.imag * z.imag for z in ft]
    return fft.transform(tmp)


def _autocovariances(data: Sequence[float], max_lag: int) -> list[float]:
    """Unbiased auto-covariances of already-centered ``data``, lags ``0..max_lag``.

    # C++ parity: ``autocovariances(begin, end, out, maxLag)``
    # (ql/math/autocovariance.hpp:96-110) — the overload that does NOT remove
    # the mean. The divisor decreases by one per lag (``w2 -= 1.0``), which is
    # what makes the estimator unbiased.
    """
    n_data = len(data)
    qassert.require(max_lag < n_data, "number of covariances must be less than data size")
    ft = _double_ft(data)
    w1 = 1.0 / float(len(ft))
    w2 = float(n_data)
    out: list[float] = []
    for k in range(max_lag + 1):
        out.append(ft[k].real * w1 / w2)
        w2 -= 1.0
    return out


# --- garch.cpp anonymous namespace ------------------------------------------


class _Garch11Constraint(Constraint):
    """``omega > 0``, ``alpha, beta >= 0``, ``gamma_lower <= alpha+beta < gamma_upper``.

    # C++ parity: ``Garch11Constraint`` (garch.cpp:33-51). The parameter
    # vector is ordered ``[omega, alpha, beta]``.
    """

    __slots__ = ("_gamma_lower", "_gamma_upper")

    def __init__(self, gamma_lower: float, gamma_upper: float) -> None:
        self._gamma_lower: float = gamma_lower
        self._gamma_upper: float = gamma_upper

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        # C++ parity: garch.cpp:40-45.
        qassert.require(params.size >= 3, "size of parameters vector < 3")
        return bool(
            params[0] > 0
            and params[1] >= 0
            and params[2] >= 0
            and params[1] + params[2] < self._gamma_upper
            and params[1] + params[2] >= self._gamma_lower
        )


class _FitAcfConstraint(Constraint):
    """``gamma_lower <= gamma < gamma_upper``, ``0 <= beta <= gamma``.

    # C++ parity: ``FitAcfConstraint`` (garch.cpp:207-224). The parameter
    # vector is ordered ``[gamma, beta]``.
    """

    __slots__ = ("_gamma_lower", "_gamma_upper")

    def __init__(self, gamma_lower: float, gamma_upper: float) -> None:
        self._gamma_lower: float = gamma_lower
        self._gamma_upper: float = gamma_upper

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        # C++ parity: garch.cpp:214-218.
        qassert.require(params.size >= 2, "size of parameters vector < 2")
        return bool(
            params[0] >= self._gamma_lower
            and params[0] < self._gamma_upper
            and params[1] >= 0
            and params[1] <= params[0]
        )


class _Garch11CostFunction(CostFunction):
    """Half the mean negative log-likelihood of ``r2`` under ``[omega, alpha, beta]``.

    # C++ parity: ``Garch11CostFunction`` (garch.cpp:54-139).

    The recursion ``sigma2 = omega + alpha*u2 + beta*sigma2`` is sequential —
    ``sigma2`` at step ``i`` depends on step ``i-1`` — so this stays a scalar
    Python loop rather than a numpy expression. That is also what keeps the
    floating-point accumulation order identical to C++'s.
    """

    __slots__ = ("_r2",)

    def __init__(self, r2: Sequence[float]) -> None:
        # C++ parity: garch.cpp:66-68 — held by reference.
        self._r2: Sequence[float] = r2

    def value(self, x: npt.NDArray[np.float64]) -> float:
        # C++ parity: garch.cpp:70-80.
        omega = float(x[0])
        alpha = float(x[1])
        beta = float(x[2])
        retval = 0.0
        sigma2 = 0.0
        u2 = 0.0
        for r2 in self._r2:
            sigma2 = omega + alpha * u2 + beta * sigma2
            u2 = r2
            retval += math.log(sigma2) + u2 / sigma2
        return retval / (2.0 * len(self._r2))

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: garch.cpp:82-93 — the per-observation contributions,
        # each already divided by 2N, so their sum is ``value(x)``.
        omega = float(x[0])
        alpha = float(x[1])
        beta = float(x[2])
        norm = 2.0 * len(self._r2)
        retval = np.zeros(len(self._r2), dtype=np.float64)
        sigma2 = 0.0
        u2 = 0.0
        for i, r2 in enumerate(self._r2):
            sigma2 = omega + alpha * u2 + beta * sigma2
            u2 = r2
            retval[i] = (math.log(sigma2) + u2 / sigma2) / norm
        return retval

    def gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
        # C++ parity: garch.cpp:95-114.
        omega = float(x[0])
        alpha = float(x[1])
        beta = float(x[2])
        g0 = 0.0
        g1 = 0.0
        g2 = 0.0
        sigma2 = 0.0
        u2 = 0.0
        sigma2prev = sigma2
        u2prev = u2
        norm = 2.0 * len(self._r2)
        for r2 in self._r2:
            sigma2 = omega + alpha * u2 + beta * sigma2
            u2 = r2
            w = (sigma2 - u2) / (sigma2 * sigma2)
            g0 += w
            g1 += u2prev * w
            g2 += sigma2prev * w
            u2prev = u2
            sigma2prev = sigma2
        grad[0] = g0 / norm
        grad[1] = g1 / norm
        grad[2] = g2 / norm

    def value_and_gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> float:
        # C++ parity: garch.cpp:116-139 — one pass for both.
        omega = float(x[0])
        alpha = float(x[1])
        beta = float(x[2])
        retval = 0.0
        g0 = 0.0
        g1 = 0.0
        g2 = 0.0
        sigma2 = 0.0
        u2 = 0.0
        sigma2prev = sigma2
        u2prev = u2
        norm = 2.0 * len(self._r2)
        for r2 in self._r2:
            sigma2 = omega + alpha * u2 + beta * sigma2
            u2 = r2
            retval += math.log(sigma2) + u2 / sigma2
            w = (sigma2 - u2) / (sigma2 * sigma2)
            g0 += w
            g1 += u2prev * w
            g2 += sigma2prev * w
            u2prev = u2
            sigma2prev = sigma2
        grad[0] = g0 / norm
        grad[1] = g1 / norm
        grad[2] = g2 / norm
        return retval / norm


class _FitAcfProblem(LeastSquareProblem):
    """Least-squares fit of the theoretical GARCH(1,1) ACF to the sample one.

    # C++ parity: ``FitAcfProblem`` (garch.cpp:142-204). Parameters are
    # ``[gamma, beta]``; residual 0 is the kurtosis-like ratio
    # ``A2^2/A4``, residual 1 is ``rho(1)``, and residuals ``2..`` are the
    # geometric decay ``gamma^(lag-1) * rho(1)``.
    """

    __slots__ = ("_a2", "_acf", "_idx")

    def __init__(self, a2: float, acf: Sequence[float], idx: Sequence[int]) -> None:
        # C++ parity: garch.cpp:158-159.
        self._a2: float = a2
        self._acf: Sequence[float] = acf
        self._idx: Sequence[int] = idx

    def size(self) -> int:
        # C++ parity: garch.cpp:161.
        return len(self._idx)

    def target_and_value(
        self,
        x: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        fct2fit: npt.NDArray[np.float64],
    ) -> None:
        # C++ parity: garch.cpp:163-178.
        a4 = self._acf[0] + self._a2 * self._a2
        gamma = float(x[0])
        beta = float(x[1])
        target[0] = self._a2 * self._a2 / a4
        fct2fit[0] = (1 - 3 * gamma * gamma - 2 * beta * beta + 4 * beta * gamma) / (
            3 * (1 - gamma * gamma)
        )
        target[1] = self._acf[1] / a4
        fct2fit[1] = gamma * (1 - fct2fit[0]) - beta
        for i in range(2, len(self._idx)):
            target[i] = self._acf[self._idx[i]] / a4
            fct2fit[i] = math.pow(gamma, self._idx[i] - 1) * fct2fit[1]

    def target_value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
        grad_fct2fit: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        fct2fit: npt.NDArray[np.float64],
    ) -> None:
        # C++ parity: garch.cpp:180-204.
        a4 = self._acf[0] + self._a2 * self._a2
        gamma = float(x[0])
        beta = float(x[1])
        target[0] = self._a2 * self._a2 / a4
        w1 = 1 - 3 * gamma * gamma - 2 * beta * beta + 4 * beta * gamma
        w2 = 1 - gamma * gamma
        fct2fit[0] = w1 / (3 * w2)
        grad_fct2fit[0][0] = (
            (2.0 / 3.0) * ((2 * beta - 3 * gamma) * w2 + 2 * w1 * gamma) / (w2 * w2)
        )
        grad_fct2fit[0][1] = (4.0 / 3.0) * (gamma - beta) / w2
        target[1] = self._acf[1] / a4
        fct2fit[1] = gamma * (1 - fct2fit[0]) - beta
        grad_fct2fit[1][0] = (1 - fct2fit[0]) - gamma * grad_fct2fit[0][0]
        grad_fct2fit[1][1] = -gamma * grad_fct2fit[0][1] - 1
        for i in range(2, len(self._idx)):
            target[i] = self._acf[self._idx[i]] / a4
            w1 = math.pow(gamma, self._idx[i] - 1)
            fct2fit[i] = w1 * fct2fit[1]
            grad_fct2fit[i][0] = (self._idx[i] - 1) * (w1 / gamma) * fct2fit[1] + w1 * grad_fct2fit[
                1
            ][0]
            grad_fct2fit[i][1] = w1 * grad_fct2fit[1][1]


class _Guess(NamedTuple):
    """``initialGuess1`` / ``initialGuess2``: the C++ return value plus out-params."""

    gamma_lower: float
    alpha: float
    beta: float
    omega: float


def _gamma_lower_from(a: float) -> float:
    """Smallest feasible ``gamma`` implied by the kurtosis ratio ``A``.

    # C++ parity: the shared line in garch.cpp:238 and :324.
    """
    if a <= 1.0 / 3.0 - _TOL_LEVEL:
        return math.sqrt((1 - 3 * a) / (3 - 3 * a)) + _TOL_LEVEL
    return _TOL_LEVEL


def _refine_by_acf_fit(
    acf: Sequence[float],
    mean_r2: float,
    idx: Sequence[int],
    gamma: float,
    beta: float,
    gamma_lower: float,
    alpha: float,
    omega: float,
) -> tuple[float, float, float]:
    """Constrained non-linear least-squares refinement of ``(gamma, beta)``.

    # C++ parity: the identical ``try { ... } catch (const std::exception&) {}``
    # blocks in garch.cpp:294-311 and garch.cpp:350-367. On ANY failure — or
    # when the refined point violates the GARCH constraint — the caller's
    # values are returned unchanged, which is what "failed -- returning
    # initial values" means there.
    """
    constraints = _Garch11Constraint(gamma_lower, 1.0 - _TOL_LEVEL)
    x = np.array([gamma, beta], dtype=np.float64)
    try:
        c = _FitAcfConstraint(gamma_lower, 1.0 - _TOL_LEVEL)
        nnls = NonLinearLeastSquare(c)
        nnls.set_initial_value(x)
        pr = _FitAcfProblem(mean_r2, acf, idx)
        x = nnls.perform(pr)
        guess = np.array(
            [mean_r2 * (1 - float(x[0])), float(x[0]) - float(x[1]), float(x[1])],
            dtype=np.float64,
        )
        if constraints.test(guess):
            omega = float(guess[0])
            alpha = float(guess[1])
            beta = float(guess[2])
    except Exception:  # C++ catches std::exception (garch.cpp:309, :365)
        pass  # failed -- returning initial values
    return alpha, beta, omega


def _initial_guess1(acf: Sequence[float], mean_r2: float) -> _Guess:
    """Moment-matching initial guess, refined against the ACF.

    # C++ parity: ``initialGuess1`` (garch.cpp:230-313).
    """
    a21 = acf[1]
    a4 = acf[0] + mean_r2 * mean_r2

    a = mean_r2 * mean_r2 / a4  # 1/sigma^2
    b = a21 / a4  # rho(1)

    gamma_lower = _gamma_lower_from(a)

    gamma = gamma_lower + (1 - gamma_lower) * 0.5
    beta = min(gamma, max(gamma * (1 - a) - b, 0.0))
    alpha = gamma - beta
    omega = mean_r2 * (1 - gamma)

    if abs(a - 0.5) < QL_EPSILON:
        # C++ parity: garch.cpp:246-250.
        gamma = max(gamma_lower, -(1 + 4 * b * b) / (4 * b))
        beta = min(gamma, max(gamma * (1 - a) - b, 0.0))
        alpha = gamma - beta
        omega = mean_r2 * (1 - gamma)
    elif a > 1.0 - QL_EPSILON:
        # C++ parity: garch.cpp:252-256.
        gamma = max(gamma_lower, -(1 + b * b) / (2 * b))
        beta = min(gamma, max(gamma * (1 - a) - b, 0.0))
        alpha = gamma - beta
        omega = mean_r2 * (1 - gamma)
    else:
        # C++ parity: garch.cpp:257-279 — the quadratic in beta.
        d2 = (3 * a - 1) * (2 * b * b + (1 - a) * (2 * a - 1))
        if d2 >= 0:
            d = math.sqrt(d2)
            root = (b - d) / (2 * a - 1)
            g = 0.0
            if root >= _TOL_LEVEL and root <= 1.0 - _TOL_LEVEL:
                g = (root + b) / (1 - a)
            if g < gamma_lower:
                root = (b + d) / (2 * a - 1)
                if root >= _TOL_LEVEL and root <= 1.0 - _TOL_LEVEL:
                    g = (root + b) / (1 - a)
            if g >= gamma_lower:
                gamma = g
                beta = min(gamma, max(gamma * (1 - a) - b, 0.0))
                alpha = gamma - beta
                omega = mean_r2 * (1 - gamma)

    # C++ parity: garch.cpp:282-288 — lags 0 and 1 always, then any lag whose
    # ACF is positive and strictly decreasing.
    idx: list[int] = []
    n_cov = len(acf) - 1
    for i in range(n_cov + 1):
        if i < 2 or (acf[i] > 0 and acf[i - 1] > 0 and acf[i - 1] > acf[i]):
            idx.append(i)

    alpha, beta, omega = _refine_by_acf_fit(
        acf, mean_r2, idx, gamma, beta, gamma_lower, alpha, omega
    )
    return _Guess(gamma_lower, alpha, beta, omega)


def _initial_guess2(acf: Sequence[float], mean_r2: float) -> _Guess:
    """Initial guess from the ACF decay ratio, refined against the ACF.

    # C++ parity: ``initialGuess2`` (garch.cpp:318-369).

    NOTE the missing ``alpha`` assignment described in the module docstring:
    C++ leaves it at whatever the caller's ``Array`` element happened to
    hold unless the least-squares refinement succeeds. Here it starts at 0.0.
    """
    a21 = acf[1]
    a4 = acf[0] + mean_r2 * mean_r2
    a = mean_r2 * mean_r2 / a4  # 1/sigma^2
    b = a21 / a4  # rho(1)
    gamma_lower = _gamma_lower_from(a)

    # C++ parity: garch.cpp:328-341 — mean of acf(i)/acf(i-1) over the lags
    # where the ACF is positive and strictly decreasing.
    gamma = 0.0
    nn = 0
    idx: list[int] = []
    n_cov = len(acf) - 1
    for i in range(n_cov + 1):
        if i < 2:
            idx.append(i)
        if i > 1 and acf[i] > 0 and acf[i - 1] > 0 and acf[i - 1] > acf[i]:
            gamma += acf[i] / acf[i - 1]
            nn += 1
            idx.append(i)
    if nn > 0:
        gamma /= nn
    # C++ parity: garch.cpp:342 — ``if (gamma < gammaLower) gamma = gammaLower;``.
    gamma = max(gamma, gamma_lower)
    beta = min(gamma, max(gamma * (1 - a) - b, 0.0))
    omega = mean_r2 * (1 - gamma)
    # C++ never assigns alpha here — see the docstring.
    alpha = 0.0

    alpha, beta, omega = _refine_by_acf_fit(
        acf, mean_r2, idx, gamma, beta, gamma_lower, alpha, omega
    )
    return _Guess(gamma_lower, alpha, beta, omega)


def _calibrate_r2_with_guess(
    r2: Sequence[float],
    method: OptimizationMethod,
    constraints: Constraint,
    end_criteria: EndCriteria,
    initial_guess: npt.NDArray[np.float64],
) -> Garch11CalibrationResult:
    """Minimize the GARCH cost over ``r2`` from ``initial_guess``.

    # C++ parity: ``Garch11::calibrate_r2(r2, method, constraints,
    # endCriteria, initGuess, &alpha, &beta, &omega)`` (garch.cpp:525-542) —
    # the one overload every other one funnels into. The optimum is read back
    # as ``[omega, alpha, beta]``.
    """
    cost = _Garch11CostFunction(r2)
    problem = Problem(cost, constraints, initial_guess)
    # C++ discards the EndCriteria::Type minimize() returns (garch.cpp:534-536).
    method.minimize(problem, end_criteria)
    optimum = problem.current_value
    return Garch11CalibrationResult(
        problem, float(optimum[1]), float(optimum[2]), float(optimum[0])
    )


def _calibrate_r2_from_mode(
    mode: Garch11Mode,
    r2: Sequence[float],
    mean_r2: float,
    method: OptimizationMethod,
    end_criteria: EndCriteria,
) -> Garch11CalibrationResult:
    """Build the initial guess(es) for ``mode`` and optimize from them.

    # C++ parity: ``Garch11::calibrate_r2(mode, r2, mean_r2, method,
    # endCriteria, &alpha, &beta, &omega)`` (garch.cpp:402-500).
    """
    data_size = float(len(r2))
    qassert.require(data_size >= 4, "Data series is too short to fit GARCH model")
    qassert.require(mean_r2 > 0, "Data series is constant")
    # NOTE: garch.cpp:407-413 seeds the out-parameters here
    # (``alpha = beta = 0``, ``omega = mean_r2*n/(n-1)``). Every path below
    # overwrites all three, so those stores are dead upstream and have no
    # counterpart here.

    # C++ parity: garch.cpp:415-422.
    max_lag = int(math.sqrt(data_size))
    tmp = [x - mean_r2 for x in r2]
    acf = _autocovariances(tmp, max_lag)
    qassert.require(acf[0] > 0, "Data series is constant")

    cost = _Garch11CostFunction(r2)

    # Two initial guesses based on fitting the ACF (garch.cpp:426-440).
    # C++ leaves the unused Array uninitialised; zeros here are only ever
    # read on a path C++ leaves undefined.
    gamma_lower = 0.0
    opt1 = np.zeros(3, dtype=np.float64)
    f_cost1 = QL_MAX_REAL
    if mode != Garch11Mode.GammaGuess:
        guess1 = _initial_guess1(acf, mean_r2)
        gamma_lower = guess1.gamma_lower
        opt1[0] = guess1.omega
        opt1[1] = guess1.alpha
        opt1[2] = guess1.beta
        f_cost1 = cost.value(opt1)

    opt2 = np.zeros(3, dtype=np.float64)
    f_cost2 = QL_MAX_REAL
    if mode != Garch11Mode.MomentMatchingGuess:
        guess2 = _initial_guess2(acf, mean_r2)
        gamma_lower = guess2.gamma_lower
        opt2[0] = guess2.omega
        opt2[1] = guess2.alpha
        opt2[2] = guess2.beta
        f_cost2 = cost.value(opt2)

    constraints = _Garch11Constraint(gamma_lower, 1.0 - _TOL_LEVEL)

    if mode != Garch11Mode.DoubleOptimization:
        return _optimize_once(
            r2, method, constraints, end_criteria, opt1, opt2, f_cost1, f_cost2
        )
    return _optimize_twice(
        r2, method, constraints, end_criteria, cost, opt1, opt2, f_cost1, f_cost2
    )


def _optimize_once(
    r2: Sequence[float],
    method: OptimizationMethod,
    constraints: Constraint,
    end_criteria: EndCriteria,
    opt1: npt.NDArray[np.float64],
    opt2: npt.NDArray[np.float64],
    f_cost1: float,
    f_cost2: float,
) -> Garch11CalibrationResult:
    """Optimize from the cheaper of the two guesses, falling back to it on failure.

    # C++ parity: garch.cpp:445-460 — the ``mode != DoubleOptimization`` branch.
    """
    try:
        return _calibrate_r2_with_guess(
            r2, method, constraints, end_criteria, opt1 if f_cost1 <= f_cost2 else opt2
        )
    except Exception:  # C++ catches std::exception (garch.cpp:450)
        best = opt1 if f_cost1 <= f_cost2 else opt2
        return Garch11CalibrationResult(None, float(best[1]), float(best[2]), float(best[0]))


def _optimize_twice(
    r2: Sequence[float],
    method: OptimizationMethod,
    constraints: Constraint,
    end_criteria: EndCriteria,
    cost: _Garch11CostFunction,
    opt1: npt.NDArray[np.float64],
    opt2: npt.NDArray[np.float64],
    f_cost1: float,
    f_cost2: float,
) -> Garch11CalibrationResult:
    """Optimize from BOTH guesses and keep the cheaper result.

    # C++ parity: garch.cpp:461-498 — the ``DoubleOptimization`` branch.

    ``opt1`` and ``opt2`` are OVERWRITTEN in place with the optimized points,
    exactly as upstream does, and are read back at the end — which is why the
    comparison can score the original guess (when ``constraints.test`` fails)
    while returning the optimized point.
    """
    ret1: Problem | None = None
    ret2: Problem | None = None
    try:
        result1 = _calibrate_r2_with_guess(r2, method, constraints, end_criteria, opt1)
        ret1 = result1.problem
        opt1[1] = result1.alpha
        opt1[2] = result1.beta
        opt1[0] = result1.omega
        if constraints.test(opt1):
            f_cost1 = min(f_cost1, cost.value(opt1))
    except Exception:  # C++ catches std::exception (garch.cpp:471)
        f_cost1 = QL_MAX_REAL

    try:
        result2 = _calibrate_r2_with_guess(r2, method, constraints, end_criteria, opt2)
        ret2 = result2.problem
        opt2[1] = result2.alpha
        opt2[2] = result2.beta
        opt2[0] = result2.omega
        if constraints.test(opt2):
            f_cost2 = min(f_cost2, cost.value(opt2))
    except Exception:  # C++ catches std::exception (garch.cpp:483)
        f_cost2 = QL_MAX_REAL

    winner, problem = (opt1, ret1) if f_cost1 <= f_cost2 else (opt2, ret2)
    return Garch11CalibrationResult(
        problem, float(winner[1]), float(winner[2]), float(winner[0])
    )


class Garch11(VolatilityCompositor):
    """GARCH(1,1) volatility compositor.

    # C++ parity: ``class Garch11 : public VolatilityCompositor``
    # (ql/models/volatility/garch.{hpp,cpp}).

    Volatilities are expressed on an annual basis. See the module docstring
    for the meaning of the constructor arguments and for how the C++
    ``calibrate_r2`` overload set maps onto this class.
    """

    Mode: ClassVar[type[Garch11Mode]] = Garch11Mode
    """Alias so ``Garch11.Mode.BestOfTwo`` reads like C++ ``Garch11::BestOfTwo``."""

    __slots__ = ("_alpha", "_beta", "_gamma", "_log_likelihood", "_mode", "_vl")

    @overload
    def __init__(self, alpha_or_series: float, beta_or_mode: float, vl: float) -> None: ...

    @overload
    def __init__(
        self,
        alpha_or_series: TimeSeries[float],
        beta_or_mode: Garch11Mode = ...,
    ) -> None: ...

    def __init__(
        self,
        alpha_or_series: float | TimeSeries[float],
        beta_or_mode: float | Garch11Mode | None = None,
        vl: float | None = None,
    ) -> None:
        """Either fixed parameters ``(a, b, vl)`` or a series to calibrate to.

        # C++ parity: the two constructors at garch.hpp:56-63. ``vl`` is the
        # long-term VARIANCE, not omega.
        """
        if isinstance(alpha_or_series, TimeSeries):
            # C++ parity: garch.hpp:60-63 — gamma_ is left UNINITIALISED
            # until calibrate() sets it; 0.0 here is the Python stand-in.
            qassert.require(vl is None, "Garch11(series, mode) takes no third argument")
            self._alpha: float = 0.0
            self._beta: float = 0.0
            self._gamma: float = 0.0
            self._vl: float = 0.0
            self._log_likelihood: float = 0.0
            self._mode: Garch11Mode = (
                Garch11Mode.BestOfTwo if beta_or_mode is None else Garch11Mode(beta_or_mode)
            )
            self.calibrate(alpha_or_series)
        else:
            # C++ parity: garch.hpp:56-58.
            qassert.require(
                beta_or_mode is not None and vl is not None,
                "Garch11(alpha, beta, vl) needs all three parameters",
            )
            assert beta_or_mode is not None  # narrowing only
            assert vl is not None  # narrowing only
            self._alpha = float(alpha_or_series)
            self._beta = float(beta_or_mode)
            self._gamma = 1 - self._alpha - self._beta
            self._vl = vl
            self._log_likelihood = 0.0
            self._mode = Garch11Mode.BestOfTwo

    # --- inspectors ---------------------------------------------------------

    def alpha(self) -> float:
        """# C++ parity: garch.hpp:68."""
        return self._alpha

    def beta(self) -> float:
        """# C++ parity: garch.hpp:69."""
        return self._beta

    def omega(self) -> float:
        """Constant term ``vl * (1 - alpha - beta)``. # C++ parity: garch.hpp:70."""
        return self._vl * self._gamma

    def lt_vol(self) -> float:
        """Long-term variance level. # C++ parity: ``ltVol()``, garch.hpp:71."""
        return self._vl

    def log_likelihood(self) -> float:
        """# C++ parity: ``logLikelihood()``, garch.hpp:72."""
        return self._log_likelihood

    def mode(self) -> Garch11Mode:
        """# C++ parity: garch.hpp:73."""
        return self._mode

    # --- VolatilityCompositor interface -------------------------------------

    def calculate(self, volatility_series: TimeSeries[float]) -> TimeSeries[float]:
        """Conditional volatility series under the current parameters.

        # C++ parity: garch.hpp:78-80 — delegates to the static overload with
        # ``alpha()``, ``beta()``, ``omega()``.
        """
        return Garch11.calculate_series(volatility_series, self.alpha(), self.beta(), self.omega())

    def calibrate(
        self,
        volatility_series: TimeSeries[float] | Sequence[float],
        method: OptimizationMethod | None = None,
        end_criteria: EndCriteria | None = None,
        initial_guess: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Fit ``alpha``, ``beta`` and ``omega`` by maximum likelihood.

        # C++ parity: the ``calibrate`` family at garch.hpp:81-150. The three
        # ``time_series`` overloads and the three ``ForwardIterator``
        # templates collapse into this one method: a ``TimeSeries`` argument
        # is reduced to its ``values()`` (garch.hpp:82-83), and a plain
        # sequence stands in for the iterator pair.

        With ``initial_guess`` the mode is bypassed entirely and the
        optimizer starts there (garch.hpp:136-150); without it the mode
        selects the guess(es) as described in the module docstring.
        """
        values = (
            list(volatility_series.values())
            if isinstance(volatility_series, TimeSeries)
            else list(volatility_series)
        )
        r2, mean_r2 = Garch11.to_r2(values)
        if initial_guess is not None:
            qassert.require(
                method is not None and end_criteria is not None,
                "calibrate() with an initial guess also needs a method and end criteria",
            )
            result = Garch11.calibrate_r2(
                r2, method=method, end_criteria=end_criteria, initial_guess=initial_guess
            )
        else:
            qassert.require(
                (method is None) == (end_criteria is None),
                "calibrate() takes a method and end criteria together or not at all",
            )
            result = Garch11.calibrate_r2(
                r2, mode=self._mode, mean_r2=mean_r2, method=method, end_criteria=end_criteria
            )
        # C++ parity: garch.hpp:114-118 — note vl_ receives OMEGA from
        # calibrate_r2 and is only then divided by gamma, so omega() round-trips.
        self._alpha = result.alpha
        self._beta = result.beta
        self._vl = result.omega
        self._gamma = 1 - self._alpha - self._beta
        self._vl /= self._gamma
        self._log_likelihood = (
            -result.problem.function_value
            if result.problem is not None
            else -Garch11.cost_function(values, self.alpha(), self.beta(), self.omega())
        )

    # --- additional interface -----------------------------------------------

    def forecast(self, r: float, sigma2: float) -> float:
        """One-step-ahead conditional variance.

        # C++ parity: garch.hpp:152-154 —
        # ``gamma*vl + alpha*r^2 + beta*sigma2``.
        """
        return self._gamma * self._vl + self._alpha * r * r + self._beta * sigma2

    @staticmethod
    def calculate_series(
        quote_series: TimeSeries[float], alpha: float, beta: float, omega: float
    ) -> TimeSeries[float]:
        """Conditional volatility series under explicit parameters.

        # C++ parity: ``static time_series Garch11::calculate(
        # const time_series&, Real alpha, Real beta, Real omega)``
        # (garch.hpp:89-90, garch.cpp:373-390).

        # C++ parity divergence — the NAME. C++ overloads ``calculate`` on
        # arity, one virtual and one static; Python cannot bind one name to
        # both, so the static form is ``calculate_series``.

        The output has ONE MORE point than the input: the last variance is
        filed under a SYNTHETIC date obtained by repeating the final date gap
        (garch.cpp:386-388). The first input date carries no output, so the
        net effect is a one-step shift forward.
        """
        items = quote_series.items()
        qassert.require(
            len(items) >= 2,
            "Garch11.calculate needs at least two observations to extrapolate the final date",
        )
        retval: TimeSeries[float] = TimeSeries()
        u = items[0][1]
        sigma2 = u * u
        for i in range(1, len(items)):
            date, value = items[i]
            sigma2 = omega + alpha * u * u + beta * sigma2
            retval[date] = math.sqrt(sigma2)
            u = value
        sigma2 = omega + alpha * u * u + beta * sigma2
        last = items[-1][0]
        prev = items[-2][0]
        retval[last + (last - prev)] = math.sqrt(sigma2)
        return retval

    @staticmethod
    def to_r2(values: Sequence[float]) -> tuple[list[float], float]:
        """Squared returns and their running mean.

        # C++ parity: ``static Real to_r2(begin, end, std::vector<Volatility>&)``
        # (garch.hpp:157-168); C++ returns the mean and fills the vector,
        # Python returns ``(r2, mean_r2)``.

        The mean is accumulated with a weight that decays as
        ``w /= (w + 1)`` from 1, i.e. ``w_n = 1/n`` — an incremental arithmetic
        mean, NOT an exponentially weighted one.
        """
        r2: list[float] = []
        mean_r2 = 0.0
        w = 1.0
        for value in values:
            u2 = value * value
            mean_r2 = (1.0 - w) * mean_r2 + w * u2
            r2.append(u2)
            w /= w + 1.0
        return r2, mean_r2

    @staticmethod
    def cost_function(values: Sequence[float], alpha: float, beta: float, omega: float) -> float:
        """Half the mean negative log-likelihood of ``values`` (NOT of ``r2``).

        # C++ parity: ``static Real costFunction(begin, end, alpha, beta, omega)``
        # (garch.hpp:237-249). Note this squares each element itself, so it
        # takes the RETURN series, whereas ``_Garch11CostFunction`` takes the
        # already-squared one.
        """
        retval = 0.0
        u2 = 0.0
        sigma2 = 0.0
        n = 0
        for value in values:
            sigma2 = omega + alpha * u2 + beta * sigma2
            u2 = value * value
            retval += math.log(sigma2) + u2 / sigma2
            n += 1
        return retval / (2 * n) if n > 0 else 0.0

    @staticmethod
    def calibrate_r2(
        r2: Sequence[float],
        *,
        mode: Garch11Mode | None = None,
        mean_r2: float | None = None,
        method: OptimizationMethod | None = None,
        end_criteria: EndCriteria | None = None,
        constraints: Constraint | None = None,
        initial_guess: npt.NDArray[np.float64] | None = None,
    ) -> Garch11CalibrationResult:
        """Calibrate GARCH(1,1) against a series of SQUARED returns.

        # C++ parity: the six ``calibrate_r2`` overloads at garch.hpp:171-235,
        # merged into one keyword-only signature — see the module docstring
        # for the overload-to-keyword mapping.

        With ``initial_guess`` the optimizer starts there and ``mode`` is
        irrelevant; ``constraints`` then defaults to the GARCH stationarity
        box ``(0, 1 - 1e-8)`` (garch.cpp:507) and ``mean_r2``, if given,
        centers ``r2`` first (garch.cpp:518-521).

        Without ``initial_guess`` both ``mode`` and ``mean_r2`` are required,
        and ``method`` / ``end_criteria`` default to ``Simplex(0.001)`` and
        ``EndCriteria(10000, 500, 1e-8, 1e-8, 1e-8)`` (garch.cpp:396-397).
        """
        if initial_guess is not None:
            qassert.require(
                method is not None and end_criteria is not None,
                "calibrate_r2 with an initial guess also needs a method and end criteria",
            )
            assert method is not None  # narrowing only
            assert end_criteria is not None  # narrowing only
            series = list(r2) if mean_r2 is None else [x - mean_r2 for x in r2]
            box = constraints if constraints is not None else _Garch11Constraint(0.0, 1.0 - _TOL_LEVEL)
            return _calibrate_r2_with_guess(series, method, box, end_criteria, initial_guess)

        qassert.require(
            mode is not None and mean_r2 is not None,
            "calibrate_r2 without an initial guess needs both a mode and mean_r2",
        )
        assert mode is not None  # narrowing only
        assert mean_r2 is not None  # narrowing only
        qassert.require(
            constraints is None,
            "no C++ calibrate_r2 overload takes a mode and an explicit constraint",
        )
        qassert.require(
            (method is None) == (end_criteria is None),
            "calibrate_r2 takes a method and end criteria together or not at all",
        )
        # C++ parity: garch.cpp:393-400.
        chosen_method = method if method is not None else Simplex(0.001)
        chosen_criteria = (
            end_criteria
            if end_criteria is not None
            else EndCriteria(10000, 500, _TOL_LEVEL, _TOL_LEVEL, _TOL_LEVEL)
        )
        return _calibrate_r2_from_mode(mode, r2, mean_r2, chosen_method, chosen_criteria)


__all__ = ["Garch11", "Garch11CalibrationResult", "Garch11Mode"]
