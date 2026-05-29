"""ContinuousArithmeticAsianVecerEngine — Vecer 2001 PDE engine.

# C++ parity:
# ql/experimental/exoticoptions/continuousarithmeticasianvecerengine.{hpp,cpp}
# (v1.42.1).

Vecer 2001 PDE for continuously-averaged arithmetic Asian options.
Reduces the 2-dimensional PDE (S, A) to a single-state PDE in
``z = A/(S*scale)`` by changing variables; then time-marches a
Crank-Nicolson scheme with Neumann (delta=1) boundary at the upper
edge and Dirichlet (=0) at the lower edge.

Reference: J. Vecer, "A new PDE approach for pricing arithmetic
average Asian options", Journal of Computational Finance 4(4), 2001.

Divergences from C++:

* The C++ engine uses ``QuantLib::TridiagonalOperator``;
  the Python port uses ``scipy.linalg.solve_banded`` for the
  implicit step. The explicit step is a direct numpy matvec on a
  tridiagonal structure.
* Boundary conditions are applied identically: u[0] = 0 (Dirichlet
  at lower edge), u[N] = u[N-1] + h (Neumann delta = 1 at upper
  edge).

Limitations (mirrors C++):

* Only unseasoned options (``start_date >= today``) are supported;
  the seasoned branch raises.
* Only ``Arithmetic`` averaging + European exercise.
* Call-Put parity is applied after the call price is computed.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_banded  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOptionArguments,
)
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.quote import Quote
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

_VECER_EPS = 1e-5


def _cont_strategy(t: float, t1: float, t2: float, v: float, r: float) -> float:
    """Replication of running-average forward by holding ``cont_strategy(t)`` assets.

    # C++ parity:
    # ``ContinuousArithmeticAsianVecerEngine::cont_strategy(t, T1, T2, v, r)``.
    """
    qassert.require(t1 <= t2, "Average Start must be before Average End")
    if abs(t - t2) < _VECER_EPS:
        return 0.0
    if t < t1:
        if abs(r - v) >= _VECER_EPS:
            return (
                math.exp(v * (t - t2))
                * (1.0 - math.exp((v - r) * (t2 - t1)))
                / ((r - v) * (t2 - t1))
            )
        return math.exp(v * (t - t2))
    if abs(r - v) >= _VECER_EPS:
        return (
            math.exp(v * (t - t2))
            * (1.0 - math.exp((v - r) * (t2 - t)))
            / ((r - v) * (t2 - t1))
        )
    return math.exp(v * (t - t2)) * (t2 - t) / (t2 - t1)


class ContinuousArithmeticAsianVecerEngine(
    GenericEngine[ContinuousAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Vecer 2001 PDE engine for continuously-averaged arithmetic Asians.

    # C++ parity: ``ContinuousArithmeticAsianVecerEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        current_average: Quote,
        start_date: Date,
        time_steps: int = 100,
        asset_steps: int = 100,
        z_min: float = -1.0,
        z_max: float = 1.0,
    ) -> None:
        super().__init__(
            ContinuousAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._current_average: Quote = current_average
        self._start_date: Date = start_date
        self._z_min: float = z_min
        self._z_max: float = z_max
        self._time_steps: int = time_steps
        self._asset_steps: int = asset_steps
        process.register_with(self)
        current_average.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915  (faithful C++ port — Vecer PDE engine is one long routine)
        """Run the Vecer PDE.

        # C++ parity:
        # ``ContinuousArithmeticAsianVecerEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(
            args.average_type == AverageType.Arithmetic,
            "not an Arithmetic average option",
        )
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )

        rfdc = self._process.risk_free_rate().day_counter()
        s_0 = self._process.state_variable().value()

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, StrikedTypePayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, StrikedTypePayoff)

        maturity = args.exercise.last_date()
        x_strike = payoff.strike()
        qassert.require(
            self._z_min <= 0 and self._z_max >= 0,
            "strike (0 for vecer fixed strike asian)  not on Grid",
        )

        sigma = self._process.black_volatility().black_vol(
            maturity, x_strike, extrapolate=True
        )
        r = self._process.risk_free_rate().zero_rate(
            maturity, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        q = self._process.dividend_yield().zero_rate(
            maturity, Compounding.Continuous, Frequency.NoFrequency
        ).rate()

        today = self._process.risk_free_rate().reference_date()
        qassert.require(
            self._start_date >= today, "Seasoned Asian not yet implemented"
        )

        t = rfdc.year_fraction(today, maturity)
        t1 = rfdc.year_fraction(today, self._start_date)  # Average Begin
        t2 = t  # Average End (= Maturity in this version)

        results.reset()

        if (t2 - t1) < 0.001:
            # Vanilla short-circuit: degenerate to a European option.
            european_option = EuropeanOption(payoff, args.exercise)
            european_option.set_pricing_engine(AnalyticEuropeanEngine(self._process))
            results.value = european_option.npv()
            return

        theta = 0.5  # Crank-Nicolson.
        z_0 = (
            _cont_strategy(0.0, t1, t2, q, r) - math.exp(-r * t) * x_strike / s_0
        )

        qassert.require(
            self._z_min <= z_0 <= self._z_max, "spot not on grid"
        )

        n = self._asset_steps
        h = (self._z_max - self._z_min) / float(n)  # space step
        k = t / float(self._time_steps)  # time step

        sigma2 = sigma * sigma

        s_vec = np.array(
            [self._z_min + i * h for i in range(n + 1)], dtype=np.float64
        )

        # Initial condition: u(z, T) = max(z, 0) (call payoff in
        # the transformed variable; put recovered via parity).
        u: npt.NDArray[np.float64] = np.maximum(s_vec, 0.0)

        # Time loop (backwards in C++ but the C++ engine indexes
        # j=1..M corresponding to t = T - j*k).
        for j in range(1, self._time_steps + 1):
            # --- explicit part: u <- (I + (1-theta)*k*L) u with
            #     L = 0.5 * sigma^2 * vecer_term^2 * d2/dz2 (centred,
            #     coeff 1/h^2) plus Dirichlet/Neumann at the edges.
            if theta != 1.0:
                u = self._explicit_step(u, s_vec, h, k, theta, sigma2, q, r, t, t1, t2, j - 1, n)
            # --- implicit part: solve (I - theta*k*L) u_new = rhs
            if theta != 0.0:
                u = self._implicit_step(u, s_vec, h, k, theta, sigma2, q, r, t, t1, t2, j, n)

        # Linear interpolate at z_0.
        lower_i = math.floor((z_0 - self._z_min) / h)
        pv = u[lower_i] + (u[lower_i + 1] - u[lower_i]) * (z_0 - s_vec[lower_i]) / h

        results.value = s_0 * pv

        if payoff.option_type() == OptionType.Put:
            # Apply call-put parity for Asians.
            if r == q:
                expected_average = s_0
            else:
                expected_average = (
                    s_0
                    * (math.exp((r - q) * t2) - math.exp((r - q) * t1))
                    / ((r - q) * (t2 - t1))
                )
            asian_forward = math.exp(-r * t2) * (expected_average - x_strike)
            assert results.value is not None
            results.value = results.value - asian_forward

    @staticmethod
    def _explicit_step(
        u: npt.NDArray[np.float64],
        s_vec: npt.NDArray[np.float64],
        h: float,
        k: float,
        theta: float,
        sigma2: float,
        q: float,
        r: float,
        t: float,
        t1: float,
        t2: float,
        j_idx: int,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Explicit Crank-Nicolson half-step.

        # C++ parity: anonymous explicit_part TridiagonalOperator
        # block inside ``calculate``.
        """
        # Lower / diag / upper of the implicit-scaling matrix L:
        #   diag[i] = -2/h^2 * coeff[i]
        #   off[i]  = 1/h^2 * coeff[i]
        # where coeff[i] = 0.5 * sigma^2 * vecer_term^2.
        coeff = np.zeros_like(u)
        for i in range(1, n):
            vt = s_vec[i] - math.exp(-q * (t - j_idx * k)) * _cont_strategy(
                t - j_idx * k, t1, t2, q, r
            )
            coeff[i] = 0.5 * sigma2 * vt * vt

        # (I + (1-theta)*k * L) u, with row 0 = identity, row N = -1, +1.
        new_u = u.copy()
        scale = (1.0 - theta) * k / (h * h)
        for i in range(1, n):
            new_u[i] = u[i] + scale * coeff[i] * (u[i + 1] - 2.0 * u[i] + u[i - 1])
        # First row identity → new_u[0] = u[0]; lock to Dirichlet=0 below.
        new_u[0] = 0.0
        # Last row: applyTo with (-1, 1, 0) on the last row of (1-theta)*k*L
        # but the C++ also lock the Neumann BC after applying:
        new_u[n] = new_u[n - 1] + h
        return new_u

    @staticmethod
    def _implicit_step(
        u: npt.NDArray[np.float64],
        s_vec: npt.NDArray[np.float64],
        h: float,
        k: float,
        theta: float,
        sigma2: float,
        q: float,
        r: float,
        t: float,
        t1: float,
        t2: float,
        j_idx: int,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Implicit Crank-Nicolson half-step.

        # C++ parity: anonymous implicit_part TridiagonalOperator
        # block + solveFor inside ``calculate``.
        """
        # Assemble banded form for solve_banded(l=1, u=1).
        ab = np.zeros((3, n + 1), dtype=np.float64)
        rhs = u.copy()
        rhs[0] = 0.0  # Lower BC.
        rhs[n] = h  # Upper BC (Neumann delta=1).

        coeff_scale = theta * k / (h * h)
        for i in range(1, n):
            vt = s_vec[i] - math.exp(-q * (t - j_idx * k)) * _cont_strategy(
                t - j_idx * k, t1, t2, q, r
            )
            coeff = 0.5 * sigma2 * vt * vt
            # Tridiagonal row of (I - theta*k*L):
            #   -coeff*1/h^2 (lower) | 1 + 2*coeff/h^2 (diag) | -coeff/h^2 (upper)
            ab[0, i + 1] = -coeff_scale * coeff  # superdiag entry at column i+1
            ab[1, i] = 1.0 + 2.0 * coeff_scale * coeff  # diag at column i
            ab[2, i - 1] = -coeff_scale * coeff  # subdiag entry at column i-1
        # First row: 1.0 * u[0] = rhs[0] = 0.
        ab[1, 0] = 1.0
        # Last row: -u[N-1] + u[N] = rhs[N] = h (Neumann).
        ab[1, n] = 1.0
        ab[2, n - 1] = -1.0

        # scipy.linalg.solve_banded handles tridiagonal systems.
        result = cast("Any", solve_banded((1, 1), ab, rhs))
        return np.asarray(result, dtype=np.float64)

    def update(self) -> None:
        self.notify_observers()


__all__ = ["ContinuousArithmeticAsianVecerEngine"]
