"""SquareRootCLVModel — Collocating Local Volatility with a CIR kernel.

# C++ parity: ql/experimental/models/squarerootclvmodel.{hpp,cpp} (v1.42.1).

The square-root flavour of the CLV framework (Grzelak 2016) collocates a
square-root (Cox-Ingersoll-Ross) kernel onto the market terminal
distribution. The kernel's terminal law is a (scaled) non-central
chi-squared with

    df  = 4 theta kappa / sigma^2
    ncp = 4 kappa exp(-kappa t) / (sigma^2 (1 - exp(-kappa t))) x0

(``kappa = a``, ``theta = b``, ``sigma`` of the ``SquareRootProcess``).

``collocationPointsX`` are the non-central-chi-squared Gauss-quadrature
nodes (Golub-Welsch via the moment-based orthogonal polynomial),
optionally clamped to the ``[pMin, pMax]`` quantile window and rescaled.
``collocationPointsY`` map those through the chi-squared CDF and the
market inverse-CDF. The mapping ``g(t, x)`` Lagrange-interpolates the
per-maturity collocation pairs, linearly interpolating across maturities.

Delegations (docs/carve-outs.md Category 3): the non-central
chi-squared quantile uses ``scipy.stats.ncx2.ppf`` (the C++ used
``boost::math::quantile``); the CDF uses the ported
``NonCentralCumulativeChiSquareDistribution``; the Gauss-quadrature
eigenproblem uses ``GaussianQuadrature`` (scipy ``eigh_tridiagonal``).
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, final

import numpy as np
from scipy.stats import ncx2  # pyright: ignore[reportMissingTypeStubs]

from pquantlib import qassert
from pquantlib.experimental.math.gaussian_noncentral_chisquared_polynomial import (
    GaussNonCentralChiSquaredPolynomial,
)
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.math.distributions.non_central_chi_square_distribution import (
    NonCentralCumulativeChiSquareDistribution,
)
from pquantlib.math.integrals.gaussian_quadrature import GaussianQuadrature
from pquantlib.math.interpolations.lagrange_interpolation import LagrangeInterpolation
from pquantlib.methods.finitedifferences.utilities.gbsm_rnd_calculator import (
    GBSMRNDCalculator,
)
from pquantlib.patterns.lazy_object import LazyObject

if TYPE_CHECKING:
    from collections.abc import Callable

    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib.processes.square_root_process import SquareRootProcess
    from pquantlib.time.date import Date

_QL_MAX_REAL = sys.float_info.max


@final
class SquareRootCLVModel(LazyObject):
    """Square-root (CIR-kernel) Collocating Local Volatility model.

    # C++ parity: ``class SquareRootCLVModel : public LazyObject`` in
    # squarerootclvmodel.hpp:41-93.
    """

    __slots__ = (
        "_bs_process",
        "_g",
        "_lagrange_order",
        "_maturity_dates",
        "_p_max",
        "_p_min",
        "_rnd_calculator",
        "_sqrt_process",
    )

    def __init__(
        self,
        bs_process: GeneralizedBlackScholesProcess,
        sqrt_process: SquareRootProcess,
        maturity_dates: list[Date],
        lagrange_order: int,
        p_max: float | None = None,
        p_min: float | None = None,
    ) -> None:
        # C++ parity: squarerootclvmodel.cpp:36-45.
        super().__init__()
        self._p_max: float | None = p_max
        self._p_min: float | None = p_min
        self._bs_process: GeneralizedBlackScholesProcess = bs_process
        self._sqrt_process: SquareRootProcess = sqrt_process
        self._maturity_dates: list[Date] = list(maturity_dates)
        self._lagrange_order: int = lagrange_order
        self._rnd_calculator: GBSMRNDCalculator = GBSMRNDCalculator(bs_process)
        self._g: Callable[[float, float], float] | None = None

    # --- distribution passthroughs --------------------------------------

    def cdf(self, d: Date, k: float) -> float:
        # C++ parity: squarerootclvmodel.cpp:47-49.
        return self._rnd_calculator.cdf(k, self._bs_process.time(d))

    def inv_cdf(self, d: Date, q: float) -> float:
        # C++ parity: squarerootclvmodel.cpp:52-54.
        return self._rnd_calculator.invcdf(q, self._bs_process.time(d))

    # --- chi-squared kernel parameters ----------------------------------

    def _non_central_chi_squared_params(self, d: Date) -> tuple[float, float]:
        # C++ parity: squarerootclvmodel.cpp:56-70.
        t = self._bs_process.time(d)
        kappa = self._sqrt_process.a()
        theta = self._sqrt_process.b()
        sigma = self._sqrt_process.sigma()
        df = 4.0 * theta * kappa / (sigma * sigma)
        ncp = (
            4.0
            * kappa
            * math.exp(-kappa * t)
            / (sigma * sigma * (1.0 - math.exp(-kappa * t)))
            * self._sqrt_process.x0()
        )
        return df, ncp

    # --- collocation points ---------------------------------------------

    def collocation_points_x(self, d: Date) -> Array:
        # C++ parity: squarerootclvmodel.cpp:73-102.
        df, ncp = self._non_central_chi_squared_params(d)
        x = GaussianQuadrature(
            self._lagrange_order, GaussNonCentralChiSquaredPolynomial(df, ncp)
        ).x()
        x = np.sort(x)

        x_min = max(
            float(x[0]),
            0.0
            if self._p_min is None
            else float(ncx2.ppf(self._p_min, df, ncp)),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        x_max = min(
            float(x[-1]),
            _QL_MAX_REAL
            if self._p_max is None
            else float(ncx2.ppf(self._p_max, df, ncp)),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )

        b = x_min - float(x[0])
        a = (x_max - x_min) / (float(x[-1]) - float(x[0]))
        return a * x + b

    def collocation_points_y(self, d: Date) -> Array:
        # C++ parity: squarerootclvmodel.cpp:104-119.
        x = self.collocation_points_x(d)
        df, ncp = self._non_central_chi_squared_params(d)
        dist = NonCentralCumulativeChiSquareDistribution(df, ncp)
        s = np.empty(x.shape[0], dtype=np.float64)
        for i in range(s.shape[0]):
            q = dist(float(x[i]))
            s[i] = self.inv_cdf(d, q)
        return s

    # --- mapping function -----------------------------------------------

    def g(self) -> Callable[[float, float], float]:
        # C++ parity: squarerootclvmodel.cpp:121-124.
        self.calculate()
        assert self._g is not None
        return self._g

    def _perform_calculations(self) -> None:
        # C++ parity: squarerootclvmodel.cpp:126-128. Build the per-maturity
        # (time, Lagrange) pairs here using the model's own public methods,
        # then hand plain data to the mapping function.
        maturity_dates = sorted(
            self._maturity_dates, key=lambda dt: dt.serial_number()
        )
        times: list[float] = []
        interpl: list[LagrangeInterpolation] = []
        for d in maturity_dates:
            x = self.collocation_points_x(d)
            y = self.collocation_points_y(d)
            times.append(self._bs_process.time(d))
            interpl.append(LagrangeInterpolation(x, y))
        self._g = _SquareRootMappingFunction(times=times, interpl=interpl)


class _SquareRootMappingFunction:
    """CLV mapping g(t, x) for the square-root kernel.

    # C++ parity: ``SquareRootCLVModel::MappingFunction`` in
    # squarerootclvmodel.hpp:69-81 + squarerootclvmodel.cpp:130-182.

    Per maturity, a ``LagrangeInterpolation`` over the (x, y) collocation
    pair. Between maturities, linear interpolation in t (with no
    extrapolation beyond the maturity span — matching C++).
    """

    __slots__ = ("_interpl", "_times")

    def __init__(
        self,
        *,
        times: list[float],
        interpl: list[LagrangeInterpolation],
    ) -> None:
        self._times: list[float] = times
        self._interpl: list[LagrangeInterpolation] = interpl

    def __call__(self, t: float, x: float) -> float:
        # C++ parity: squarerootclvmodel.cpp:162-182 — lower_bound on the
        # maturity map, exact-hit or linear interpolation between brackets.
        times = self._times
        # lower_bound: first maturity time >= t.
        idx = 0
        while idx < len(times) and times[idx] < t and not close_enough(times[idx], t):
            idx += 1

        if idx < len(times) and close_enough(times[idx], t):
            return self._interpl[idx](x, allow_extrapolation=True)

        qassert.require(
            0 < idx < len(times),
            "extrapolation to large or small t is not allowed",
        )

        t1 = times[idx]
        y1 = self._interpl[idx](x, allow_extrapolation=True)
        t0 = times[idx - 1]
        y0 = self._interpl[idx - 1](x, allow_extrapolation=True)
        return y0 + (y1 - y0) / (t1 - t0) * (t - t0)


__all__ = ["SquareRootCLVModel"]
