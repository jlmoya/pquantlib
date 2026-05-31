"""NormalCLVModel — Collocating Local Volatility with a normal kernel.

# C++ parity: ql/experimental/models/normalclvmodel.{hpp,cpp} (v1.42.1).

The Collocating Local Volatility (CLV) framework of A. Grzelak (2016,
"The CLV Framework - A Fresh Look at Efficient Pricing with Smile",
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2747541) maps a
tractable "kernel" process onto the market-implied terminal
distribution of an underlying via a collocation polynomial. The
*normal* CLV uses an Ornstein-Uhlenbeck kernel.

For each maturity, ``collocationPointsX`` are the OU collocation nodes
(``E[X] + std[X] * x_i`` for the Gauss-Hermite abscissae ``x_i``), and
``collocationPointsY`` are the corresponding underlying values obtained
by mapping the kernel CDF through the market inverse-CDF (via
``GBSMRNDCalculator``). The mapping function ``g(t, x)`` interpolates
those collocation y-points (linearly across maturities, Lagrange across
the node dimension) so that a path of the OU kernel maps to a
distribution consistent with the market smile.

The Gauss-Hermite abscissae use ``sqrt(2) * H_n`` where ``H_n`` are the
physicists' Hermite roots (``scipy.special.roots_hermite``) — exactly
the C++ ``M_SQRT2 * GaussHermiteIntegration(n).x()``. Delegating the
quadrature nodes to scipy is consistent with the project's
numerical-tooling policy (docs/carve-outs.md Category 3).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

import numpy as np
from scipy.special import roots_hermite  # pyright: ignore[reportMissingTypeStubs]

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.interpolations.lagrange_interpolation import LagrangeInterpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.methods.finitedifferences.utilities.gbsm_rnd_calculator import (
    GBSMRNDCalculator,
)
from pquantlib.patterns.lazy_object import LazyObject

if TYPE_CHECKING:
    from collections.abc import Callable

    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
    from pquantlib.time.date import Date

_N = CumulativeNormalDistribution()
_INV_CUM_NORMAL = InverseCumulativeNormal()


def _gauss_hermite_x(order: int) -> Array:
    """``sqrt(2) * H_n`` physicists' Hermite abscissae (descending order).

    # C++ parity: ``M_SQRT2 * GaussHermiteIntegration(order).x()``. The
    # C++ Gauss-Hermite nodes come out in descending order (largest
    # first); scipy returns ascending, so we reverse to match the C++
    # collocation-point ordering exactly.
    """
    nodes, _ = roots_hermite(order)
    return math.sqrt(2.0) * np.asarray(nodes[::-1], dtype=np.float64)


@final
class NormalCLVModel(LazyObject):
    """Normal (OU-kernel) Collocating Local Volatility model.

    # C++ parity: ``class NormalCLVModel : public LazyObject`` in
    # normalclvmodel.hpp:48-115.
    """

    __slots__ = (
        "_bs_process",
        "_g",
        "_maturity_dates",
        "_maturity_times",
        "_ou_process",
        "_rnd_calculator",
        "_sigma",
        "_x",
    )

    def __init__(
        self,
        bs_process: GeneralizedBlackScholesProcess,
        ou_process: OrnsteinUhlenbeckProcess,
        maturity_dates: list[Date],
        lagrange_order: int,
        p_max: float | None = None,
        p_min: float | None = None,
    ) -> None:
        # C++ parity: normalclvmodel.cpp:39-61.
        super().__init__()
        self._x: Array = _gauss_hermite_x(lagrange_order)
        if p_max is not None:
            self._sigma: float = float(self._x[-1]) / _INV_CUM_NORMAL(p_max)
        elif p_min is not None:
            self._sigma = float(self._x[0]) / _INV_CUM_NORMAL(p_min)
        else:
            self._sigma = 1.0
        self._bs_process: GeneralizedBlackScholesProcess = bs_process
        self._ou_process: OrnsteinUhlenbeckProcess = ou_process
        self._maturity_dates: list[Date] = list(maturity_dates)
        self._rnd_calculator: GBSMRNDCalculator = GBSMRNDCalculator(bs_process)
        self._maturity_times: list[float] = [
            bs_process.time(d) for d in maturity_dates
        ]
        for i in range(1, len(self._maturity_times)):
            qassert.require(
                self._maturity_times[i - 1] < self._maturity_times[i],
                "dates must be sorted",
            )
        bs_process.register_with(self)
        ou_process.register_with(self)
        self._g: Callable[[float, float], float] | None = None

    # --- distribution passthroughs --------------------------------------

    def cdf(self, d: Date, k: float) -> float:
        # C++ parity: normalclvmodel.cpp:63-65.
        return self._rnd_calculator.cdf(k, self._bs_process.time(d))

    def inv_cdf(self, d: Date, q: float) -> float:
        # C++ parity: normalclvmodel.cpp:68-70.
        return self._rnd_calculator.invcdf(q, self._bs_process.time(d))

    # --- collocation points ---------------------------------------------

    def collocation_points_x(self, d: Date) -> Array:
        # C++ parity: normalclvmodel.cpp:72-81.
        t = self._bs_process.time(d)
        x0 = self._ou_process.x0()
        expectation = self._ou_process.expectation_1d(0.0, x0, t)
        std_dev = self._ou_process.std_deviation_1d(0.0, x0, t)
        return expectation + std_dev * self._x

    def collocation_points_y(self, d: Date) -> Array:
        # C++ parity: normalclvmodel.cpp:83-92.
        s = np.empty(self._x.shape[0], dtype=np.float64)
        for i in range(s.shape[0]):
            s[i] = self.inv_cdf(d, _N(float(self._x[i]) / self._sigma))
        return s

    # --- mapping function -----------------------------------------------

    def g(self) -> Callable[[float, float], float]:
        # C++ parity: normalclvmodel.cpp:95-98.
        self.calculate()
        assert self._g is not None
        return self._g

    def _perform_calculations(self) -> None:
        # C++ parity: normalclvmodel.cpp:134-136. Gather the collocation
        # y-columns here (via the model's own public methods) and hand the
        # plain data to the mapping function — keeping the helper decoupled
        # from the model's private state.
        n_nodes = self._x.shape[0]
        n_mat = len(self._maturity_dates)
        s = np.empty((n_nodes, n_mat), dtype=np.float64)
        for j in range(n_mat):
            s[:, j] = self.collocation_points_y(self._maturity_dates[j])
        self._g = _MappingFunction(
            sigma=self._sigma,
            ou_process=self._ou_process,
            x=self._x,
            maturity_times=self._maturity_times,
            collocation_y=s,
        )


class _MappingFunction:
    """CLV mapping function g(t, x).

    # C++ parity: ``NormalCLVModel::MappingFunction`` in
    # normalclvmodel.hpp:76-103 + normalclvmodel.cpp:100-132.

    Holds, per collocation node, a ``LinearInterpolation`` of the
    underlying collocation y-values across maturities; a single
    ``LagrangeInterpolation`` over the kernel node abscissae provides the
    polynomial mapping at a given maturity.
    """

    __slots__ = ("_interpl", "_lagrange", "_ou_process", "_sigma", "_x", "_y")

    def __init__(
        self,
        *,
        sigma: float,
        ou_process: OrnsteinUhlenbeckProcess,
        x: Array,
        maturity_times: list[float],
        collocation_y: Array,
    ) -> None:
        self._sigma: float = sigma
        self._ou_process: OrnsteinUhlenbeckProcess = ou_process
        self._x: Array = x
        self._y: Array = np.empty(x.shape[0], dtype=np.float64)
        # Lagrange interpolation over (x_, x_) — node abscissae are x_; the
        # y-vector is supplied fresh per call via value_with.
        self._lagrange: LagrangeInterpolation = LagrangeInterpolation(x, x)
        # One time-interpolation per node row (rows of collocation_y).
        t_arr = np.asarray(maturity_times, dtype=np.float64)
        n_nodes = x.shape[0]
        self._interpl: list[LinearInterpolation] = [
            LinearInterpolation(t_arr, collocation_y[i, :]) for i in range(n_nodes)
        ]

    def __call__(self, t: float, x: float) -> float:
        # C++ parity: normalclvmodel.cpp:119-132.
        for i in range(self._y.shape[0]):
            self._y[i] = self._interpl[i](t, allow_extrapolation=True)
        x0 = self._ou_process.x0()
        expectation = self._ou_process.expectation_1d(0.0, x0, t)
        std_dev = self._ou_process.std_deviation_1d(0.0, x0, t)
        r = self._sigma * (x - expectation) / std_dev
        return self._lagrange.value_with(self._y, r)


__all__ = ["NormalCLVModel"]
