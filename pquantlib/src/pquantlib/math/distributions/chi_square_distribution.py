"""Chi-square distributions — central CDF, Sankaran approximation, inverse.

# C++ parity: ql/math/distributions/chisquaredistribution.{hpp,cpp} (v1.43).

The fourth class declared in the same C++ header,
``NonCentralCumulativeChiSquareDistribution``, lives in
:mod:`pquantlib.math.distributions.non_central_chi_square_distribution` and is
re-exported here so the C++ header maps onto one importable surface.
"""

from __future__ import annotations

import math

from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.gamma_distribution import CumulativeGammaDistribution
from pquantlib.math.distributions.non_central_chi_square_distribution import (
    NonCentralCumulativeChiSquareDistribution,
)
from pquantlib.math.solvers1d.brent import Brent


class CumulativeChiSquareDistribution:
    """Central chi-square CDF with ``df`` degrees of freedom.

    # C++ parity: ``class CumulativeChiSquareDistribution`` —
    # chisquaredistribution.hpp:32-39, chisquaredistribution.cpp:30-32.

    Defined *through* :class:`CumulativeGammaDistribution`, so it inherits
    that routine's 3e-7 series tolerance rather than being an independent
    evaluation of the regularized incomplete Gamma.
    """

    __slots__ = ("_df",)

    def __init__(self, df: float) -> None:
        self._df: float = float(df)

    def __call__(self, x: float) -> float:
        # C++ parity: chisquaredistribution.cpp:31.
        return CumulativeGammaDistribution(0.5 * self._df)(0.5 * x)


class NonCentralCumulativeChiSquareSankaranApprox:
    """Sankaran's closed-form approximation to the non-central chi-square CDF.

    # C++ parity: ``class NonCentralCumulativeChiSquareSankaranApprox`` —
    # chisquaredistribution.hpp:60-68, chisquaredistribution.cpp:98-108.

    A normal approximation after a power transform; it is an *approximation*,
    not a truncated exact series, so it does not converge to
    :class:`~pquantlib.math.distributions.non_central_chi_square_distribution.NonCentralCumulativeChiSquareDistribution`
    as any parameter is refined.
    """

    __slots__ = ("_df", "_ncp")

    def __init__(self, df: float, ncp: float) -> None:
        self._df: float = float(df)
        self._ncp: float = float(ncp)

    def __call__(self, x: float) -> float:
        # C++ parity: chisquaredistribution.cpp:100-107.
        df = self._df
        ncp = self._ncp
        h = 1 - 2 * (df + ncp) * (df + 3 * ncp) / (3 * (df + 2 * ncp) ** 2)
        p = (df + 2 * ncp) / (df + ncp) ** 2
        m = (h - 1) * (1 - 3 * h)

        u = (math.pow(x / (df + ncp), h) - (1 + h * p * (h - 1 - 0.5 * (2 - h) * m * p))) / (
            h * math.sqrt(2 * p) * (1 + 0.5 * m * p)
        )

        return CumulativeNormalDistribution()(u)


class InverseNonCentralCumulativeChiSquareDistribution:
    """Inverse of the non-central chi-square CDF, by doubling search + Brent.

    # C++ parity: ``class InverseNonCentralCumulativeChiSquareDistribution`` —
    # chisquaredistribution.hpp:70-81, chisquaredistribution.cpp:110-140.

    The evaluation budget is shared between the two stages: every doubling of
    the upper bracket consumes one of the ``max_evaluations``, and whatever is
    left is handed to Brent. That is why ``max_evaluations`` defaults to 10 and
    why the routine legitimately raises for inputs whose bracket search eats
    the budget — the raise is pinned behaviour, not a port artefact.
    """

    __slots__ = ("_accuracy", "_guess", "_max_evaluations", "_non_central_dist")

    def __init__(
        self,
        df: float,
        ncp: float,
        max_evaluations: int = 10,
        accuracy: float = 1e-8,
    ) -> None:
        self._non_central_dist: NonCentralCumulativeChiSquareDistribution = (
            NonCentralCumulativeChiSquareDistribution(df, ncp)
        )
        self._guess: float = df + ncp
        self._max_evaluations: int = max_evaluations
        self._accuracy: float = accuracy

    def __call__(self, x: float) -> float:
        # C++ parity: chisquaredistribution.cpp:122-140.
        # First find the right side of the interval.
        upper = self._guess
        evaluations = self._max_evaluations
        while self._non_central_dist(upper) < x and evaluations > 0:
            upper *= 2.0
            evaluations -= 1

        # Use a Brent solver for the rest.
        solver = Brent()
        solver.set_max_evaluations(evaluations)
        x_min = 0.0 if evaluations == self._max_evaluations else 0.5 * upper
        return solver.solve(
            lambda y: self._non_central_dist(y) - x,
            self._accuracy,
            0.75 * upper,
            x_min,
            upper,
        )


__all__ = [
    "CumulativeChiSquareDistribution",
    "InverseNonCentralCumulativeChiSquareDistribution",
    "NonCentralCumulativeChiSquareDistribution",
    "NonCentralCumulativeChiSquareSankaranApprox",
]
