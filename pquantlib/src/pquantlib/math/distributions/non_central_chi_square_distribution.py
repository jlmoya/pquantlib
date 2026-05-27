"""Non-central chi-square cumulative distribution.

# C++ parity: ql/math/distributions/chisquaredistribution.{hpp,cpp} (v1.42.1).

The C++ class ``NonCentralCumulativeChiSquareDistribution`` uses an
in-house series expansion driven by ``errmax = 1e-12`` and a 10000-iter
cap. The pquantlib port delegates to ``scipy.stats.ncx2.cdf``, which
uses Boost's continued-fraction implementation under the hood.

Agreement: ``scipy.stats.ncx2.cdf`` and the C++ series typically match
to 1e-12 or better for the parameter ranges used by Cox-Ingersoll-Ross
discount-bond options. Empirically we confirm TIGHT-tier agreement on
the L4-B probe values.

The Sankaran approximation and the inverse non-central chi-square
solver from the C++ header are not ported in L4-B (deferred — only the
straight CDF is needed by Cox-Ingersoll-Ross::discountBondOption).
"""

from __future__ import annotations

from scipy.stats import ncx2  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]


class NonCentralCumulativeChiSquareDistribution:
    """Cumulative distribution function of a non-central chi-square.

    # C++ parity: ``class NonCentralCumulativeChiSquareDistribution``
    # in chisquaredistribution.hpp:42-50 (v1.42.1).

    Parameters
    ----------
    df: degrees of freedom (``df_`` in C++).
    ncp: non-centrality parameter (``ncp_`` in C++).
    """

    __slots__ = ("_df", "_ncp")

    def __init__(self, df: float, ncp: float) -> None:
        self._df: float = float(df)
        self._ncp: float = float(ncp)

    def __call__(self, x: float) -> float:
        """Return ``P[X <= x]`` for ``X ~ chi^2_{df}(ncp)``.

        # C++ parity: chisquaredistribution.cpp:34-95 — the series
        # expansion is delegated to scipy's continued-fraction
        # implementation here. ``x <= 0`` returns 0.
        """
        if x <= 0.0:
            return 0.0
        return float(ncx2.cdf(x, self._df, self._ncp))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
