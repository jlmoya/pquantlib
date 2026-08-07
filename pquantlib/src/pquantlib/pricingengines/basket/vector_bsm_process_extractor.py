"""VectorBsmProcessExtractor — per-asset market data out of a process vector.

# C++ parity:
# ql/pricingengines/basket/vectorbsmprocessextractor.{hpp,cpp} (v1.43),
# ``QuantLib::detail::VectorBsmProcessExtractor``.

Public helper shared by :class:`SingleFactorBsmBasketEngine`,
:class:`ChoiBasketEngine` and :class:`DengLiZhouBasketEngine`. It pulls the
five quantities those engines need out of a list of
:class:`GeneralizedBlackScholesProcess`:

* :meth:`get_spot` — ``p.x0()`` per asset;
* :meth:`get_dividend_yield_df` — ``p.dividendYield().discount(T)``;
* :meth:`get_interest_rate_df` — the single risk-free discount factor,
  after checking every asset agrees on it;
* :meth:`get_black_variance` — ``p.blackVolatility().blackVariance(T, x0)``;
* :meth:`get_black_std_dev` — ``p.blackVolatility().blackVol(T, x0) *
  sqrt(t)``.

Two details a port gets wrong easily, both pinned by the probe:

1. :meth:`get_interest_rate_df` compares the per-asset **discount factors**
   with ``close_enough``, not the curve objects and not their identity. Two
   distinct ``FlatForward`` objects at the same rate are accepted; different
   rates throw ``"interest rates need to be the same for all underlyings"``.
   (``GaussianCopulaSpreadEngine`` really does compare curve identity — that
   check must not be copied here.)
2. :meth:`get_black_std_dev` is **signed**: it multiplies ``blackVol`` by
   ``sqrt(t)``, and ``blackVol`` of a ``BlackConstantVol`` built with a
   negative volatility is negative. :meth:`get_black_variance` is
   ``vol**2 * t`` and therefore never negative. ``SingleFactorBsmBasketEngine``
   depends on that asymmetry: a negative volatility is how a negative-weight
   leg satisfies its ``a * sig >= 0`` guard.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.date import Date


class VectorBsmProcessExtractor:
    """Extract per-asset market data from a vector of BSM processes.

    # C++ parity: ``detail::VectorBsmProcessExtractor``
    # (vectorbsmprocessextractor.hpp:32-48).
    """

    __slots__ = ("_processes",)

    def __init__(self, processes: Sequence[GeneralizedBlackScholesProcess]) -> None:
        # C++ parity: vectorbsmprocessextractor.cpp:29-32.
        self._processes: tuple[GeneralizedBlackScholesProcess, ...] = tuple(processes)

    # --- internals ---------------------------------------------------------

    def _extract(
        self, f: Callable[[GeneralizedBlackScholesProcess], float]
    ) -> Array:
        """``extractProcesses`` — map ``f`` over the processes.

        # C++ parity: vectorbsmprocessextractor.cpp:34-42.
        """
        return np.array([f(p) for p in self._processes], dtype=np.float64)

    # --- public surface ----------------------------------------------------

    def get_spot(self) -> Array:
        """``p.x0()`` per asset.

        # C++ parity: ``getSpot`` (vectorbsmprocessextractor.cpp:63-65).
        """
        return self._extract(lambda p: p.x0())

    def get_dividend_yield_df(self, maturity_date: Date) -> Array:
        """``p.dividendYield().discount(T)`` per asset.

        # C++ parity: ``getDividendYieldDf``
        # (vectorbsmprocessextractor.cpp:67-74).
        """
        return self._extract(lambda p: p.dividend_yield().discount(maturity_date))

    def get_interest_rate_df(self, maturity_date: Date) -> float:
        """The common risk-free discount factor at ``maturity_date``.

        # C++ parity: ``getInterestRateDf``
        # (vectorbsmprocessextractor.cpp:44-61).

        Raises:
            LibraryException: if the per-asset discount factors are not all
                ``close_enough`` to one another. The comparison is on the
                *values*; distinct curve objects producing equal discounts are
                accepted.
        """
        dr = self._extract(lambda p: p.risk_free_rate().discount(maturity_date))
        first = float(dr[0])
        qassert.require(
            all(close_enough(float(x), first) for x in dr[1:]),
            "interest rates need to be the same for all underlyings",
        )
        return first

    def get_black_variance(self, maturity_date: Date) -> Array:
        """``p.blackVolatility().blackVariance(T, p.x0())`` per asset.

        # C++ parity: ``getBlackVariance``
        # (vectorbsmprocessextractor.cpp:76-83).
        """
        return self._extract(
            lambda p: p.black_volatility().black_variance(maturity_date, p.x0())
        )

    def get_black_std_dev(self, maturity_date: Date) -> Array:
        """``blackVol(T, x0) * sqrt(t)`` per asset — **signed**.

        # C++ parity: ``getBlackStdDev``
        # (vectorbsmprocessextractor.cpp:85-93).

        Deliberately not ``sqrt(get_black_variance(...))``: a negative
        volatility must produce a negative standard deviation, which is what
        lets ``SingleFactorBsmBasketEngine`` price negative-weight legs.
        """

        def one(p: GeneralizedBlackScholesProcess) -> float:
            vol_ts = p.black_volatility()
            maturity = vol_ts.time_from_reference(maturity_date)
            return vol_ts.black_vol(maturity_date, p.x0()) * math.sqrt(maturity)

        return self._extract(one)


__all__ = ["VectorBsmProcessExtractor"]
