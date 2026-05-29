"""CmsMarket — container for CMS-coupon market quotes.

# C++ parity: ql/termstructures/volatility/swaption/cmsmarket.{hpp,cpp}
# (v1.42.1).

The C++ class is a 2-D grid of CMS bid-ask spread quotes per
``(swap_length, swap_index)`` pair, plus a per-cell ``CmsCouponPricer``
that knows how to re-price the spot/forward CMS leg against a swaption
volatility structure. The full C++ behaviour (``reprice``,
``performCalculations``) depends on:

  * ``CmsCouponPricer`` (abstract base + 4 concretes — conundrum /
    Hagan numeric integral / Hagan analytic / linear-TSR).
  * ``Swap`` / ``MakeCms`` factory machinery (``cms.hpp`` builder).
  * ``YieldTermStructure`` discounting handle.
  * ``IborIndex`` for the floating leg of the CMS swap.
  * ``CapFloorTermVolatilityStructure`` indirectly through the pricer.

PQuantLib defers all of those (no ``CmsCouponPricer`` port — see
``cashflows/coupon_pricer.py`` docstring). Phase 11 W2-B lands the
**structural-container** subset:

  * Stores the input ``swap_lengths``, ``swap_indexes``,
    ``bid_ask_spreads`` (as a 2-D grid of bid/ask :class:`Quote`
    pairs), ``pricers`` (held opaquely — typed ``object`` since the
    pricer type is not ported), and the discount ``YieldTermStructure``.
  * Computes ``swap_tenors`` = ``[idx.tenor() for idx in swap_indexes]``
    once at construction.
  * Exposes inspectors (``swap_lengths``, ``swap_tenors``,
    ``swap_indexes``, ``bid_ask_spreads``).

The :meth:`reprice` and related ``weighted_*`` methods are stubs that
raise :class:`LibraryException` — full implementation depends on the
deferred ``CmsCouponPricer`` port.

This shape suffices to let :class:`CmsMarketCalibration` express its
calibration loop over the spread grid (which it does by reading
:attr:`bid_ask_spreads` directly — the loop's residual function takes
a callable as a parameter rather than reading the model spreads from
the market object).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.period import Period


@runtime_checkable
class SwapIndexLike(Protocol):
    """Structural surface a CmsMarket column index must expose.

    Mirrors the C++ ``SwapIndex::tenor()`` usage. Accepts any object
    exposing ``tenor() -> Period`` — concrete :class:`SwapIndex` or
    test doubles alike. (The :class:`SwaptionVolatilityCube` family
    follows the same pattern for ``AtmSwapIndexProtocol``.)
    """

    def tenor(self) -> Period: ...


class CmsMarket:
    """Container for a 2-D grid of CMS bid-ask spread quotes.

    Args:
        swap_lengths: list of swap lengths (``Period``) on the row axis.
        swap_indexes: list of :class:`SwapIndex` instances on the column
            axis.
        bid_ask_spreads: ``(n_swap_lengths, n_swap_indexes)`` grid of
            ``(bid_quote, ask_quote)`` pairs. Each row corresponds to a
            ``swap_length``; each column to a ``swap_index``.
        pricers: list of CMS-coupon pricers, one per swap index. Held
            as ``object`` since the pricer class is not ported in
            PQuantLib (see module docstring).
        discount_curve: yield term structure used to discount the CMS
            leg NPVs.

    Attributes:
        n_exercise: number of swap lengths.
        n_swap_indexes: number of swap indexes.
        swap_tenors: list of ``Period`` taken from each swap index.

    # C++ parity: ``CmsMarket(...)`` constructor at cmsmarket.cpp:51-79.
    """

    def __init__(
        self,
        *,
        swap_lengths: Sequence[Period],
        swap_indexes: Sequence[SwapIndex | SwapIndexLike],
        bid_ask_spreads: Sequence[Sequence[tuple[Quote, Quote]]],
        pricers: Sequence[object],
        discount_curve: YieldTermStructure,
    ) -> None:
        n_exercise = len(swap_lengths)
        n_swap_indexes = len(swap_indexes)
        qassert.require(
            n_exercise > 0,
            "CmsMarket: empty swap_lengths",
        )
        qassert.require(
            n_swap_indexes > 0,
            "CmsMarket: empty swap_indexes",
        )
        qassert.require(
            len(pricers) == n_swap_indexes,
            f"CmsMarket: pricers ({len(pricers)}) must match "
            f"swap_indexes ({n_swap_indexes})",
        )
        qassert.require(
            len(bid_ask_spreads) == n_exercise,
            f"CmsMarket: bid_ask_spreads outer ({len(bid_ask_spreads)}) "
            f"must match swap_lengths ({n_exercise})",
        )
        for i, row in enumerate(bid_ask_spreads):
            qassert.require(
                len(row) == n_swap_indexes,
                f"CmsMarket: bid_ask_spreads row {i} length ({len(row)}) "
                f"must match swap_indexes ({n_swap_indexes})",
            )

        self._swap_lengths: list[Period] = list(swap_lengths)
        self._swap_indexes: list[SwapIndex | SwapIndexLike] = list(swap_indexes)
        self._bid_ask_spreads: list[list[tuple[Quote, Quote]]] = [
            list(row) for row in bid_ask_spreads
        ]
        self._pricers: list[object] = list(pricers)
        self._discount_curve: YieldTermStructure = discount_curve
        self._n_exercise: int = n_exercise
        self._n_swap_indexes: int = n_swap_indexes
        # # C++ parity: ``swap_tenors_`` populated from each index's tenor()
        # # (cmsmarket.cpp:73).
        self._swap_tenors: list[Period] = [
            idx.tenor() for idx in swap_indexes
        ]

    # --- inspectors --------------------------------------------------------

    @property
    def n_exercise(self) -> int:
        return self._n_exercise

    @property
    def n_swap_indexes(self) -> int:
        return self._n_swap_indexes

    def swap_lengths(self) -> list[Period]:
        """Return the row-axis swap lengths.

        # C++ parity: ``CmsMarket::swapLengths()`` (cmsmarket.hpp:60).
        """
        return list(self._swap_lengths)

    def swap_tenors(self) -> list[Period]:
        """Return the column-axis swap tenors (``= swap_indexes[i].tenor()``).

        # C++ parity: ``CmsMarket::swapTenors()`` (cmsmarket.hpp:59).
        """
        return list(self._swap_tenors)

    def swap_indexes(self) -> list[SwapIndex | SwapIndexLike]:
        """Return the list of swap indexes."""
        return list(self._swap_indexes)

    def bid_ask_spreads(self) -> list[list[tuple[Quote, Quote]]]:
        """Return the 2-D bid-ask spread grid.

        Each entry is a ``(bid, ask)`` pair of :class:`Quote` instances.
        """
        return [list(row) for row in self._bid_ask_spreads]

    def discount_curve(self) -> YieldTermStructure:
        return self._discount_curve

    def pricer(self, swap_index_position: int) -> object:
        """Return the CMS pricer at column ``swap_index_position``."""
        qassert.require(
            0 <= swap_index_position < self._n_swap_indexes,
            f"CmsMarket: swap index position {swap_index_position} "
            f"out of range [0, {self._n_swap_indexes})",
        )
        return self._pricers[swap_index_position]

    # --- pricing (deferred) ----------------------------------------------

    def reprice(self, vol_structure: object, mean_reversion: float) -> None:
        """Re-price the CMS leg grid against ``vol_structure``.

        Stub — full implementation requires the deferred
        ``CmsCouponPricer`` port (see module docstring).

        # C++ parity: ``CmsMarket::reprice`` (cmsmarket.cpp:140-170).

        Args:
            vol_structure: a swaption volatility structure (held opaquely;
                the C++ method takes a ``Handle<SwaptionVolatilityStructure>``).
            mean_reversion: per the ``MeanRevertingPricer`` mix-in.
        """
        del vol_structure, mean_reversion
        raise LibraryException(
            "CmsMarket.reprice requires CmsCouponPricer (not ported); "
            "structural container only — see W2-B follow-up notes",
        )

    def weighted_spread_error(self, weights: object) -> float:
        """Stub — see :meth:`reprice` docstring."""
        del weights
        raise LibraryException(
            "CmsMarket.weighted_spread_error requires CmsCouponPricer "
            "(not ported); structural container only",
        )


__all__ = ["CmsMarket", "SwapIndexLike"]
