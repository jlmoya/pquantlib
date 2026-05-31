"""SwapSpreadIndex — spread between two swap-rate indexes.

# C++ parity: ql/experimental/coupons/swapspreadindex.hpp + .cpp (v1.42.1,
# 099987f0).

The fixing is the gearing-weighted combination of two underlying
:class:`~pquantlib.indexes.swap_index.SwapIndex` fixings

    fixing = gearing1 * swapIndex1.fixing + gearing2 * swapIndex2.fixing

with the canonical CMS-spread defaults ``gearing1 = +1``, ``gearing2 = -1``.

The two sub-indexes must agree on fixing days, fixing calendar, currency,
day counter, fixed-leg tenor and fixed-leg convention (C++ enforces all six
via ``QL_REQUIRE`` in the constructor).

# C++ parity divergence (registration): the C++ constructor calls
# ``registerWith(swapIndex1_)`` / ``registerWith(swapIndex2_)``. PQuantLib's
# Index hierarchy does not yet thread upstream observer registration for
# indexes (no observable Settings module — same as SwapIndex/IborIndex which
# likewise do not register with their term structures), so it is omitted here.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.indexes.interest_rate_index import InterestRateIndex
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.time.date import Date


class SwapSpreadIndex(InterestRateIndex):
    """Spread between two swap-rate indexes (CMS-spread underlying)."""

    def __init__(
        self,
        family_name: str,
        swap_index1: SwapIndex,
        swap_index2: SwapIndex,
        gearing1: float = 1.0,
        gearing2: float = -1.0,
    ) -> None:
        # C++ passes swapIndex1's tenor as the (meaningless) index tenor —
        # "does not make sense, but we have to provide one" (swapspreadindex.cpp).
        super().__init__(
            family_name,
            swap_index1.tenor(),
            swap_index1.fixing_days(),
            swap_index1.currency(),
            swap_index1.fixing_calendar(),
            swap_index1.day_counter(),
        )
        self._swap_index1: SwapIndex = swap_index1
        self._swap_index2: SwapIndex = swap_index2
        self._gearing1: float = gearing1
        self._gearing2: float = gearing2

        # C++ builds a composite name with 4-decimal fixed gearings:
        #   "<idx1>(g1) + <idx2>(g2)"  (swapspreadindex.cpp lines 42-46).
        self._name = (
            f"{swap_index1.name()}({gearing1:.4f}) + "
            f"{swap_index2.name()}({gearing2:.4f})"
        )

        # The six compatibility requirements (swapspreadindex.cpp lines 48-83).
        qassert.require(
            swap_index1.fixing_days() == swap_index2.fixing_days(),
            f"index1 fixing days ({swap_index1.fixing_days()}) must be equal "
            f"to index2 fixing days ({swap_index2.fixing_days()})",
        )
        qassert.require(
            swap_index1.fixing_calendar() == swap_index2.fixing_calendar(),
            f"index1 fixingCalendar ({swap_index1.fixing_calendar()}) must be "
            f"equal to index2 fixingCalendar ({swap_index2.fixing_calendar()})",
        )
        qassert.require(
            swap_index1.currency() == swap_index2.currency(),
            f"index1 currency ({swap_index1.currency()}) must be equal to "
            f"index2 currency ({swap_index2.currency()})",
        )
        qassert.require(
            swap_index1.day_counter() == swap_index2.day_counter(),
            f"index1 dayCounter ({swap_index1.day_counter()}) must be equal "
            f"to index2 dayCounter ({swap_index2.day_counter()})",
        )
        qassert.require(
            swap_index1.fixed_leg_tenor() == swap_index2.fixed_leg_tenor(),
            f"index1 fixedLegTenor ({swap_index1.fixed_leg_tenor()}) must be "
            f"equal to index2 fixedLegTenor ({swap_index2.fixed_leg_tenor()})",
        )
        qassert.require(
            swap_index1.fixed_leg_convention() == swap_index2.fixed_leg_convention(),
            f"index1 fixedLegConvention ({swap_index1.fixed_leg_convention()}) "
            f"must be equal to index2 fixedLegConvention "
            f"({swap_index2.fixed_leg_convention()})",
        )

    # --- InterestRateIndex interface ------------------------------------------

    def maturity_date(self, value_date: Date) -> Date:
        """C++ parity: ``QL_FAIL`` — a spread index has no single maturity."""
        qassert.fail("SwapSpreadIndex does not provide a single maturity date")

    def forecast_fixing(self, fixing_date: Date) -> float:
        """gearing1 * fix1 + gearing2 * fix2 (swapspreadindex.hpp lines 63-69).

        Uses ``forecast_todays_fixing=False`` so a historic fixing on the
        evaluation date is honoured, matching ``swapIndex->fixing(date, false)``.
        """
        return self._gearing1 * self._swap_index1.fixing(fixing_date, False) + (
            self._gearing2 * self._swap_index2.fixing(fixing_date, False)
        )

    def past_fixing(self, fixing_date: Date) -> float:
        """C++ parity: swapspreadindex.hpp lines 71-81.

        Returns NaN (the float ``Null<Real>()`` analogue) when *either*
        sub-index has no past fixing, signalling a missing spread fixing.

        # C++ parity divergence: C++ ``SwapIndex::pastFixing`` returns
        # ``Null<Real>()`` for a missing fixing, whereas PQuantLib's
        # ``Index.past_fixing`` *raises*. To preserve the C++ "missing => Null"
        # contract we probe ``has_historical_fixing`` first and short-circuit
        # to NaN, matching swapspreadindex.hpp's null-propagation.
        """
        if not (
            self._swap_index1.has_historical_fixing(fixing_date)
            and self._swap_index2.has_historical_fixing(fixing_date)
        ):
            return math.nan
        f1 = self._swap_index1.past_fixing(fixing_date)
        f2 = self._swap_index2.past_fixing(fixing_date)
        if math.isnan(f1) or math.isnan(f2):
            return math.nan
        return self._gearing1 * f1 + self._gearing2 * f2

    def allows_native_fixings(self) -> bool:
        """C++ parity: spread indexes forbid native fixings."""
        return False

    # --- inspectors ------------------------------------------------------------

    def swap_index1(self) -> SwapIndex:
        return self._swap_index1

    def swap_index2(self) -> SwapIndex:
        return self._swap_index2

    def gearing1(self) -> float:
        return self._gearing1

    def gearing2(self) -> float:
        return self._gearing2
