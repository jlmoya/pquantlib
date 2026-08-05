"""OvernightIndexedSwapIndex — swap-rate index whose float leg is an OIS leg.

# C++ parity: ql/indexes/swapindex.{hpp,cpp} (v1.43)

Unlike the ISDA-fix families, none of the conventions are declared at the call
site: the fixed leg is always annual / ``ModifiedFollowing``, and the fixing
calendar and day counter are inherited from the overnight index. Only the
family name, tenor, settlement days and currency are supplied.

``underlying_swap`` builds an ``OvernightIndexedSwap`` via ``make_ois`` rather
than the vanilla-swap builder ``SwapIndex`` uses.

Carve-over: PQuantLib's ``make_ois`` / ``OvernightIndexedCoupon`` implement the
compound averaging flavour only (documented L2-D/L3-C carve-out), so
``averaging_method`` is stored and exposed for parity but ``Simple`` is
rejected at ``underlying_swap`` time rather than silently compounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.currencies.currency import Currency
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.instruments.overnight_indexed_swap import OvernightIndexedSwap


class OvernightIndexedSwapIndex(SwapIndex):
    """OIS-rate index: annual fixed leg vs a compounded overnight leg."""

    def __init__(
        self,
        family_name: str,
        tenor: Period,
        settlement_days: int,
        currency: Currency,
        overnight_index: OvernightIndex,
        telescopic_value_dates: bool = False,
        averaging_method: RateAveraging = RateAveraging.Compound,
    ) -> None:
        super().__init__(
            family_name,
            tenor,
            settlement_days,
            currency,
            overnight_index.fixing_calendar(),
            Period(1, TimeUnit.Years),
            BusinessDayConvention.ModifiedFollowing,
            overnight_index.day_counter(),
            overnight_index,
        )
        self._overnight_index: OvernightIndex = overnight_index
        self._telescopic_value_dates: bool = telescopic_value_dates
        self._averaging_method: RateAveraging = averaging_method

    # --- inspectors ------------------------------------------------------------

    def overnight_index(self) -> OvernightIndex:
        return self._overnight_index

    def averaging_method(self) -> RateAveraging:
        return self._averaging_method

    def telescopic_value_dates(self) -> bool:
        return self._telescopic_value_dates

    # --- underlying swap -------------------------------------------------------

    def underlying_swap(self, fixing_date: Date) -> OvernightIndexedSwap:
        """Mirror C++ ``OvernightIndexedSwapIndex::underlyingSwap`` via ``make_ois``."""
        qassert.require(
            self._averaging_method == RateAveraging.Compound,
            "only RateAveraging.Compound is implemented for the overnight leg",
        )
        # Local import: instruments/ pulls in the L3 swap stack, which would
        # invert the module import order if hoisted (same reason SwapIndex
        # defers its MakeVanillaSwap import).
        from pquantlib.instruments.make_ois import make_ois  # noqa: PLC0415

        return make_ois(
            swap_tenor=self._tenor,
            overnight_index=self._overnight_index,
            fixed_rate=0.0,
            effective_date=self.value_date(fixing_date),
            fixed_leg_day_count=self._day_counter,
            telescopic_value_dates=self._telescopic_value_dates,
        )
