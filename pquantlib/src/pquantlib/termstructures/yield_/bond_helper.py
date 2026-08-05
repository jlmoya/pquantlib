"""BondHelper / FixedRateBondHelper — bootstrap from a bond price quote.

# C++ parity: ql/termstructures/yield/bondhelpers.{hpp,cpp} (v1.43).

``BondHelper`` wraps a ``Bond``, prices it off the curve being bootstrapped
with a ``DiscountingBondEngine``, and reports the resulting clean or dirty
price as the implied quote.

An earlier revision of this file left ``implied_quote`` raising, with a
docstring saying the implementation waited on "L3 ``Bond`` +
``DiscountingBondEngine``". Both landed; the helper was simply never
finished, and the note outlived its premise. It is finished here.

C++ takes a COPY of the bond (``ext::make_shared<Bond>(*bond)``, which also
slices it to the base class) so that the helper owns it outright and external
code cannot attach a different pricing engine underneath the bootstrap. Python
keeps the reference the caller passed — copying an instrument is not free here
and the slicing has no Python analogue — so the C++ warning applies with more
force: **discard the bond after building the helper, or at least never set a
pricing engine on it.**
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.instruments.bond import Bond


class BondPriceType(IntEnum):
    """C++ parity: ``Bond::Price::Type``."""

    Clean = 0
    Dirty = 1


class BondHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """Bootstrap helper wrapping a Bond instrument."""

    def __init__(
        self,
        price: Quote | float,
        bond: Bond,
        price_type: BondPriceType = BondPriceType.Clean,
    ) -> None:
        super().__init__(price)
        self._bond: Bond = bond
        self._price_type: BondPriceType = price_type
        # C++ parity: bondhelpers.cpp:35-38. latestDate_ is the LAST CASHFLOW
        # date, which can fall after the bond's maturity date because of
        # payment-date adjustment; earliestDate_ is the next cashflow date.
        self._latest_date = bond.cashflows()[-1].date()
        self._earliest_date = bond.next_cash_flow_date()

    # --- BootstrapHelper interface --------------------------------------

    def set_term_structure(self, ts: YieldTermStructureProtocol) -> None:
        """Attach the curve and re-engine the bond onto it.

        # C++ parity: ``BondHelper::setTermStructure`` relinks a handle the
        # engine already holds; without handles the engine is rebuilt instead.
        """
        super().set_term_structure(ts)
        # Local import: termstructures/ should not depend on pricingengines/
        # at module-load time.
        from pquantlib.pricingengines.bond.discounting_bond_engine import (  # noqa: PLC0415
            DiscountingBondEngine,
        )

        self._bond.set_pricing_engine(DiscountingBondEngine(ts))  # type: ignore[arg-type]

    def implied_quote(self) -> float:
        """Clean or dirty price of the bond under the bootstrapped curve.

        # C++ parity: ``BondHelper::impliedQuote`` (bondhelpers.cpp:53-69).
        """
        qassert.require(
            self._term_structure is not None, "BondHelper: term structure not set"
        )
        # C++ forces a recalculation because the helper deliberately did not
        # register as an observer of the curve.
        self._bond.update()
        if self._price_type == BondPriceType.Clean:
            return self._bond.clean_price()
        if self._price_type == BondPriceType.Dirty:
            return self._bond.dirty_price()
        return qassert.fail("This price type isn't implemented.")

    # --- inspectors ------------------------------------------------------

    def bond(self) -> Bond:
        return self._bond

    def price_type(self) -> BondPriceType:
        return self._price_type


class FixedRateBondHelper(BondHelper):
    """Fixed-coupon bond helper for curve bootstrap.

    # C++ parity: ``FixedRateBondHelper`` (bondhelpers.cpp:82-106).

    C++ additionally forwards ex-coupon parameters (``exCouponPeriod``,
    ``exCouponCalendar``, ``exCouponConvention``, ``exCouponEndOfMonth``) to
    the ``FixedRateBond`` it builds. PQuantLib's ``FixedRateBond`` has no
    ex-coupon support, so there is nothing to forward them to; that is an
    ``instruments/bonds`` gap, not a helper one, and inventing dead parameters
    here would hide it.
    """

    def __init__(
        self,
        price: Quote | float,
        settlement_days: int,
        face_amount: float,
        schedule: Schedule,
        coupons: Sequence[float],
        day_counter: DayCounter,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        redemption: float = 100.0,
        issue_date: Date | None = None,
        payment_calendar: Calendar | None = None,
        price_type: BondPriceType = BondPriceType.Clean,
    ) -> None:
        # Local import: termstructures/ should not depend on instruments/ at
        # module-load time (bond_helper is imported from the bootstrap side).
        from pquantlib.instruments.bonds.fixed_rate_bond import (  # noqa: PLC0415
            FixedRateBond,
        )

        super().__init__(
            price,
            FixedRateBond(
                settlement_days,
                face_amount,
                schedule,
                coupons,
                day_counter,
                payment_convention,
                redemption,
                issue_date,
                payment_calendar,
            ),
            price_type,
        )
