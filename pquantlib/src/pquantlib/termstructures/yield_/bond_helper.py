"""BondHelper — bootstrap from bond quote.

# C++ parity: ql/termstructures/yield/bondhelpers.{hpp,cpp} class BondHelper.

C++ ``BondHelper`` wraps a ``Bond`` instrument and bootstraps the discount
curve from the bond's clean / dirty price. Both depend on L3 ``Bond`` +
``DiscountingBondEngine``.

This commit ports the constructor surface + inspectors + ``set_term_structure``;
``implied_quote`` raises until L3 lands the Bond instrument.

Once L3 has ``pquantlib.instruments.bond.Bond``, the implementation flips to:

    bond_.set_pricing_engine(DiscountingBondEngine(self._term_structure))
    if self._price_type == BondPriceType.Clean:
        return self._bond.clean_price()
    elif self._price_type == BondPriceType.Dirty:
        return self._bond.dirty_price()
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from pquantlib import qassert
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class BondPriceType(IntEnum):
    """C++ parity: ``Bond::Price::Type``."""

    Clean = 0
    Dirty = 1


class BondHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """Bootstrap helper wrapping a Bond instrument (L3-dependent)."""

    def __init__(
        self,
        price: Quote | float,
        bond: Any,
        price_type: BondPriceType = BondPriceType.Clean,
    ) -> None:
        super().__init__(price)
        self._bond: Any = bond
        self._price_type: BondPriceType = price_type
        # In C++ initialization reads bond.settlementDate() / maturityDate() to
        # populate earliest/latest. Both require the Bond instrument; we leave
        # them None until L3 lands a Bond protocol.

    def implied_quote(self) -> float:
        qassert.fail(
            "BondHelper.implied_quote requires L3 Bond + DiscountingBondEngine "
            "(deferred to L3).",
        )

    def bond(self) -> Any:
        return self._bond

    def price_type(self) -> BondPriceType:
        return self._price_type
