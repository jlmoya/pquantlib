"""ImpliedStdDevQuote — implied standard deviation of an underlying.

# C++ parity: ql/quotes/impliedstddevquote.{hpp,cpp} (v1.43)

The quote inverts the Black formula for the standard deviation that reprices
``price`` given ``forward`` and ``strike`` (undiscounted, undisplaced). It is a
``LazyObject``, so the solve runs once per notification.

Two behaviours of the C++ implementation are load-bearing and reproduced here:

* ``impliedStdev_`` is a *mutable* member seeded with the constructor's
  ``guess`` and then **overwritten by each solve**. The next solve is therefore
  seeded with the previous answer, not with ``guess`` — observable whenever
  ``accuracy`` is loose enough for the starting point to matter.
* ``performCalculations`` dereferences ``price_`` *before* the ``try`` block and
  ``forward_`` *inside* it. A missing forward is swallowed to ``0.0``; a missing
  price propagates.

# C++ parity divergence (Handle vs object): C++ takes ``Handle<Quote>`` for the
forward and the price. This port does not implement the ``Handle`` indirection
(see ``cashflows/cms_coupon_pricer.py``) and threads the ``Quote`` directly;
``None`` models C++'s *empty* handle, which ``isValid()`` explicitly tests for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.pricingengines.black_formula import black_formula_implied_std_dev
from pquantlib.quotes.quote import Quote

if TYPE_CHECKING:
    from pquantlib.payoffs import OptionType

# C++ ``static const Real discount = 1.0; static const Real displacement = 0.0;``
# in ImpliedStdDevQuote::performCalculations — the quote is always undiscounted
# and undisplaced.
_DISCOUNT: float = 1.0
_DISPLACEMENT: float = 0.0


class ImpliedStdDevQuote(Quote, LazyObject):
    """Quote for the Black implied standard deviation of an underlying."""

    def __init__(
        self,
        option_type: OptionType,
        forward: Quote | None,
        price: Quote | None,
        strike: float,
        guess: float,
        accuracy: float = 1.0e-6,
        max_iter: int = 100,
    ) -> None:
        super().__init__()
        self._implied_stdev: float = guess
        self._option_type: OptionType = option_type
        self._strike: float = strike
        self._accuracy: float = accuracy
        self._max_iter: int = max_iter
        self._forward: Quote | None = forward
        self._price: Quote | None = price
        if forward is not None:
            forward.register_with(self)
        if price is not None:
            price.register_with(self)

    # --- Quote interface --------------------------------------------------

    def value(self) -> float:
        self.calculate()
        return self._implied_stdev

    def is_valid(self) -> bool:
        return (
            self._price is not None
            and self._forward is not None
            and self._price.is_valid()
            and self._forward.is_valid()
        )

    # --- LazyObject interface ---------------------------------------------

    def _perform_calculations(self) -> None:
        # Outside the try, exactly as in C++: a missing price is NOT swallowed.
        qassert.require(self._price is not None, "empty Handle cannot be dereferenced")
        assert self._price is not None
        black_price = self._price.value()
        try:
            # Inside the try, exactly as in C++: a missing forward IS swallowed.
            qassert.require(self._forward is not None, "empty Handle cannot be dereferenced")
            assert self._forward is not None
            self._implied_stdev = black_formula_implied_std_dev(
                self._option_type,
                self._strike,
                self._forward.value(),
                black_price,
                _DISCOUNT,
                _DISPLACEMENT,
                self._implied_stdev,
                self._accuracy,
                self._max_iter,
            )
        except LibraryException:
            # C++ ``catch (Error&)`` — a failed inversion reads as zero vol,
            # not as an exception.
            self._implied_stdev = 0.0
