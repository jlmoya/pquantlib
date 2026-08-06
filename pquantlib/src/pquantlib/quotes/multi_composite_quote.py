"""MultiCompositeQuote — Quote combining N underlying Quotes through a function.

# C++ parity: ql/quotes/multicompositequote.hpp (v1.43)

The N-ary generalisation of
:class:`~pquantlib.quotes.composite_quote.CompositeQuote`. C++ templates on an
``ArrayFunction`` taking a ``QuantLib::Array``; PQuantLib's ``Array`` is a 1-D
``numpy`` vector (see ``math/array.py``), so the callback signature ports
directly as ``Callable[[Array], float]`` — the element values arrive in
constructor order.

The computed value is cached and dropped on any element's notification, exactly
as in C++ where ``value_`` is reset to ``Null<Real>()`` by ``update()``.

# C++ parity divergence (Handle vs object): C++ holds
``std::vector<Handle<Quote>>``. This port threads the ``Quote`` objects
directly (see ``cashflows/cms_coupon_pricer.py``); ``None`` models C++'s
*empty* handle, which ``isValid()`` explicitly tests for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.quotes.quote import Quote

if TYPE_CHECKING:
    from pquantlib.math.array import Array


class MultiCompositeQuote(Quote):
    """Quote of the form ``f(array of element values)``.

    # C++ parity: ``class MultiCompositeQuote : public Quote, public Observer``.
    """

    __slots__ = ("_cached", "_elements", "_function")

    def __init__(
        self,
        elements: Sequence[Quote | None],
        function: Callable[[Array], float],
    ) -> None:
        super().__init__()
        self._elements: list[Quote | None] = list(elements)
        self._function: Callable[[Array], float] = function
        self._cached: float | None = None
        for element in self._elements:
            if element is not None:
                element.register_with(self)

    # --- inspectors -------------------------------------------------------

    def input_value(self, i: int) -> float:
        """Value of the ``i``-th element.

        # C++ parity: ``elements_.at(i)->value()`` — ``at`` is bounds-checked
        # and throws ``std::out_of_range``; Python's list indexing raises
        # ``IndexError``, which is the same contract in the local idiom.
        """
        element = self._elements[i]
        qassert.require(element is not None, "empty Handle cannot be dereferenced")
        assert element is not None
        return element.value()

    # --- Quote interface --------------------------------------------------

    def value(self) -> float:
        if self._cached is None:
            qassert.require(self.is_valid(), "invalid MultiCompositeQuote")
            args: Array = np.empty(len(self._elements), dtype=np.float64)
            for i, element in enumerate(self._elements):
                # Guaranteed non-None by the is_valid() check above.
                assert element is not None
                args[i] = element.value()
            self._cached = self._function(args)
        return self._cached

    def is_valid(self) -> bool:
        return all(element is not None and element.is_valid() for element in self._elements)

    # --- Observer interface -----------------------------------------------

    def update(self) -> None:
        self._cached = None
        self.notify_observers()
