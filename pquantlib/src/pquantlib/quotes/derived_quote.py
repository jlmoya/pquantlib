"""DerivedQuote — Quote whose value depends on another Quote.

# C++ parity: ql/quotes/derivedquote.hpp (v1.42.1)

C++ uses a template parameter for the unary function; Python uses
``Callable[[float], float]``. C++ lazy-caches the computed value in a
mutable field, invalidated on update(); Python does the same.
"""

from __future__ import annotations

from collections.abc import Callable

from pquantlib import qassert
from pquantlib.quotes.quote import Quote


class DerivedQuote(Quote):
    """Quote of the form ``f(element.value())``."""

    __slots__ = ("_cached", "_element", "_function")

    def __init__(self, element: Quote, function: Callable[[float], float]) -> None:
        super().__init__()
        self._element: Quote = element
        self._function: Callable[[float], float] = function
        self._cached: float | None = None
        element.register_with(self)

    def value(self) -> float:
        if self._cached is None:
            qassert.require(self.is_valid(), "invalid DerivedQuote")
            self._cached = self._function(self._element.value())
        return self._cached

    def is_valid(self) -> bool:
        return self._element.is_valid()

    def update(self) -> None:
        self._cached = None
        self.notify_observers()
