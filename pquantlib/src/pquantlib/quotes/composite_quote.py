"""CompositeQuote — Quote combining two underlying Quotes.

# C++ parity: ql/quotes/compositequote.hpp (v1.42.1)
"""

from __future__ import annotations

from collections.abc import Callable

from pquantlib import qassert
from pquantlib.quotes.quote import Quote


class CompositeQuote(Quote):
    """Quote of the form ``f(element1.value(), element2.value())``."""

    __slots__ = ("_cached", "_element1", "_element2", "_function")

    def __init__(
        self,
        element1: Quote,
        element2: Quote,
        function: Callable[[float, float], float],
    ) -> None:
        super().__init__()
        self._element1: Quote = element1
        self._element2: Quote = element2
        self._function: Callable[[float, float], float] = function
        self._cached: float | None = None
        element1.register_with(self)
        element2.register_with(self)

    def value1(self) -> float:
        return self._element1.value()

    def value2(self) -> float:
        return self._element2.value()

    def value(self) -> float:
        if self._cached is None:
            qassert.require(self.is_valid(), "invalid CompositeQuote")
            self._cached = self._function(self._element1.value(), self._element2.value())
        return self._cached

    def is_valid(self) -> bool:
        return self._element1.is_valid() and self._element2.is_valid()

    def update(self) -> None:
        self._cached = None
        self.notify_observers()
