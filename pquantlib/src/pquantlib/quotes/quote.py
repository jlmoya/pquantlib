"""Quote — purely abstract base class for market observables.

# C++ parity: ql/quote.hpp (v1.42.1)

Quote IS an Observable: subclasses notify their observers when their
value changes. Concrete leaves (SimpleQuote / DerivedQuote /
CompositeQuote) live alongside in this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.patterns.observer import Observable


class Quote(Observable, ABC):
    """Abstract market observable yielding a real-valued quotation.

    A Quote may transiently be in an *invalid* state (e.g. before its
    first ``set_value`` call). ``value()`` raises
    ``LibraryException("invalid Quote")`` in that state; check
    ``is_valid()`` first if invalidity is plausible.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def value(self) -> float:
        """Return the current quotation. Raises if invalid."""

    @abstractmethod
    def is_valid(self) -> bool:
        """Return whether ``value()`` would succeed."""
