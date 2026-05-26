"""SimpleQuote — concrete leaf Quote storing a mutable scalar value.

# C++ parity: ql/quotes/simplequote.hpp (v1.42.1)

Python uses ``float | None`` as the storage type, with ``None``
representing the C++ ``Null<Real>()`` sentinel for "invalid". The
diff returned by ``set_value`` mirrors the C++ semantics for
float-to-float transitions; for transitions involving invalid state,
Python returns ``0.0`` (C++ produces a huge poison-value diff that's
never used by callers — they treat any non-zero as a change).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.quotes.quote import Quote


class SimpleQuote(Quote):
    """Market element returning a stored value."""

    __slots__ = ("_value",)

    def __init__(self, value: float | None = None) -> None:
        super().__init__()
        self._value: float | None = value

    def value(self) -> float:
        qassert.require(self.is_valid(), "invalid SimpleQuote")
        assert self._value is not None
        return self._value

    def is_valid(self) -> bool:
        return self._value is not None

    def set_value(self, value: float | None = None) -> float:
        """Update the stored value; return the diff from the prior value.

        Notifies observers iff the diff is non-zero. Transitions to/from
        ``None`` always notify (and return 0.0 — see module docstring).
        """
        if self._value is None or value is None:
            if self._value is value:
                return 0.0
            self._value = value
            self.notify_observers()
            return 0.0
        diff = value - self._value
        if diff != 0.0:
            self._value = value
            self.notify_observers()
        return diff

    def reset(self) -> None:
        """Make the quote invalid."""
        self.set_value(None)
