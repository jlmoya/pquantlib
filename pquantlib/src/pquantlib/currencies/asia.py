"""Asian currencies.

# C++ parity: ql/currencies/asia.hpp + asia.cpp (v1.42.1).

Ports only the currencies the L1-B probe references (JPY).
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Rounding


class JPYCurrency(Currency):
    """Japanese yen — ISO 392.

    Per the C++ source the yen carries ``fractionsPerUnit = 100`` (historical
    sen), though the modern yen is undivided in practice.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Japanese yen",
            code="JPY",
            numeric_code=392,
            symbol="\xa5",  # yen sign
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class ILSCurrency(Currency):
    """Israeli shekel — ISO 376 — 100 agorot."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Israeli shekel",
            code="ILS",
            numeric_code=376,
            symbol="NIS",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
