"""American currencies.

# C++ parity: ql/currencies/america.hpp + america.cpp (v1.42.1).

Ports only the currencies the L1-B probe references (USD). Other
hemisphere currencies (CAD, BRL, MXN, ARS, CLP, COP, PEN, VEB, etc.)
are deferred to follow-up clusters — same translation pattern as below.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Rounding


class USDCurrency(Currency):
    """U.S. dollar — ISO 840 — divided into 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="U.S. dollar",
            code="USD",
            numeric_code=840,
            symbol="$",
            fraction_symbol="\xa2",  # cent sign
            fractions_per_unit=100,
            rounding=Rounding(),
        )
