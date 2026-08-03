"""African currencies.

# C++ parity: ql/currencies/africa.hpp + africa.cpp (v1.43).

Ports the currencies referenced by the probes — currently ZAR, added in v1.43
for the ZARONIA overnight index.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Rounding


class ZARCurrency(Currency):
    """South-African rand — ISO 710 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="South-African rand",
            code="ZAR",
            numeric_code=710,
            symbol="R",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
