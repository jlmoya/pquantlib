"""European currencies.

# C++ parity: ql/currencies/europe.hpp + europe.cpp (v1.42.1).

Ports only the currencies the L1-B probe references (EUR, GBP, CHF).
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import ClosestRounding, Rounding


class EURCurrency(Currency):
    """European Euro — ISO 978 — closest-rounding to 2 decimals."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="European Euro",
            code="EUR",
            numeric_code=978,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=ClosestRounding(2),
        )


class GBPCurrency(Currency):
    """British pound sterling — ISO 826 — 100 pence."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="British pound sterling",
            code="GBP",
            numeric_code=826,
            symbol="\xa3",  # pound sign
            fraction_symbol="p",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class CHFCurrency(Currency):
    """Swiss franc — ISO 756 — 100 centimes."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Swiss franc",
            code="CHF",
            numeric_code=756,
            symbol="SwF",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
