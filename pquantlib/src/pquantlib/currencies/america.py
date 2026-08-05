"""American currencies.

# C++ parity: ql/currencies/america.hpp + america.cpp (v1.42.1).

Ports the currencies referenced by the American index families: USD, CAD
(CADLibor / CDOR / CORRA) and BRL (CDI). The remaining hemisphere currencies
(MXN, ARS, CLP, COP, PEN, VEB, ...) follow the same translation pattern.
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


class CADCurrency(Currency):
    """Canadian dollar — ISO 124 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Canadian dollar",
            code="CAD",
            numeric_code=124,
            symbol="Can$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class BRLCurrency(Currency):
    """Brazilian real — ISO 986 — 100 centavos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Brazilian real",
            code="BRL",
            numeric_code=986,
            symbol="R$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
