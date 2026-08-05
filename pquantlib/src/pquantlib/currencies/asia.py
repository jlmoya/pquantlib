"""Asian currencies.

# C++ parity: ql/currencies/asia.hpp + asia.cpp (v1.42.1).

Ports the currencies referenced by the Asian index families: JPY, ILS,
THB (Bibor / THBFIX), KRW (KOFR) and CNY (Shibor).
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


class THBCurrency(Currency):
    """Thai baht — ISO 764 — 100 stang."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Thai baht",
            code="THB",
            numeric_code=764,
            symbol="Bht",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class KRWCurrency(Currency):
    """South-Korean won — ISO 410 — 100 chon."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="South-Korean won",
            code="KRW",
            numeric_code=410,
            symbol="W",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class CNYCurrency(Currency):
    """Chinese yuan — ISO 156 — 100 fen."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Chinese yuan",
            code="CNY",
            numeric_code=156,
            symbol="Y",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
