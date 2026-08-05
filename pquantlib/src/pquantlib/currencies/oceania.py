"""Oceanian currencies.

# C++ parity: ql/currencies/oceania.hpp + oceania.cpp (v1.43).

All 2 currencies C++ v1.43 declares in this header, in source order.
Data (name / ISO code / numeric code / symbol / fraction symbol / fractions
per unit / rounding / triangulation currency) is transcribed from
``oceania.cpp`` and cross-validated against the ``currencies/all`` probe.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Rounding


class AUDCurrency(Currency):
    """Australian dollar — ISO AUD/36 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Australian dollar",
            code="AUD",
            numeric_code=36,
            symbol="A$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class NZDCurrency(Currency):
    """New Zealand dollar — ISO NZD/554 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="New Zealand dollar",
            code="NZD",
            numeric_code=554,
            symbol="NZ$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
