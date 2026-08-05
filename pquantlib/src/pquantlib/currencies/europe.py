"""European currencies.

# C++ parity: ql/currencies/europe.hpp + europe.cpp (v1.42.1).

Ports the currencies referenced by the European index families: EUR, GBP,
CHF, NOK (NIBOR), DKK (DKKLibor / DESTR), SEK (SEKLibor / SWESTR),
RUB (MOSPRIME), CZK (PRIBOR), RON (ROBOR), TRY (TRLibor) and PLN (WIBOR).
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


class NOKCurrency(Currency):
    """Norwegian krone — ISO 578 — 100 oere."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Norwegian krone",
            code="NOK",
            numeric_code=578,
            symbol="NKr",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class DKKCurrency(Currency):
    """Danish krone — ISO 208 — 100 oere."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Danish krone",
            code="DKK",
            numeric_code=208,
            symbol="Dkr",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class SEKCurrency(Currency):
    """Swedish krona — ISO 752 — 100 oere."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Swedish krona",
            code="SEK",
            numeric_code=752,
            symbol="kr",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class CZKCurrency(Currency):
    """Czech koruna — ISO 203 — 100 haleru."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Czech koruna",
            code="CZK",
            numeric_code=203,
            symbol="Kc",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class PLNCurrency(Currency):
    """Polish zloty — ISO 985 — 100 groszy."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Polish zloty",
            code="PLN",
            numeric_code=985,
            symbol="zl",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class RONCurrency(Currency):
    """Romanian new leu — ISO 946 — 100 bani."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Romanian new leu",
            code="RON",
            numeric_code=946,
            symbol="L",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class RUBCurrency(Currency):
    """Russian ruble — ISO 643 — 100 kopeyki."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Russian ruble",
            code="RUB",
            numeric_code=643,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class TRYCurrency(Currency):
    """New Turkish lira — ISO 949 — 100 new kurus."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="New Turkish lira",
            code="TRY",
            numeric_code=949,
            symbol="YTL",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
