"""African currencies.

# C++ parity: ql/currencies/africa.hpp + africa.cpp (v1.43).

All 14 currencies C++ v1.43 declares in this header, in source order.
Data (name / ISO code / numeric code / symbol / fraction symbol / fractions
per unit / rounding / triangulation currency) is transcribed from
``africa.cpp`` and cross-validated against the ``currencies/all`` probe.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Rounding


class AOACurrency(Currency):
    """Angolan kwanza — ISO AOA/973 — 100 cêntimo."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Angolan kwanza",
            code="AOA",
            numeric_code=973,
            symbol="AOA",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class BWPCurrency(Currency):
    """Botswanan pula — ISO BWP/72 — 100 thebe."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Botswanan pula",
            code="BWP",
            numeric_code=72,
            symbol="P",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class EGPCurrency(Currency):
    """Egyptian pound — ISO EGP/818 — 100 piastres."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Egyptian pound",
            code="EGP",
            numeric_code=818,
            symbol="EGP",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class ETBCurrency(Currency):
    """Ethiopian birr — ISO ETB/230 — 100 santim."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Ethiopian birr",
            code="ETB",
            numeric_code=230,
            symbol="ETB",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class GHSCurrency(Currency):
    """Ghanaian cedi — ISO GHS/936 — 100 pesewas."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Ghanaian cedi",
            code="GHS",
            numeric_code=936,
            symbol="GHS",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class KESCurrency(Currency):
    """Kenyan shilling — ISO KES/404 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Kenyan shilling",
            code="KES",
            numeric_code=404,
            symbol="KES",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class MADCurrency(Currency):
    """Moroccan dirham — ISO MAD/504 — 100 santim."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Moroccan dirham",
            code="MAD",
            numeric_code=504,
            symbol="MAD",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class MURCurrency(Currency):
    """Mauritian rupee — ISO MUR/480 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Mauritian rupee",
            code="MUR",
            numeric_code=480,
            symbol="MUR",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class NGNCurrency(Currency):
    """Nigerian Naira — ISO NGN/566 — 100 kobo."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Nigerian Naira",
            code="NGN",
            numeric_code=566,
            symbol="N",
            fraction_symbol="K",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class TNDCurrency(Currency):
    """Tunisian dinar — ISO TND/788 — 1000 millim."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Tunisian dinar",
            code="TND",
            numeric_code=788,
            symbol="TND",
            fraction_symbol="",
            fractions_per_unit=1000,
            rounding=Rounding(),
        )


class UGXCurrency(Currency):
    """Ugandan shilling — ISO UGX/800 — no subdivisions."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Ugandan shilling",
            code="UGX",
            numeric_code=800,
            symbol="UGX",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
        )


class XOFCurrency(Currency):
    """West African CFA franc — ISO XOF/952 — 100 centime."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="West African CFA franc",
            code="XOF",
            numeric_code=952,
            symbol="XOF",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class ZARCurrency(Currency):
    """South-African rand — ISO ZAR/710 — 100 cents."""

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


class ZMWCurrency(Currency):
    """Zambian kwacha — ISO ZMW/967 — 100 ngwee."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Zambian kwacha",
            code="ZMW",
            numeric_code=967,
            symbol="ZMW",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
