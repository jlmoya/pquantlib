"""American currencies.

# C++ parity: ql/currencies/america.hpp + america.cpp (v1.43).

All 16 currencies C++ v1.43 declares in this header, in source order.
Data (name / ISO code / numeric code / symbol / fraction symbol / fractions
per unit / rounding / triangulation currency) is transcribed from
``america.cpp`` and cross-validated against the ``currencies/all`` probe.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Rounding


class ARSCurrency(Currency):
    """Argentinian peso — ISO ARS/32 — 100 centavos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Argentinian peso",
            code="ARS",
            numeric_code=32,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class BRLCurrency(Currency):
    """Brazilian real — ISO BRL/986 — 100 centavos."""

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


class CADCurrency(Currency):
    """Canadian dollar — ISO CAD/124 — 100 cents."""

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


class CLPCurrency(Currency):
    """Chilean peso — ISO CLP/152 — 100 centavos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Chilean peso",
            code="CLP",
            numeric_code=152,
            symbol="Ch$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class COPCurrency(Currency):
    """Colombian peso — ISO COP/170 — 100 centavos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Colombian peso",
            code="COP",
            numeric_code=170,
            symbol="Col$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class MXNCurrency(Currency):
    """Mexican peso — ISO MXN/484 — 100 centavos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Mexican peso",
            code="MXN",
            numeric_code=484,
            symbol="Mex$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class PENCurrency(Currency):
    """Peruvian nuevo sol — ISO PEN/604 — 100 centimos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Peruvian nuevo sol",
            code="PEN",
            numeric_code=604,
            symbol="S/.",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class PEICurrency(Currency):
    """Peruvian inti — ISO PEI/998 — 100 centimos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Peruvian inti",
            code="PEI",
            numeric_code=998,
            symbol="I/.",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class PEHCurrency(Currency):
    """Peruvian sol — ISO PEH/999 — 100 centavos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Peruvian sol",
            code="PEH",
            numeric_code=999,
            symbol="S./",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class TTDCurrency(Currency):
    """Trinidad & Tobago dollar — ISO TTD/780 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Trinidad & Tobago dollar",
            code="TTD",
            numeric_code=780,
            symbol="TT$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class USDCurrency(Currency):
    """U.S. dollar — ISO USD/840 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="U.S. dollar",
            code="USD",
            numeric_code=840,
            symbol="$",
            fraction_symbol="\xa2",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class VEBCurrency(Currency):
    """Venezuelan bolivar — ISO VEB/862 — 100 centimos."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Venezuelan bolivar",
            code="VEB",
            numeric_code=862,
            symbol="Bs",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class MXVCurrency(Currency):
    """Mexican Unidad de Inversion — ISO MXV/979."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Mexican Unidad de Inversion",
            code="MXV",
            numeric_code=979,
            symbol="MXV",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
        )


class COUCurrency(Currency):
    """Unidad de Valor Real (UVR) (funds code) — ISO COU/970."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Unidad de Valor Real (UVR) (funds code)",
            code="COU",
            numeric_code=970,
            symbol="COU",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class CLFCurrency(Currency):
    """Unidad de Fomento (funds code) — ISO CLF/990."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Unidad de Fomento (funds code)",
            code="CLF",
            numeric_code=990,
            symbol="CLF",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
        )


class UYUCurrency(Currency):
    """Uruguayan peso — ISO UYU/858."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Uruguayan peso",
            code="UYU",
            numeric_code=858,
            symbol="UYU",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
        )
