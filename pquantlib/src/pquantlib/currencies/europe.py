"""European currencies.

# C++ parity: ql/currencies/europe.hpp + europe.cpp (v1.43).

All 42 currencies C++ v1.43 declares in this header, in source order.
Data (name / ISO code / numeric code / symbol / fraction symbol / fractions
per unit / rounding / triangulation currency) is transcribed from
``europe.cpp`` and cross-validated against the ``currencies/all`` probe.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import ClosestRounding, Rounding


class BGLCurrency(Currency):
    """Bulgarian lev — ISO BGL/100 — 100 stotinki."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Bulgarian lev",
            code="BGL",
            numeric_code=100,
            symbol="lv",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class BYRCurrency(Currency):
    """Belarussian ruble — ISO BYR/974 — no subdivisions."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Belarussian ruble",
            code="BYR",
            numeric_code=974,
            symbol="BR",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
        )


class CHFCurrency(Currency):
    """Swiss franc — ISO CHF/756 — 100 cents."""

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


class CYPCurrency(Currency):
    """Cyprus pound — ISO CYP/196 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Cyprus pound",
            code="CYP",
            numeric_code=196,
            symbol="\xa3C",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class CZKCurrency(Currency):
    """Czech koruna — ISO CZK/203 — 100 haleru."""

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


class DKKCurrency(Currency):
    """Danish krone — ISO DKK/208 — 100 øre."""

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


class EEKCurrency(Currency):
    """Estonian kroon — ISO EEK/233 — 100 senti."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Estonian kroon",
            code="EEK",
            numeric_code=233,
            symbol="KR",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class EURCurrency(Currency):
    """European Euro — ISO EUR/978 — 100 cents."""

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
    """British pound sterling — ISO GBP/826 — 100 pence."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="British pound sterling",
            code="GBP",
            numeric_code=826,
            symbol="\xa3",
            fraction_symbol="p",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class HUFCurrency(Currency):
    """Hungarian forint — ISO HUF/348 — no subdivisions."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Hungarian forint",
            code="HUF",
            numeric_code=348,
            symbol="Ft",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
        )


class ISKCurrency(Currency):
    """Iceland krona — ISO ISK/352 — 100 aurar."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Iceland krona",
            code="ISK",
            numeric_code=352,
            symbol="IKr",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class LTLCurrency(Currency):
    """Lithuanian litas — ISO LTL/440 — 100 centu."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Lithuanian litas",
            code="LTL",
            numeric_code=440,
            symbol="Lt",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class LVLCurrency(Currency):
    """Latvian lat — ISO LVL/428 — 100 santims."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Latvian lat",
            code="LVL",
            numeric_code=428,
            symbol="Ls",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class MKDCurrency(Currency):
    """Macedonian denar — ISO MKD/807 — 100 deni."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Macedonian denar",
            code="MKD",
            numeric_code=807,
            symbol="den",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class NOKCurrency(Currency):
    """Norwegian krone — ISO NOK/578 — 100 øre."""

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


class PLNCurrency(Currency):
    """Polish zloty — ISO PLN/985 — 100 groszy."""

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


class ROLCurrency(Currency):
    """Romanian leu — ISO ROL/642 — 100 bani."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Romanian leu",
            code="ROL",
            numeric_code=642,
            symbol="L",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class RONCurrency(Currency):
    """Romanian new leu — ISO RON/946 — 100 bani."""

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
    """Russian ruble — ISO RUB/643 — 100 kopeyki."""

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


class SEKCurrency(Currency):
    """Swedish krona — ISO SEK/752 — 100 öre."""

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


class SITCurrency(Currency):
    """Slovenian tolar — ISO SIT/705 — 100 stotinov."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Slovenian tolar",
            code="SIT",
            numeric_code=705,
            symbol="SlT",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class TRLCurrency(Currency):
    """Turkish lira — ISO TRL/792 — 100 kurus."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Turkish lira",
            code="TRL",
            numeric_code=792,
            symbol="TL",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class TRYCurrency(Currency):
    """New Turkish lira — ISO TRY/949 — 100 new kurus."""

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


class ATSCurrency(Currency):
    """Austrian shilling — ISO ATS/40 — 100 groschen. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Austrian shilling",
            code="ATS",
            numeric_code=40,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class BEFCurrency(Currency):
    """Belgian franc — ISO BEF/56. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Belgian franc",
            code="BEF",
            numeric_code=56,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class DEMCurrency(Currency):
    """Deutsche mark — ISO DEM/276 — 100 pfennig. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Deutsche mark",
            code="DEM",
            numeric_code=276,
            symbol="DM",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class ESPCurrency(Currency):
    """Spanish peseta — ISO ESP/724 — 100 centimos. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Spanish peseta",
            code="ESP",
            numeric_code=724,
            symbol="Pta",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class FIMCurrency(Currency):
    """Finnish markka — ISO FIM/246 — 100 penniä. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Finnish markka",
            code="FIM",
            numeric_code=246,
            symbol="mk",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class FRFCurrency(Currency):
    """French franc — ISO FRF/250 — 100 centimes. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="French franc",
            code="FRF",
            numeric_code=250,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class GRDCurrency(Currency):
    """Greek drachma — ISO GRD/300 — 100 lepta. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Greek drachma",
            code="GRD",
            numeric_code=300,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class IEPCurrency(Currency):
    """Irish punt — ISO IEP/372 — 100 pence. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Irish punt",
            code="IEP",
            numeric_code=372,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class ITLCurrency(Currency):
    """Italian lira — ISO ITL/380. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Italian lira",
            code="ITL",
            numeric_code=380,
            symbol="L",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class LUFCurrency(Currency):
    """Luxembourg franc — ISO LUF/442 — 100 centimes. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Luxembourg franc",
            code="LUF",
            numeric_code=442,
            symbol="F",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class MTLCurrency(Currency):
    """Maltese lira — ISO MTL/470 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Maltese lira",
            code="MTL",
            numeric_code=470,
            symbol="Lm",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class NLGCurrency(Currency):
    """Dutch guilder — ISO NLG/528 — 100 cents. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Dutch guilder",
            code="NLG",
            numeric_code=528,
            symbol="f",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class PTECurrency(Currency):
    """Portuguese escudo — ISO PTE/620 — 100 centavos. Pre-euro; triangulates through EUR."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Portuguese escudo",
            code="PTE",
            numeric_code=620,
            symbol="Esc",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
            triangulation_currency=EURCurrency(),
        )


class SKKCurrency(Currency):
    """Slovak koruna — ISO SKK/703 — 100 halierov."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Slovak koruna",
            code="SKK",
            numeric_code=703,
            symbol="Sk",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class UAHCurrency(Currency):
    """Ukrainian hryvnia — ISO UAH/980 — 100 kopiykas."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Ukrainian hryvnia",
            code="UAH",
            numeric_code=980,
            symbol="hrn",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class RSDCurrency(Currency):
    """Serbian dinar — ISO RSD/941 — 100 para/napa."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Serbian dinar",
            code="RSD",
            numeric_code=941,
            symbol="RSD",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class HRKCurrency(Currency):
    """Croatian kuna — ISO HRK/191 — 100 lipa."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Croatian kuna",
            code="HRK",
            numeric_code=191,
            symbol="HRK",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class BGNCurrency(Currency):
    """Bulgarian lev — ISO BGN/975 — 100 stotinki."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Bulgarian lev",
            code="BGN",
            numeric_code=975,
            symbol="BGN",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class GELCurrency(Currency):
    """Georgian lari — ISO GEL/981 — 100 tetri."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Georgian lari",
            code="GEL",
            numeric_code=981,
            symbol="GEL",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
