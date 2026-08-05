"""Asian currencies.

# C++ parity: ql/currencies/asia.hpp + asia.cpp (v1.43).

All 29 currencies C++ v1.43 declares in this header, in source order.
Data (name / ISO code / numeric code / symbol / fraction symbol / fractions
per unit / rounding / triangulation currency) is transcribed from
``asia.cpp`` and cross-validated against the ``currencies/all`` probe.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Rounding


class BDTCurrency(Currency):
    """Bangladesh taka — ISO BDT/50 — 100 paisa."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Bangladesh taka",
            code="BDT",
            numeric_code=50,
            symbol="Bt",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class CNYCurrency(Currency):
    """Chinese yuan — ISO CNY/156 — 100 fen."""

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


class HKDCurrency(Currency):
    """Hong Kong dollar — ISO HKD/344 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Hong Kong dollar",
            code="HKD",
            numeric_code=344,
            symbol="HK$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class IDRCurrency(Currency):
    """Indonesian Rupiah — ISO IDR/360 — 100 sen."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Indonesian Rupiah",
            code="IDR",
            numeric_code=360,
            symbol="Rp",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class ILSCurrency(Currency):
    """Israeli shekel — ISO ILS/376 — 100 agorot."""

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


class INRCurrency(Currency):
    """Indian rupee — ISO INR/356 — 100 paise."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Indian rupee",
            code="INR",
            numeric_code=356,
            symbol="Rs",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class IQDCurrency(Currency):
    """Iraqi dinar — ISO IQD/368 — 1000 fils."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Iraqi dinar",
            code="IQD",
            numeric_code=368,
            symbol="ID",
            fraction_symbol="",
            fractions_per_unit=1000,
            rounding=Rounding(),
        )


class IRRCurrency(Currency):
    """Iranian rial — ISO IRR/364 — no subdivisions."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Iranian rial",
            code="IRR",
            numeric_code=364,
            symbol="Rls",
            fraction_symbol="",
            fractions_per_unit=1,
            rounding=Rounding(),
        )


class JPYCurrency(Currency):
    """Japanese yen — ISO JPY/392 — 100 sen."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Japanese yen",
            code="JPY",
            numeric_code=392,
            symbol="\xa5",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class KRWCurrency(Currency):
    """South-Korean won — ISO KRW/410 — 100 chon."""

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


class KWDCurrency(Currency):
    """Kuwaiti dinar — ISO KWD/414 — 1000 fils."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Kuwaiti dinar",
            code="KWD",
            numeric_code=414,
            symbol="KD",
            fraction_symbol="",
            fractions_per_unit=1000,
            rounding=Rounding(),
        )


class KZTCurrency(Currency):
    """Kazakstanti Tenge — ISO KZT/398."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Kazakstanti Tenge",
            code="KZT",
            numeric_code=398,
            symbol="Kzt",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class MYRCurrency(Currency):
    """Malaysian Ringgit — ISO MYR/458 — 100 sen."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Malaysian Ringgit",
            code="MYR",
            numeric_code=458,
            symbol="RM",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class NPRCurrency(Currency):
    """Nepal rupee — ISO NPR/524 — 100 paise."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Nepal rupee",
            code="NPR",
            numeric_code=524,
            symbol="NRs",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class PKRCurrency(Currency):
    """Pakistani rupee — ISO PKR/586 — 100 paisa."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Pakistani rupee",
            code="PKR",
            numeric_code=586,
            symbol="Rs",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class SARCurrency(Currency):
    """Saudi riyal — ISO SAR/682 — 100 halalat."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Saudi riyal",
            code="SAR",
            numeric_code=682,
            symbol="SRls",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class SGDCurrency(Currency):
    """Singapore dollar — ISO SGD/702 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Singapore dollar",
            code="SGD",
            numeric_code=702,
            symbol="S$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class THBCurrency(Currency):
    """Thai baht — ISO THB/764 — 100 stang."""

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


class TWDCurrency(Currency):
    """Taiwan dollar — ISO TWD/901 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Taiwan dollar",
            code="TWD",
            numeric_code=901,
            symbol="NT$",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class VNDCurrency(Currency):
    """Vietnamese Dong — ISO VND/704 — 100 xu."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Vietnamese Dong",
            code="VND",
            numeric_code=704,
            symbol="",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class QARCurrency(Currency):
    """Qatari riyal — ISO QAR/634 — 100 diram."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Qatari riyal",
            code="QAR",
            numeric_code=634,
            symbol="QAR",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class BHDCurrency(Currency):
    """Bahraini dinar — ISO BHD/48 — 1000 fils."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Bahraini dinar",
            code="BHD",
            numeric_code=48,
            symbol="BHD",
            fraction_symbol="",
            fractions_per_unit=1000,
            rounding=Rounding(),
        )


class OMRCurrency(Currency):
    """Omani rial — ISO OMR/512 — 1000 baisa."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Omani rial",
            code="OMR",
            numeric_code=512,
            symbol="OMR",
            fraction_symbol="",
            fractions_per_unit=1000,
            rounding=Rounding(),
        )


class JODCurrency(Currency):
    """Jordanian dinar — ISO JOD/400 — 100 qirshes."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Jordanian dinar",
            code="JOD",
            numeric_code=400,
            symbol="JOD",
            fraction_symbol="",
            fractions_per_unit=1000,
            rounding=Rounding(),
        )


class AEDCurrency(Currency):
    """United Arab Emirates dirham — ISO AED/784 — 100 fils."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="United Arab Emirates dirham",
            code="AED",
            numeric_code=784,
            symbol="AED",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class PHPCurrency(Currency):
    """Philippine peso — ISO PHP/608 — 100 centavo."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Philippine peso",
            code="PHP",
            numeric_code=608,
            symbol="PHP",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class CNHCurrency(Currency):
    """Chinese yuan (Hong Kong) — ISO CNH/156 — 100 fen."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Chinese yuan (Hong Kong)",
            code="CNH",
            numeric_code=156,
            symbol="CNH",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class LKRCurrency(Currency):
    """Sri Lankan rupee — ISO LKR/144 — 100 cents."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Sri Lankan rupee",
            code="LKR",
            numeric_code=144,
            symbol="LKR",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )


class UZSCurrency(Currency):
    """Uzbekistani Som — ISO UZS/860 — 100 tiyin."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Uzbekistani Som",
            code="UZS",
            numeric_code=860,
            symbol="UZS",
            fraction_symbol="",
            fractions_per_unit=100,
            rounding=Rounding(),
        )
