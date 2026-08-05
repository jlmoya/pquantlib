"""Crypto currencies.

# C++ parity: ql/currencies/crypto.hpp + crypto.cpp (v1.43).

All 8 currencies C++ v1.43 declares in this header, in source order.
Data (name / ISO code / numeric code / symbol / fraction symbol / fractions
per unit / rounding / triangulation currency) is transcribed from
``crypto.cpp`` and cross-validated against the ``currencies/all`` probe.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Rounding


class BTCCurrency(Currency):
    """Bitcoin — ISO BTC/10000."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Bitcoin",
            code="BTC",
            numeric_code=10000,
            symbol="BTC",
            fraction_symbol="",
            fractions_per_unit=100000,
            rounding=Rounding(),
        )


class ETHCurrency(Currency):
    """Ethereum — ISO ETH/10001."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Ethereum",
            code="ETH",
            numeric_code=10001,
            symbol="ETH",
            fraction_symbol="",
            fractions_per_unit=100000,
            rounding=Rounding(),
        )


class ETCCurrency(Currency):
    """Ethereum Classic — ISO ETC/10002."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Ethereum Classic",
            code="ETC",
            numeric_code=10002,
            symbol="ETC",
            fraction_symbol="",
            fractions_per_unit=100000,
            rounding=Rounding(),
        )


class BCHCurrency(Currency):
    """Bitcoin Cash — ISO BCH/10003."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Bitcoin Cash",
            code="BCH",
            numeric_code=10003,
            symbol="BCH",
            fraction_symbol="",
            fractions_per_unit=100000,
            rounding=Rounding(),
        )


class XRPCurrency(Currency):
    """Ripple — ISO XRP/10004."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Ripple",
            code="XRP",
            numeric_code=10004,
            symbol="XRP",
            fraction_symbol="",
            fractions_per_unit=100000,
            rounding=Rounding(),
        )


class LTCCurrency(Currency):
    """Litecoin — ISO LTC/10005."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Litecoin",
            code="LTC",
            numeric_code=10005,
            symbol="LTC",
            fraction_symbol="",
            fractions_per_unit=100000,
            rounding=Rounding(),
        )


class DASHCurrency(Currency):
    """Dash coin — ISO DASH/10006."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Dash coin",
            code="DASH",
            numeric_code=10006,
            symbol="DASH",
            fraction_symbol="",
            fractions_per_unit=100000,
            rounding=Rounding(),
        )


class ZECCurrency(Currency):
    """Zcash — ISO ZEC/10007."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            name="Zcash",
            code="ZEC",
            numeric_code=10007,
            symbol="ZEC",
            fraction_symbol="",
            fractions_per_unit=100000,
            rounding=Rounding(),
        )
