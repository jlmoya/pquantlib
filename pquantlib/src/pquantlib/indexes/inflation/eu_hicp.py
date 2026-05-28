"""EU HICP zero-inflation index + YoY sibling.

# C++ parity: ql/indexes/inflation/euhicp.hpp (v1.42.1).

Defaults: family_name = ``"HICP"``, region = ``Region.Europe``,
revised = False, frequency = Monthly, availability lag = 1 month,
currency = EUR. The YoY family_name is ``"YY_HICP"``.
"""

from __future__ import annotations

from pquantlib.currencies.europe import EURCurrency
from pquantlib.indexes.inflation.inflation_index import (
    YoYInflationIndex,
    ZeroInflationIndex,
)
from pquantlib.indexes.inflation.region import Region
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class EUHICP(ZeroInflationIndex):
    """EU HICP zero-inflation index. # C++ parity: ``EUHICP`` in euhicp.hpp."""

    def __init__(self, ts: object | None = None) -> None:
        super().__init__(
            family_name="HICP",
            region=Region.Europe,
            revised=False,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=EURCurrency(),
            ts=ts,
        )


class YoYEUHICP(YoYInflationIndex):
    """Quoted year-on-year EU HICP. # C++ parity: ``YYEUHICP`` in euhicp.hpp."""

    def __init__(self, interpolated: bool = False, ts: object | None = None) -> None:
        super().__init__(
            family_name="YY_HICP",
            region=Region.Europe,
            revised=False,
            interpolated=interpolated,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=EURCurrency(),
            ts=ts,
        )
