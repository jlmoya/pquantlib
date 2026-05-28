"""UK RPI zero-inflation index + YoY sibling.

# C++ parity: ql/indexes/inflation/ukrpi.hpp (v1.42.1).
"""

from __future__ import annotations

from pquantlib.currencies.europe import GBPCurrency
from pquantlib.indexes.inflation.inflation_index import (
    YoYInflationIndex,
    ZeroInflationIndex,
)
from pquantlib.indexes.inflation.region import Region
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class UKRPI(ZeroInflationIndex):
    """UK Retail Price Inflation Index. # C++ parity: ``UKRPI`` in ukrpi.hpp."""

    def __init__(self, ts: object | None = None) -> None:
        super().__init__(
            family_name="RPI",
            region=Region.UnitedKingdom,
            revised=False,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=GBPCurrency(),
            ts=ts,
        )


class YoYUKRPI(YoYInflationIndex):
    """Quoted year-on-year UK RPI. # C++ parity: ``YYUKRPI`` in ukrpi.hpp."""

    def __init__(self, interpolated: bool = False, ts: object | None = None) -> None:
        super().__init__(
            family_name="YY_RPI",
            region=Region.UnitedKingdom,
            revised=False,
            interpolated=interpolated,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=GBPCurrency(),
            ts=ts,
        )
