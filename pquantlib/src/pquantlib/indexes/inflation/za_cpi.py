"""South African CPI zero-inflation index + YoY sibling.

# C++ parity: ql/indexes/inflation/zacpi.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.africa import ZARCurrency
from pquantlib.indexes.inflation.inflation_index import (
    YoYInflationIndex,
    ZeroInflationIndex,
)
from pquantlib.indexes.inflation.region import ZARegion
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class ZACPI(ZeroInflationIndex):
    """South African CPI zero-inflation index. # C++ parity: ``ZACPI`` in zacpi.hpp."""

    def __init__(self, ts: object | None = None) -> None:
        super().__init__(
            family_name="CPI",
            region=ZARegion(),
            revised=False,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=ZARCurrency(),
            ts=ts,
        )


class YYZACPI(YoYInflationIndex):
    """Quoted year-on-year South African CPI. # C++ parity: ``YYZACPI`` in zacpi.hpp."""

    def __init__(self, interpolated: bool = False, ts: object | None = None) -> None:
        super().__init__(
            family_name="YY_CPI",
            region=ZARegion(),
            revised=False,
            interpolated=interpolated,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=ZARCurrency(),
            ts=ts,
        )
