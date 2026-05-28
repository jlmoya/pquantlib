"""FR HICP zero-inflation index + YoY sibling.

# C++ parity: ql/indexes/inflation/frhicp.hpp (v1.42.1).
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


class FRHICP(ZeroInflationIndex):
    """FR HICP zero-inflation index. # C++ parity: ``FRHICP`` in frhicp.hpp."""

    def __init__(self, ts: object | None = None) -> None:
        super().__init__(
            family_name="HICP",
            region=Region.France,
            revised=False,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=EURCurrency(),
            ts=ts,
        )


class YoYFRHICP(YoYInflationIndex):
    """Quoted year-on-year FR HICP. # C++ parity: ``YYFRHICP`` in frhicp.hpp."""

    def __init__(self, interpolated: bool = False, ts: object | None = None) -> None:
        super().__init__(
            family_name="YY_HICP",
            region=Region.France,
            revised=False,
            interpolated=interpolated,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=EURCurrency(),
            ts=ts,
        )
