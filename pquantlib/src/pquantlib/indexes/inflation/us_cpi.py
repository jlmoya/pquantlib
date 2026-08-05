"""US CPI zero-inflation index + YoY sibling.

# C++ parity: ql/indexes/inflation/uscpi.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.america import USDCurrency
from pquantlib.indexes.inflation.inflation_index import (
    YoYInflationIndex,
    ZeroInflationIndex,
)
from pquantlib.indexes.inflation.region import USRegion
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class USCPI(ZeroInflationIndex):
    """US CPI zero-inflation index. # C++ parity: ``USCPI`` in uscpi.hpp."""

    def __init__(self, ts: object | None = None) -> None:
        super().__init__(
            family_name="CPI",
            region=USRegion(),
            revised=False,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=USDCurrency(),
            ts=ts,
        )


class YYUSCPI(YoYInflationIndex):
    """Quoted year-on-year US CPI. # C++ parity: ``YYUSCPI`` in uscpi.hpp."""

    def __init__(self, interpolated: bool = False, ts: object | None = None) -> None:
        super().__init__(
            family_name="YY_CPI",
            region=USRegion(),
            revised=False,
            interpolated=interpolated,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=USDCurrency(),
            ts=ts,
        )
