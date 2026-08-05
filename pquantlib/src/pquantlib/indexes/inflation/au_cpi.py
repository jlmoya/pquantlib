"""Australian CPI zero-inflation index + YoY sibling.

# C++ parity: ql/indexes/inflation/aucpi.hpp (v1.43).

Unlike the European and American families, the Australian CPI takes its
frequency and revision flag from the caller — the ABS publishes quarterly, but
the C++ constructor leaves the choice open. The availability lag is 2 months,
not the 1 month the monthly-published families use.
"""

from __future__ import annotations

from pquantlib.currencies.oceania import AUDCurrency
from pquantlib.indexes.inflation.inflation_index import (
    YoYInflationIndex,
    ZeroInflationIndex,
)
from pquantlib.indexes.inflation.region import AustraliaRegion
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_LAG = Period(2, TimeUnit.Months)


class AUCPI(ZeroInflationIndex):
    """Australian CPI zero-inflation index. # C++ parity: ``AUCPI`` in aucpi.hpp."""

    def __init__(
        self,
        frequency: Frequency,
        revised: bool,
        ts: object | None = None,
    ) -> None:
        super().__init__(
            family_name="CPI",
            region=AustraliaRegion(),
            revised=revised,
            frequency=frequency,
            availability_lag=_LAG,
            currency=AUDCurrency(),
            ts=ts,
        )


class YYAUCPI(YoYInflationIndex):
    """Quoted year-on-year Australian CPI. # C++ parity: ``YYAUCPI`` in aucpi.hpp."""

    def __init__(
        self,
        frequency: Frequency,
        revised: bool,
        interpolated: bool = False,
        ts: object | None = None,
    ) -> None:
        super().__init__(
            family_name="YY_CPI",
            region=AustraliaRegion(),
            revised=revised,
            interpolated=interpolated,
            frequency=frequency,
            availability_lag=_LAG,
            currency=AUDCurrency(),
            ts=ts,
        )
