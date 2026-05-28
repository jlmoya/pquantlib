"""UK HICP zero-inflation index.

# C++ parity: ql/indexes/inflation/ukhicp.hpp (v1.42.1).

C++ ships only the zero-inflation class; there is no ``YYUKHICP``
sibling upstream, so we don't port one. The L7-A spec lists a
``YoYUKHICP`` placeholder, but matching C++ is the binding rule —
the spec's listing is treated as a documentation slip.
"""

from __future__ import annotations

from pquantlib.currencies.europe import GBPCurrency
from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
from pquantlib.indexes.inflation.region import Region
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class UKHICP(ZeroInflationIndex):
    """UK HICP zero-inflation index. # C++ parity: ``UKHICP`` in ukhicp.hpp."""

    def __init__(self, ts: object | None = None) -> None:
        super().__init__(
            family_name="HICP",
            region=Region.UnitedKingdom,
            revised=False,
            frequency=Frequency.Monthly,
            availability_lag=Period(1, TimeUnit.Months),
            currency=GBPCurrency(),
            ts=ts,
        )
