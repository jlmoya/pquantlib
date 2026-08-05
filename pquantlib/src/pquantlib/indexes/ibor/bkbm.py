"""Bkbm — Bank Bill Benchmark rate, fixed by NZFMA.

# C++ parity: ql/indexes/ibor/bkbm.hpp (v1.43).

0 fixing days, ``ModifiedFollowing``, end-of-month on — note the roll differs
from the Australian ``Bbsw``, which uses ``HalfMonthModifiedFollowing``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.oceania import NZDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.new_zealand import NewZealand
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class Bkbm(IborIndex):
    """BKBM — NZD bank bill benchmark rate; tenor must not be daily."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        qassert.require(
            tenor.units != TimeUnit.Days,
            f"for daily tenors ({tenor}) dedicated DailyTenor constructor must be used",
        )
        super().__init__(
            "Bkbm", tenor, 0, NZDCurrency(), NewZealand(),
            BusinessDayConvention.ModifiedFollowing, True, Actual365Fixed(), h,
        )


class Bkbm1M(Bkbm):
    """1-month BKBM."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(1, TimeUnit.Months), h)


class Bkbm2M(Bkbm):
    """2-month BKBM."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(2, TimeUnit.Months), h)


class Bkbm3M(Bkbm):
    """3-month BKBM."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(3, TimeUnit.Months), h)


class Bkbm4M(Bkbm):
    """4-month BKBM."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(4, TimeUnit.Months), h)


class Bkbm5M(Bkbm):
    """5-month BKBM."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(5, TimeUnit.Months), h)


class Bkbm6M(Bkbm):
    """6-month BKBM."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(6, TimeUnit.Months), h)
