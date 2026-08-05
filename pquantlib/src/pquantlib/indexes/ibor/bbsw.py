"""Bbsw — Bank Bill Swap rate, fixed by AFMA.

# C++ parity: ql/indexes/ibor/bbsw.hpp (v1.43).

Note the two conventions that set this family apart from its Oceanian
sibling ``Bkbm``: the roll is ``HalfMonthModifiedFollowing`` (not
``ModifiedFollowing``) and there are 0 fixing days. End-of-month is on.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.oceania import AUDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.australia import Australia
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class Bbsw(IborIndex):
    """BBSW — AUD bank bill swap rate; tenor must not be daily."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        qassert.require(
            tenor.units != TimeUnit.Days,
            f"for daily tenors ({tenor}) dedicated DailyTenor constructor must be used",
        )
        super().__init__(
            "Bbsw", tenor, 0, AUDCurrency(), Australia(),
            BusinessDayConvention.HalfMonthModifiedFollowing, True, Actual365Fixed(), h,
        )


class Bbsw1M(Bbsw):
    """1-month BBSW."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(1, TimeUnit.Months), h)


class Bbsw2M(Bbsw):
    """2-month BBSW."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(2, TimeUnit.Months), h)


class Bbsw3M(Bbsw):
    """3-month BBSW."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(3, TimeUnit.Months), h)


class Bbsw4M(Bbsw):
    """4-month BBSW."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(4, TimeUnit.Months), h)


class Bbsw5M(Bbsw):
    """5-month BBSW."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(5, TimeUnit.Months), h)


class Bbsw6M(Bbsw):
    """6-month BBSW."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(6, TimeUnit.Months), h)
