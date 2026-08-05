"""Bibor — Bangkok Interbank Offered Rate, fixed by the Bank of Thailand.

# C++ parity: ql/indexes/ibor/bibor.{hpp,cpp} (v1.43).

The roll convention and end-of-month flag are tenor-dependent: Days and Weeks
tenors use ``Following`` with EOM off, Months and Years use
``ModifiedFollowing`` with EOM on — the same split the LIBOR and Euribor
families use.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.asia import THBCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.thailand import Thailand
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _bibor_convention(p: Period) -> BusinessDayConvention:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return BusinessDayConvention.Following
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return BusinessDayConvention.ModifiedFollowing
    qassert.fail("invalid time units")


def _bibor_eom(p: Period) -> bool:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return False
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return True
    qassert.fail("invalid time units")


class Bibor(IborIndex):
    """BIBOR — THB interbank offered rate; 2 fixing days; tenor must not be daily."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        qassert.require(
            tenor.units != TimeUnit.Days,
            f"for daily tenors ({tenor}) dedicated DailyTenor constructor must be used",
        )
        super().__init__(
            "Bibor", tenor, 2, THBCurrency(), Thailand(),
            _bibor_convention(tenor), _bibor_eom(tenor), Actual365Fixed(), h,
        )


class BiborSW(Bibor):
    """1-week BIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(1, TimeUnit.Weeks), h)


class Bibor1M(Bibor):
    """1-month BIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(1, TimeUnit.Months), h)


class Bibor2M(Bibor):
    """2-month BIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(2, TimeUnit.Months), h)


class Bibor3M(Bibor):
    """3-month BIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(3, TimeUnit.Months), h)


class Bibor6M(Bibor):
    """6-month BIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(6, TimeUnit.Months), h)


class Bibor1Y(Bibor):
    """1-year BIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(1, TimeUnit.Years), h)
