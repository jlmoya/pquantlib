"""Euribor — multi-tenor IBOR index fixed by the ECB.

# C++ parity: ql/indexes/ibor/euribor.{hpp,cpp} (v1.42.1)

C++ exposes one ``Euribor`` parent + one subclass per tenor
(``Euribor1W``, ``Euribor3M``, ``Euribor6M``, ``Euribor1Y``, etc.).
PQuantLib ports as a single ``Euribor(tenor)`` class with classmethod
shortcuts for the common market tenors — Python idiomatic, same C++
public surface.

Conventions:
- Family: ``"Euribor"``.
- Settlement days: 2.
- Currency: EUR.
- Calendar: TARGET.
- Day counter: Actual/360.
- Convention/EOM: Following + EOM=False for Days/Weeks tenors;
  ModifiedFollowing + EOM=True for Months/Years tenors.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _euribor_convention(p: Period) -> BusinessDayConvention:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return BusinessDayConvention.Following
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return BusinessDayConvention.ModifiedFollowing
    qassert.fail("invalid time units")


def _euribor_eom(p: Period) -> bool:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return False
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return True
    qassert.fail("invalid time units")


class Euribor(IborIndex):
    """Euribor — ECB-fixed Euro Interbank Offered Rate."""

    def __init__(
        self,
        tenor: Period,
        forecast_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        qassert.require(
            tenor.units != TimeUnit.Days,
            f"for daily tenors ({tenor}) dedicated DailyTenor constructor must be used",
        )
        super().__init__(
            "Euribor",
            tenor,
            2,
            EURCurrency(),
            TARGET(),
            _euribor_convention(tenor),
            _euribor_eom(tenor),
            Actual360(),
            forecast_term_structure,
        )

    # --- common-tenor classmethod shortcuts --------------------------------

    @classmethod
    def one_week(cls, h: YieldTermStructureProtocol | None = None) -> Euribor:
        return cls(Period(1, TimeUnit.Weeks), h)

    @classmethod
    def two_weeks(cls, h: YieldTermStructureProtocol | None = None) -> Euribor:
        return cls(Period(2, TimeUnit.Weeks), h)

    @classmethod
    def three_weeks(cls, h: YieldTermStructureProtocol | None = None) -> Euribor:
        return cls(Period(3, TimeUnit.Weeks), h)

    @classmethod
    def one_month(cls, h: YieldTermStructureProtocol | None = None) -> Euribor:
        return cls(Period(1, TimeUnit.Months), h)

    @classmethod
    def two_months(cls, h: YieldTermStructureProtocol | None = None) -> Euribor:
        return cls(Period(2, TimeUnit.Months), h)

    @classmethod
    def three_months(cls, h: YieldTermStructureProtocol | None = None) -> Euribor:
        return cls(Period(3, TimeUnit.Months), h)

    @classmethod
    def six_months(cls, h: YieldTermStructureProtocol | None = None) -> Euribor:
        return cls(Period(6, TimeUnit.Months), h)

    @classmethod
    def one_year(cls, h: YieldTermStructureProtocol | None = None) -> Euribor:
        return cls(Period(1, TimeUnit.Years), h)
