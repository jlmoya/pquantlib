"""Chinese calendar — Shanghai stock exchange (SSE) and interbank market (IB).

# C++ parity: ql/time/calendars/china.hpp + china.cpp (v1.42.1).

The C++ implementation uses two pImpl classes (``SseImpl``, ``IbImpl``) where
``IbImpl`` holds a reference to an ``SseImpl`` instance. The Python port
collapses these into a single class dispatching on ``self._market`` inside
``_is_business_day`` and ``name()``. The IB market is the SSE business-day
set plus an explicit "working weekends" override list.

Although the holidays follow lunar/Chinese cultural rules, the C++ class
is built on ``Calendar`` (not ``WesternImpl``), but its ``isWeekend`` is
Sat+Sun — so for the Python port we use ``WesternCalendar``. The Easter
table is unused.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class China(WesternCalendar):
    """Chinese calendar (SSE / IB markets)."""

    class Market(IntEnum):
        SSE = 0  # Shanghai stock exchange
        IB = 1  # Interbank market

    def __init__(self, market: Market = Market.SSE) -> None:
        super().__init__()
        qassert.require(market in (China.Market.SSE, China.Market.IB), "unknown market")
        self._market = market

    def name(self) -> str:
        if self._market == China.Market.IB:
            return "China inter bank market"
        return "Shanghai stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        if self._market == China.Market.IB:
            # If it is already a SSE business day, it must be an IB business day.
            if _sse_is_business_day(d):
                return True
            return d in _IB_WORKING_WEEKENDS
        # SSE
        return _sse_is_business_day(d)


def _sse_is_business_day(date: Date) -> bool:
    """Mirrors C++ ``China::SseImpl::isBusinessDay``."""
    w = date.weekday()
    if w in (Weekday.Saturday, Weekday.Sunday):
        return False
    d = date.day_of_month()
    m = date.month()
    y = date.year()

    return not (
        # New Year's Day
        (d == 1 and m == Month.January)
        or (y == 2005 and d == 3 and m == Month.January)
        or (y == 2006 and (d in {2, 3}) and m == Month.January)
        or (y == 2007 and d <= 3 and m == Month.January)
        or (y == 2007 and d == 31 and m == Month.December)
        or (y == 2009 and d == 2 and m == Month.January)
        or (y == 2011 and d == 3 and m == Month.January)
        or (y == 2012 and (d in {2, 3}) and m == Month.January)
        or (y == 2013 and d <= 3 and m == Month.January)
        or (y == 2014 and d == 1 and m == Month.January)
        or (y == 2015 and d <= 3 and m == Month.January)
        or (y == 2017 and d == 2 and m == Month.January)
        or (y == 2018 and d == 1 and m == Month.January)
        or (y == 2018 and d == 31 and m == Month.December)
        or (y == 2019 and d == 1 and m == Month.January)
        or (y == 2020 and d == 1 and m == Month.January)
        or (y == 2021 and d == 1 and m == Month.January)
        or (y == 2022 and d == 3 and m == Month.January)
        or (y == 2023 and d == 2 and m == Month.January)
        or (y == 2026 and (d in {1, 2}) and m == Month.January)
        # Chinese New Year
        or (y == 2004 and 19 <= d <= 28 and m == Month.January)
        or (y == 2005 and 7 <= d <= 15 and m == Month.February)
        or (y == 2006 and ((d >= 26 and m == Month.January) or (d <= 3 and m == Month.February)))
        or (y == 2007 and 17 <= d <= 25 and m == Month.February)
        or (y == 2008 and 6 <= d <= 12 and m == Month.February)
        or (y == 2009 and 26 <= d <= 30 and m == Month.January)
        or (y == 2010 and 15 <= d <= 19 and m == Month.February)
        or (y == 2011 and 2 <= d <= 8 and m == Month.February)
        or (y == 2012 and 23 <= d <= 28 and m == Month.January)
        or (y == 2013 and 11 <= d <= 15 and m == Month.February)
        or (y == 2014 and d >= 31 and m == Month.January)
        or (y == 2014 and d <= 6 and m == Month.February)
        or (y == 2015 and 18 <= d <= 24 and m == Month.February)
        or (y == 2016 and 8 <= d <= 12 and m == Month.February)
        or (y == 2017 and ((d >= 27 and m == Month.January) or (d <= 2 and m == Month.February)))
        or (y == 2018 and (15 <= d <= 21 and m == Month.February))
        or (y == 2019 and 4 <= d <= 8 and m == Month.February)
        or (y == 2020 and (d == 24 or (27 <= d <= 31)) and m == Month.January)
        or (y == 2021 and d in (11, 12, 15, 16, 17) and m == Month.February)
        or (y == 2022 and ((d == 31 and m == Month.January) or (d <= 4 and m == Month.February)))
        or (y == 2023 and 23 <= d <= 27 and m == Month.January)
        or (y == 2024 and (d == 9 or (12 <= d <= 16)) and m == Month.February)
        or (y == 2025 and ((28 <= d <= 31 and m == Month.January) or (3 <= d <= 4 and m == Month.February)))
        or (y == 2026 and ((16 <= d <= 20) or d == 23) and m == Month.February)
        # Ching Ming Festival
        or (y <= 2008 and d == 4 and m == Month.April)
        or (y == 2009 and d == 6 and m == Month.April)
        or (y == 2010 and d == 5 and m == Month.April)
        or (y == 2011 and 3 <= d <= 5 and m == Month.April)
        or (y == 2012 and 2 <= d <= 4 and m == Month.April)
        or (y == 2013 and 4 <= d <= 5 and m == Month.April)
        or (y == 2014 and d == 7 and m == Month.April)
        or (y == 2015 and 5 <= d <= 6 and m == Month.April)
        or (y == 2016 and d == 4 and m == Month.April)
        or (y == 2017 and 3 <= d <= 4 and m == Month.April)
        or (y == 2018 and 5 <= d <= 6 and m == Month.April)
        or (y == 2019 and d == 5 and m == Month.April)
        or (y == 2020 and d == 6 and m == Month.April)
        or (y == 2021 and d == 5 and m == Month.April)
        or (y == 2022 and 4 <= d <= 5 and m == Month.April)
        or (y == 2023 and d == 5 and m == Month.April)
        or (y == 2024 and 4 <= d <= 5 and m == Month.April)
        or (y == 2025 and d == 4 and m == Month.April)
        or (y == 2026 and d == 6 and m == Month.April)
        # Labor Day
        or (y <= 2007 and 1 <= d <= 7 and m == Month.May)
        or (y == 2008 and 1 <= d <= 2 and m == Month.May)
        or (y == 2009 and d == 1 and m == Month.May)
        or (y == 2010 and d == 3 and m == Month.May)
        or (y == 2011 and d == 2 and m == Month.May)
        or (y == 2012 and ((d == 30 and m == Month.April) or (d == 1 and m == Month.May)))
        or (y == 2013 and ((d >= 29 and m == Month.April) or (d == 1 and m == Month.May)))
        or (y == 2014 and 1 <= d <= 3 and m == Month.May)
        or (y == 2015 and d == 1 and m == Month.May)
        or (y == 2016 and 1 <= d <= 2 and m == Month.May)
        or (y == 2017 and d == 1 and m == Month.May)
        or (y == 2018 and ((d == 30 and m == Month.April) or (d == 1 and m == Month.May)))
        or (y == 2019 and 1 <= d <= 3 and m == Month.May)
        or (y == 2020 and d in (1, 4, 5) and m == Month.May)
        or (y == 2021 and d in (3, 4, 5) and m == Month.May)
        or (y == 2022 and 2 <= d <= 4 and m == Month.May)
        or (y == 2023 and 1 <= d <= 3 and m == Month.May)
        or (y == 2024 and 1 <= d <= 3 and m == Month.May)
        or (y == 2025 and d in (1, 2, 5) and m == Month.May)
        or (y == 2026 and d in (1, 4, 5) and m == Month.May)
        # Tuen Ng Festival
        or (y <= 2008 and d == 9 and m == Month.June)
        or (y == 2009 and (d in {28, 29}) and m == Month.May)
        or (y == 2010 and 14 <= d <= 16 and m == Month.June)
        or (y == 2011 and 4 <= d <= 6 and m == Month.June)
        or (y == 2012 and 22 <= d <= 24 and m == Month.June)
        or (y == 2013 and 10 <= d <= 12 and m == Month.June)
        or (y == 2014 and d == 2 and m == Month.June)
        or (y == 2015 and d == 22 and m == Month.June)
        or (y == 2016 and 9 <= d <= 10 and m == Month.June)
        or (y == 2017 and 29 <= d <= 30 and m == Month.May)
        or (y == 2018 and d == 18 and m == Month.June)
        or (y == 2019 and d == 7 and m == Month.June)
        or (y == 2020 and 25 <= d <= 26 and m == Month.June)
        or (y == 2021 and d == 14 and m == Month.June)
        or (y == 2022 and d == 3 and m == Month.June)
        or (y == 2023 and 22 <= d <= 23 and m == Month.June)
        or (y == 2024 and d == 10 and m == Month.June)
        or (y == 2025 and d == 2 and m == Month.June)
        or (y == 2026 and d == 19 and m == Month.June)
        # Mid-Autumn Festival
        or (y <= 2008 and d == 15 and m == Month.September)
        or (y == 2010 and 22 <= d <= 24 and m == Month.September)
        or (y == 2011 and 10 <= d <= 12 and m == Month.September)
        or (y == 2012 and d == 30 and m == Month.September)
        or (y == 2013 and 19 <= d <= 20 and m == Month.September)
        or (y == 2014 and d == 8 and m == Month.September)
        or (y == 2015 and d == 27 and m == Month.September)
        or (y == 2016 and 15 <= d <= 16 and m == Month.September)
        or (y == 2018 and d == 24 and m == Month.September)
        or (y == 2019 and d == 13 and m == Month.September)
        or (y == 2021 and (d in {20, 21}) and m == Month.September)
        or (y == 2022 and d == 12 and m == Month.September)
        or (y == 2023 and d == 29 and m == Month.September)
        or (y == 2024 and 16 <= d <= 17 and m == Month.September)
        or (y == 2026 and d == 25 and m == Month.September)
        # National Day
        or (y <= 2007 and 1 <= d <= 7 and m == Month.October)
        or (y == 2008 and ((d >= 29 and m == Month.September) or (d <= 3 and m == Month.October)))
        or (y == 2009 and 1 <= d <= 8 and m == Month.October)
        or (y == 2010 and 1 <= d <= 7 and m == Month.October)
        or (y == 2011 and 1 <= d <= 7 and m == Month.October)
        or (y == 2012 and 1 <= d <= 7 and m == Month.October)
        or (y == 2013 and 1 <= d <= 7 and m == Month.October)
        or (y == 2014 and 1 <= d <= 7 and m == Month.October)
        or (y == 2015 and 1 <= d <= 7 and m == Month.October)
        or (y == 2016 and 3 <= d <= 7 and m == Month.October)
        or (y == 2017 and 2 <= d <= 6 and m == Month.October)
        or (y == 2018 and 1 <= d <= 5 and m == Month.October)
        or (y == 2019 and 1 <= d <= 7 and m == Month.October)
        or (y == 2020 and 1 <= d <= 2 and m == Month.October)
        or (y == 2020 and 5 <= d <= 8 and m == Month.October)
        or (y == 2021 and d in (1, 4, 5, 6, 7) and m == Month.October)
        or (y == 2022 and 3 <= d <= 7 and m == Month.October)
        or (y == 2023 and 2 <= d <= 6 and m == Month.October)
        or (y == 2024 and ((1 <= d <= 4) or d == 7) and m == Month.October)
        or (y == 2025 and ((1 <= d <= 3) or (6 <= d <= 8)) and m == Month.October)
        or (y == 2026 and ((1 <= d <= 2) or (5 <= d <= 7)) and m == Month.October)
        # 70th anniversary of the victory of anti-Japanese war
        or (y == 2015 and 3 <= d <= 4 and m == Month.September)
    )


# C++ parity: ql/time/calendars/china.cpp ``China::IbImpl::isBusinessDay``
# static const std::set<Date> workingWeekends — verbatim port.
_IB_WORKING_WEEKENDS: frozenset[Date] = frozenset(
    {
        # 2005
        Date.from_ymd(5, Month.February, 2005),
        Date.from_ymd(6, Month.February, 2005),
        Date.from_ymd(30, Month.April, 2005),
        Date.from_ymd(8, Month.May, 2005),
        Date.from_ymd(8, Month.October, 2005),
        Date.from_ymd(9, Month.October, 2005),
        Date.from_ymd(31, Month.December, 2005),
        # 2006
        Date.from_ymd(28, Month.January, 2006),
        Date.from_ymd(29, Month.April, 2006),
        Date.from_ymd(30, Month.April, 2006),
        Date.from_ymd(30, Month.September, 2006),
        Date.from_ymd(30, Month.December, 2006),
        Date.from_ymd(31, Month.December, 2006),
        # 2007
        Date.from_ymd(17, Month.February, 2007),
        Date.from_ymd(25, Month.February, 2007),
        Date.from_ymd(28, Month.April, 2007),
        Date.from_ymd(29, Month.April, 2007),
        Date.from_ymd(29, Month.September, 2007),
        Date.from_ymd(30, Month.September, 2007),
        Date.from_ymd(29, Month.December, 2007),
        # 2008
        Date.from_ymd(2, Month.February, 2008),
        Date.from_ymd(3, Month.February, 2008),
        Date.from_ymd(4, Month.May, 2008),
        Date.from_ymd(27, Month.September, 2008),
        Date.from_ymd(28, Month.September, 2008),
        # 2009
        Date.from_ymd(4, Month.January, 2009),
        Date.from_ymd(24, Month.January, 2009),
        Date.from_ymd(1, Month.February, 2009),
        Date.from_ymd(31, Month.May, 2009),
        Date.from_ymd(27, Month.September, 2009),
        Date.from_ymd(10, Month.October, 2009),
        # 2010
        Date.from_ymd(20, Month.February, 2010),
        Date.from_ymd(21, Month.February, 2010),
        Date.from_ymd(12, Month.June, 2010),
        Date.from_ymd(13, Month.June, 2010),
        Date.from_ymd(19, Month.September, 2010),
        Date.from_ymd(25, Month.September, 2010),
        Date.from_ymd(26, Month.September, 2010),
        Date.from_ymd(9, Month.October, 2010),
        # 2011
        Date.from_ymd(30, Month.January, 2011),
        Date.from_ymd(12, Month.February, 2011),
        Date.from_ymd(2, Month.April, 2011),
        Date.from_ymd(8, Month.October, 2011),
        Date.from_ymd(9, Month.October, 2011),
        Date.from_ymd(31, Month.December, 2011),
        # 2012
        Date.from_ymd(21, Month.January, 2012),
        Date.from_ymd(29, Month.January, 2012),
        Date.from_ymd(31, Month.March, 2012),
        Date.from_ymd(1, Month.April, 2012),
        Date.from_ymd(28, Month.April, 2012),
        Date.from_ymd(29, Month.September, 2012),
        # 2013
        Date.from_ymd(5, Month.January, 2013),
        Date.from_ymd(6, Month.January, 2013),
        Date.from_ymd(16, Month.February, 2013),
        Date.from_ymd(17, Month.February, 2013),
        Date.from_ymd(7, Month.April, 2013),
        Date.from_ymd(27, Month.April, 2013),
        Date.from_ymd(28, Month.April, 2013),
        Date.from_ymd(8, Month.June, 2013),
        Date.from_ymd(9, Month.June, 2013),
        Date.from_ymd(22, Month.September, 2013),
        Date.from_ymd(29, Month.September, 2013),
        Date.from_ymd(12, Month.October, 2013),
        # 2014
        Date.from_ymd(26, Month.January, 2014),
        Date.from_ymd(8, Month.February, 2014),
        Date.from_ymd(4, Month.May, 2014),
        Date.from_ymd(28, Month.September, 2014),
        Date.from_ymd(11, Month.October, 2014),
        # 2015
        Date.from_ymd(4, Month.January, 2015),
        Date.from_ymd(15, Month.February, 2015),
        Date.from_ymd(28, Month.February, 2015),
        Date.from_ymd(6, Month.September, 2015),
        Date.from_ymd(10, Month.October, 2015),
        # 2016
        Date.from_ymd(6, Month.February, 2016),
        Date.from_ymd(14, Month.February, 2016),
        Date.from_ymd(12, Month.June, 2016),
        Date.from_ymd(18, Month.September, 2016),
        Date.from_ymd(8, Month.October, 2016),
        Date.from_ymd(9, Month.October, 2016),
        # 2017
        Date.from_ymd(22, Month.January, 2017),
        Date.from_ymd(4, Month.February, 2017),
        Date.from_ymd(1, Month.April, 2017),
        Date.from_ymd(27, Month.May, 2017),
        Date.from_ymd(30, Month.September, 2017),
        # 2018
        Date.from_ymd(11, Month.February, 2018),
        Date.from_ymd(24, Month.February, 2018),
        Date.from_ymd(8, Month.April, 2018),
        Date.from_ymd(28, Month.April, 2018),
        Date.from_ymd(29, Month.September, 2018),
        Date.from_ymd(30, Month.September, 2018),
        Date.from_ymd(29, Month.December, 2018),
        # 2019
        Date.from_ymd(2, Month.February, 2019),
        Date.from_ymd(3, Month.February, 2019),
        Date.from_ymd(28, Month.April, 2019),
        Date.from_ymd(5, Month.May, 2019),
        Date.from_ymd(29, Month.September, 2019),
        Date.from_ymd(12, Month.October, 2019),
        # 2020
        Date.from_ymd(19, Month.January, 2020),
        Date.from_ymd(26, Month.April, 2020),
        Date.from_ymd(9, Month.May, 2020),
        Date.from_ymd(28, Month.June, 2020),
        Date.from_ymd(27, Month.September, 2020),
        Date.from_ymd(10, Month.October, 2020),
        # 2021
        Date.from_ymd(7, Month.February, 2021),
        Date.from_ymd(20, Month.February, 2021),
        Date.from_ymd(25, Month.April, 2021),
        Date.from_ymd(8, Month.May, 2021),
        Date.from_ymd(18, Month.September, 2021),
        Date.from_ymd(26, Month.September, 2021),
        Date.from_ymd(9, Month.October, 2021),
        # 2022
        Date.from_ymd(29, Month.January, 2022),
        Date.from_ymd(30, Month.January, 2022),
        Date.from_ymd(2, Month.April, 2022),
        Date.from_ymd(24, Month.April, 2022),
        Date.from_ymd(7, Month.May, 2022),
        Date.from_ymd(8, Month.October, 2022),
        Date.from_ymd(9, Month.October, 2022),
        # 2023
        Date.from_ymd(28, Month.January, 2023),
        Date.from_ymd(29, Month.January, 2023),
        Date.from_ymd(23, Month.April, 2023),
        Date.from_ymd(6, Month.May, 2023),
        Date.from_ymd(25, Month.June, 2023),
        Date.from_ymd(7, Month.October, 2023),
        Date.from_ymd(8, Month.October, 2023),
        # 2024
        Date.from_ymd(4, Month.February, 2024),
        Date.from_ymd(9, Month.February, 2024),
        Date.from_ymd(18, Month.February, 2024),
        Date.from_ymd(7, Month.April, 2024),
        Date.from_ymd(28, Month.April, 2024),
        Date.from_ymd(11, Month.May, 2024),
        Date.from_ymd(14, Month.September, 2024),
        Date.from_ymd(29, Month.September, 2024),
        Date.from_ymd(12, Month.October, 2024),
        # 2025
        Date.from_ymd(26, Month.January, 2025),
        Date.from_ymd(8, Month.February, 2025),
        Date.from_ymd(27, Month.April, 2025),
        Date.from_ymd(28, Month.September, 2025),
        Date.from_ymd(11, Month.October, 2025),
        # 2026
        Date.from_ymd(4, Month.January, 2026),
        Date.from_ymd(14, Month.February, 2026),
        Date.from_ymd(28, Month.February, 2026),
        Date.from_ymd(9, Month.May, 2026),
        Date.from_ymd(20, Month.September, 2026),
        Date.from_ymd(10, Month.October, 2026),
    }
)
