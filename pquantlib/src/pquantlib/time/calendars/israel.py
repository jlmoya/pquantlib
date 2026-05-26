"""Israel calendar — Tel-Aviv Stock Exchange + Settlement + SHIR markets.

# C++ parity: ql/time/calendars/israel.hpp + .cpp (v1.42.1).

The Tel-Aviv / Settlement markets use a Friday+Saturday weekend, so they
do NOT inherit ``WesternCalendar`` — they override ``_is_weekend``
directly. The SHIR fixing calendar follows the C++ ``WesternImpl`` and
uses a Saturday+Sunday weekend.

Holiday tables (Purim, Passover, Independence Day, Shavuot, Fast Day,
Jewish New Year) are hard-coded from the C++ source.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

from pquantlib import qassert
from pquantlib.time.calendar import Calendar, WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class IsraelMarket(IntEnum):
    Settlement = 0  # generic settlement calendar
    TASE = 1  # Tel-Aviv stock exchange calendar
    SHIR = 2  # SHIR fixing calendar


def _date_set(*ymds: tuple[int, Month, int]) -> frozenset[Date]:
    return frozenset(Date.from_ymd(d, m, y) for d, m, y in ymds)


# C++ parity: ql/time/calendars/israel.cpp ``isPurim`` table.
_PURIM_DATES: Final[frozenset[Date]] = _date_set(
    (21, Month.March, 2000),
    (9, Month.March, 2001),
    (26, Month.February, 2002),
    (18, Month.March, 2003),
    (7, Month.March, 2004),
    (25, Month.March, 2005),
    (14, Month.March, 2006),
    (4, Month.March, 2007),
    (21, Month.March, 2008),
    (10, Month.March, 2009),
    (28, Month.February, 2010),
    (20, Month.March, 2011),
    (8, Month.March, 2012),
    (24, Month.February, 2013),
    (16, Month.March, 2014),
    (5, Month.March, 2015),
    (24, Month.March, 2016),
    (12, Month.March, 2017),
    (1, Month.March, 2018),
    (21, Month.March, 2019),
    (10, Month.March, 2020),
    (26, Month.February, 2021),
    (17, Month.March, 2022),
    (7, Month.March, 2023),
    (24, Month.March, 2024),
    (14, Month.March, 2025),
    (3, Month.March, 2026),
    (23, Month.March, 2027),
    (12, Month.March, 2028),
    (1, Month.March, 2029),
    (19, Month.March, 2030),
    (9, Month.March, 2031),
    (26, Month.February, 2032),
    (15, Month.March, 2033),
    (5, Month.March, 2034),
    (25, Month.March, 2035),
    (13, Month.March, 2036),
    (1, Month.March, 2037),
    (21, Month.March, 2038),
    (10, Month.March, 2039),
    (28, Month.February, 2040),
    (17, Month.March, 2041),
    (6, Month.March, 2042),
    (26, Month.March, 2043),
    (13, Month.March, 2044),
    (3, Month.March, 2045),
    (22, Month.March, 2046),
    (12, Month.March, 2047),
    (28, Month.February, 2048),
    (18, Month.March, 2049),
    (8, Month.March, 2050),
)

# C++ parity: ``isPassover1st`` table.
_PASSOVER_1ST_DATES: Final[frozenset[Date]] = _date_set(
    (20, Month.April, 2000),
    (8, Month.April, 2001),
    (28, Month.March, 2002),
    (17, Month.April, 2003),
    (6, Month.April, 2004),
    (24, Month.April, 2005),
    (13, Month.April, 2006),
    (3, Month.April, 2007),
    (20, Month.April, 2008),
    (9, Month.April, 2009),
    (30, Month.March, 2010),
    (19, Month.April, 2011),
    (7, Month.April, 2012),
    (26, Month.March, 2013),
    (15, Month.April, 2014),
    (4, Month.April, 2015),
    (23, Month.April, 2016),
    (11, Month.April, 2017),
    (31, Month.March, 2018),
    (20, Month.April, 2019),
    (9, Month.April, 2020),
    (28, Month.March, 2021),
    (16, Month.April, 2022),
    (6, Month.April, 2023),
    (23, Month.April, 2024),
    (13, Month.April, 2025),
    (2, Month.April, 2026),
    (22, Month.April, 2027),
    (11, Month.April, 2028),
    (31, Month.March, 2029),
    (18, Month.April, 2030),
    (8, Month.April, 2031),
    (27, Month.March, 2032),
    (14, Month.April, 2033),
    (4, Month.April, 2034),
    (24, Month.April, 2035),
    (12, Month.April, 2036),
    (31, Month.March, 2037),
    (20, Month.April, 2038),
    (9, Month.April, 2039),
    (29, Month.March, 2040),
    (16, Month.April, 2041),
    (5, Month.April, 2042),
    (25, Month.April, 2043),
    (12, Month.April, 2044),
    (2, Month.April, 2045),
    (21, Month.April, 2046),
    (11, Month.April, 2047),
    (29, Month.March, 2048),
    (17, Month.April, 2049),
    (7, Month.April, 2050),
)

# C++ parity: ``isIndependenceDay`` table.
_INDEPENDENCE_DAY_DATES: Final[frozenset[Date]] = _date_set(
    (10, Month.May, 2000),
    (26, Month.April, 2001),
    (17, Month.April, 2002),
    (7, Month.May, 2003),
    (27, Month.April, 2004),
    (12, Month.May, 2005),
    (3, Month.May, 2006),
    (24, Month.April, 2007),
    (8, Month.May, 2008),
    (29, Month.April, 2009),
    (20, Month.April, 2010),
    (10, Month.May, 2011),
    (26, Month.April, 2012),
    (16, Month.April, 2013),
    (6, Month.May, 2014),
    (23, Month.April, 2015),
    (12, Month.May, 2016),
    (2, Month.May, 2017),
    (19, Month.April, 2018),
    (9, Month.May, 2019),
    (29, Month.April, 2020),
    (15, Month.April, 2021),
    (5, Month.May, 2022),
    (26, Month.April, 2023),
    (14, Month.May, 2024),
    (1, Month.May, 2025),
    (22, Month.April, 2026),
    (12, Month.May, 2027),
    (2, Month.May, 2028),
    (19, Month.April, 2029),
    (8, Month.May, 2030),
    (29, Month.April, 2031),
    (15, Month.April, 2032),
    (4, Month.May, 2033),
    (25, Month.April, 2034),
    (15, Month.May, 2035),
    (1, Month.May, 2036),
    (21, Month.April, 2037),
    (10, Month.May, 2038),
    (28, Month.April, 2039),
    (18, Month.April, 2040),
    (7, Month.May, 2041),
    (24, Month.April, 2042),
    (14, Month.May, 2043),
    (3, Month.May, 2044),
    (20, Month.April, 2045),
    (10, Month.May, 2046),
    (1, Month.May, 2047),
    (16, Month.April, 2048),
    (6, Month.May, 2049),
    (27, Month.April, 2050),
)

# C++ parity: ``isShavuot`` table.
_SHAVUOT_DATES: Final[frozenset[Date]] = _date_set(
    (9, Month.June, 2000),
    (28, Month.May, 2001),
    (17, Month.May, 2002),
    (6, Month.June, 2003),
    (26, Month.May, 2004),
    (13, Month.June, 2005),
    (2, Month.June, 2006),
    (23, Month.May, 2007),
    (9, Month.June, 2008),
    (29, Month.May, 2009),
    (19, Month.May, 2010),
    (8, Month.June, 2011),
    (27, Month.May, 2012),
    (15, Month.May, 2013),
    (4, Month.June, 2014),
    (24, Month.May, 2015),
    (12, Month.June, 2016),
    (31, Month.May, 2017),
    (20, Month.May, 2018),
    (9, Month.June, 2019),
    (29, Month.May, 2020),
    (17, Month.May, 2021),
    (5, Month.June, 2022),
    (26, Month.May, 2023),
    (12, Month.June, 2024),
    (2, Month.June, 2025),
    (22, Month.May, 2026),
    (11, Month.June, 2027),
    (31, Month.May, 2028),
    (20, Month.May, 2029),
    (7, Month.June, 2030),
    (28, Month.May, 2031),
    (16, Month.May, 2032),
    (3, Month.June, 2033),
    (24, Month.May, 2034),
    (13, Month.June, 2035),
    (1, Month.June, 2036),
    (20, Month.May, 2037),
    (9, Month.June, 2038),
    (29, Month.May, 2039),
    (18, Month.May, 2040),
    (5, Month.June, 2041),
    (25, Month.May, 2042),
    (14, Month.June, 2043),
    (1, Month.June, 2044),
    (22, Month.May, 2045),
    (10, Month.June, 2046),
    (31, Month.May, 2047),
    (18, Month.May, 2048),
    (6, Month.June, 2049),
    (27, Month.May, 2050),
)

# C++ parity: ``isFastDay`` table.
_FAST_DAY_DATES: Final[frozenset[Date]] = _date_set(
    (10, Month.August, 2000),
    (29, Month.July, 2001),
    (18, Month.July, 2002),
    (7, Month.August, 2003),
    (27, Month.July, 2004),
    (14, Month.August, 2005),
    (3, Month.August, 2006),
    (24, Month.July, 2007),
    (10, Month.August, 2008),
    (30, Month.July, 2009),
    (20, Month.July, 2010),
    (9, Month.August, 2011),
    (29, Month.July, 2012),
    (16, Month.July, 2013),
    (5, Month.August, 2014),
    (26, Month.July, 2015),
    (14, Month.August, 2016),
    (1, Month.August, 2017),
    (22, Month.July, 2018),
    (11, Month.August, 2019),
    (30, Month.July, 2020),
    (18, Month.July, 2021),
    (7, Month.August, 2022),
    (27, Month.July, 2023),
    (13, Month.August, 2024),
    (3, Month.August, 2025),
    (23, Month.July, 2026),
    (12, Month.August, 2027),
    (1, Month.August, 2028),
    (22, Month.July, 2029),
    (8, Month.August, 2030),
    (29, Month.July, 2031),
    (18, Month.July, 2032),
    (4, Month.August, 2033),
    (25, Month.July, 2034),
    (14, Month.August, 2035),
    (3, Month.August, 2036),
    (21, Month.July, 2037),
    (10, Month.August, 2038),
    (31, Month.July, 2039),
    (19, Month.July, 2040),
    (6, Month.August, 2041),
    (27, Month.July, 2042),
    (16, Month.August, 2043),
    (2, Month.August, 2044),
    (23, Month.July, 2045),
    (12, Month.August, 2046),
    (1, Month.August, 2047),
    (19, Month.July, 2048),
    (8, Month.August, 2049),
    (28, Month.July, 2050),
)

# C++ parity: ``isNewYearsDay`` table (Jewish New Year / Rosh Hashanah).
_NEW_YEAR_DATES: Final[frozenset[Date]] = _date_set(
    (30, Month.September, 2000),
    (17, Month.September, 2001),
    (7, Month.September, 2002),
    (27, Month.September, 2003),
    (16, Month.September, 2004),
    (4, Month.October, 2005),
    (23, Month.September, 2006),
    (13, Month.September, 2007),
    (30, Month.September, 2008),
    (19, Month.September, 2009),
    (9, Month.September, 2010),
    (29, Month.September, 2011),
    (17, Month.September, 2012),
    (5, Month.September, 2013),
    (25, Month.September, 2014),
    (14, Month.September, 2015),
    (3, Month.October, 2016),
    (21, Month.September, 2017),
    (10, Month.September, 2018),
    (30, Month.September, 2019),
    (19, Month.September, 2020),
    (7, Month.September, 2021),
    (26, Month.September, 2022),
    (16, Month.September, 2023),
    (3, Month.October, 2024),
    (23, Month.September, 2025),
    (12, Month.September, 2026),
    (2, Month.October, 2027),
    (21, Month.September, 2028),
    (10, Month.September, 2029),
    (28, Month.September, 2030),
    (18, Month.September, 2031),
    (6, Month.September, 2032),
    (24, Month.September, 2033),
    (14, Month.September, 2034),
    (4, Month.October, 2035),
    (22, Month.September, 2036),
    (10, Month.September, 2037),
    (30, Month.September, 2038),
    (19, Month.September, 2039),
    (8, Month.September, 2040),
    (26, Month.September, 2041),
    (15, Month.September, 2042),
    (5, Month.October, 2043),
    (22, Month.September, 2044),
    (12, Month.September, 2045),
    (1, Month.October, 2046),
    (21, Month.September, 2047),
    (8, Month.September, 2048),
    (27, Month.September, 2049),
    (17, Month.September, 2050),
)


def _is_purim(d: Date) -> bool:
    return d in _PURIM_DATES


def _is_passover_1st(d: Date) -> bool:
    return d in _PASSOVER_1ST_DATES


def _is_independence_day(d: Date) -> bool:
    return d in _INDEPENDENCE_DAY_DATES


def _is_memorial_day(d: Date) -> bool:
    # C++ ``isMemorialDay`` = ``isIndependenceDay(d+1)``.
    return _is_independence_day(d + 1)


def _is_shavuot(d: Date) -> bool:
    return d in _SHAVUOT_DATES


def _is_fast_day(d: Date) -> bool:
    return d in _FAST_DAY_DATES


def _is_new_years_day(d: Date) -> bool:
    return d in _NEW_YEAR_DATES


def _is_yom_kippur(d: Date) -> bool:
    # C++ ``isYomKippur`` = ``isNewYearsDay(d - 9)``.
    return _is_new_years_day(d - 9)


def _is_sukkot(d: Date) -> bool:
    # C++ ``isSukkot`` = ``isYomKippur(d - 5)``.
    return _is_yom_kippur(d - 5)


def _is_simchat_torah(d: Date) -> bool:
    # C++ ``isSimchatTorah`` = ``isSukkot(d - 7)``.
    return _is_sukkot(d - 7)


class _IsraelTelAvivCalendar(Calendar):
    """Tel-Aviv Stock Exchange (also used for the Settlement market)."""

    def name(self) -> str:
        return "Tel Aviv stock exchange"

    def _is_weekend(self, w: Weekday) -> bool:
        return w in (Weekday.Friday, Weekday.Saturday)

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        y = d.year()

        return not (
            self._is_weekend(w)
            or _is_purim(d)
            or (y <= 2020 and _is_passover_1st(d + 1))  # Eve of Passover, until 2020
            or _is_passover_1st(d)
            or _is_passover_1st(d - 5)  # Eve of Passover VII, until 2020
            or _is_passover_1st(d - 6)  # Passover VII
            or _is_memorial_day(d)
            or _is_independence_day(d)
            or (y <= 2020 and _is_shavuot(d + 1))  # Eve of Shavuot, until 2020
            or _is_shavuot(d)
            or _is_fast_day(d)
            or (y <= 2019 and _is_new_years_day(d + 1))  # Eve of new year, until 2019
            or _is_new_years_day(d)
            or _is_new_years_day(d - 1)  # 2nd day of new year
            or _is_yom_kippur(d + 1)  # Eve of Yom Kippur
            or _is_yom_kippur(d)
            or _is_sukkot(d + 1)  # Eve of Sukkot
            or _is_sukkot(d)
            or _is_simchat_torah(d + 1)  # Eve of Simchat Torah
            or _is_simchat_torah(d)
        )


class _IsraelShirCalendar(WesternCalendar):
    """SHIR fixing calendar (Sat+Sun weekend — C++ inherits WesternImpl)."""

    def name(self) -> str:
        return "SHIR fixing calendar"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()

        return not (
            self._is_weekend(w)
            or _is_purim(d)
            or _is_purim(d - 1)  # Purim (Jerusalem)
            or _is_passover_1st(d + 1)  # Eve of Passover
            or _is_passover_1st(d)
            or _is_passover_1st(d - 6)  # Last day of Passover
            or _is_independence_day(d)
            or _is_shavuot(d)
            or _is_fast_day(d)
            or _is_new_years_day(d + 1)  # Eve of new year, until 2019
            or _is_new_years_day(d)
            or _is_new_years_day(d - 1)  # 2nd day of new year
            or _is_yom_kippur(d + 1)  # Eve of Yom Kippur
            or _is_yom_kippur(d)
            or _is_sukkot(d)
            or _is_simchat_torah(d)
            # one-off closings
            or (day == 27 and m == Month.February and y == 2024)  # Municipal elections
            # holidays abroad
            or (day == 1 and m == Month.January)  # Western New Year's day
            or dd == WesternCalendar.easter_monday(y) - 3  # Good Friday
            or (day >= 25 and w == Weekday.Monday and m == Month.May and y != 2022)  # Spring Bank
            or (day == 3 and m == Month.June and y == 2022)
            or (day == 25 and m == Month.December)  # Christmas
            or (day == 26 and m == Month.December)  # Boxing day
            # other days when fixings were not published
            or (day == 1 and m == Month.November and y == 2022)  # no idea why
            or (day == 2 and m == Month.January and y == 2023)
            or (day == 10 and m == Month.April and y == 2023)  # Easter Monday in 2023
        )


class Israel(Calendar):
    """Israel calendar dispatching on market (Settlement / TASE / SHIR).

    C++ uses a single ``Israel`` class that swaps the ``impl_`` pointer in
    its constructor. Python composes the chosen sub-calendar (TelAviv for
    Settlement+TASE, SHIR for SHIR) and delegates the protocol methods to
    it. The composed sub-calendars are themselves ``Calendar`` instances.
    """

    def __init__(self, market: IsraelMarket = IsraelMarket.Settlement) -> None:
        super().__init__()
        if market in (IsraelMarket.Settlement, IsraelMarket.TASE):
            self._impl: Calendar = _IsraelTelAvivCalendar()
        elif market == IsraelMarket.SHIR:
            self._impl = _IsraelShirCalendar()
        else:
            qassert.fail("unknown market")
        self._market: IsraelMarket = market

    def name(self) -> str:
        return self._impl.name()

    def _is_weekend(self, w: Weekday) -> bool:
        return self._impl._is_weekend(w)

    def _is_business_day(self, d: Date) -> bool:
        return self._impl._is_business_day(d)
