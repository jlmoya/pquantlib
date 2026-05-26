"""Japan calendar.

# C++ parity: ql/time/calendars/japan.hpp + .cpp (v1.42.1).

Sat+Sun weekend (``WesternCalendar``). The equinox-day calculation
mirrors the C++ formula verbatim, including its ``Day(double)``
truncation — ``int(x)`` in Python performs the same toward-zero
truncation as ``Day`` (which is an integral type).
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday

# C++ parity: ``Japan::Impl::isBusinessDay`` constants.
_EXACT_VERNAL_EQUINOX_TIME: float = 20.69115
_EXACT_AUTUMNAL_EQUINOX_TIME: float = 23.09
_DIFF_PER_YEAR: float = 0.242194


def _vernal_equinox_day(year: int) -> int:
    # C++ ``Day ve = Day(exact_vernal_equinox_time + moving_amount - number_of_leap_years);``
    moving_amount = (year - 2000) * _DIFF_PER_YEAR
    number_of_leap_years = (year - 2000) // 4 + (year - 2000) // 100 - (year - 2000) // 400
    return int(_EXACT_VERNAL_EQUINOX_TIME + moving_amount - number_of_leap_years)


def _autumnal_equinox_day(year: int) -> int:
    # C++ ``Day ae = Day(exact_autumnal_equinox_time + moving_amount - number_of_leap_years);``
    moving_amount = (year - 2000) * _DIFF_PER_YEAR
    number_of_leap_years = (year - 2000) // 4 + (year - 2000) // 100 - (year - 2000) // 400
    return int(_EXACT_AUTUMNAL_EQUINOX_TIME + moving_amount - number_of_leap_years)


class Japan(WesternCalendar):
    """Japanese national + bank-holiday calendar."""

    def name(self) -> str:
        return "Japan"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        day = d.day_of_month()
        m = d.month()
        y = d.year()
        ve = _vernal_equinox_day(y)
        ae = _autumnal_equinox_day(y)

        return not (
            self._is_weekend(w)
            # New Year's Day
            or (day == 1 and m == Month.January)
            # Bank Holiday
            or (day == 2 and m == Month.January)
            # Bank Holiday
            or (day == 3 and m == Month.January)
            # Coming of Age Day (2nd Monday in January), was January 15th until 2000
            or (w == Weekday.Monday and 8 <= day <= 14 and m == Month.January and y >= 2000)
            or ((day == 15 or (day == 16 and w == Weekday.Monday)) and m == Month.January and y < 2000)
            # National Foundation Day
            or ((day == 11 or (day == 12 and w == Weekday.Monday)) and m == Month.February)
            # Emperor's Birthday (Emperor Naruhito)
            or ((day == 23 or (day == 24 and w == Weekday.Monday)) and m == Month.February and y >= 2020)
            # Emperor's Birthday (Emperor Akihito)
            or (
                (day == 23 or (day == 24 and w == Weekday.Monday))
                and m == Month.December
                and 1989 <= y < 2019
            )
            # Vernal Equinox
            or ((day == ve or (day == ve + 1 and w == Weekday.Monday)) and m == Month.March)
            # Greenery Day
            or ((day == 29 or (day == 30 and w == Weekday.Monday)) and m == Month.April)
            # Constitution Memorial Day
            or (day == 3 and m == Month.May)
            # Holiday for a Nation
            or (day == 4 and m == Month.May)
            # Children's Day
            or (day == 5 and m == Month.May)
            # any of the three above observed later if on Saturday or Sunday
            or (day == 6 and m == Month.May and w in (Weekday.Monday, Weekday.Tuesday, Weekday.Wednesday))
            # Marine Day (3rd Monday in July), was July 20th until 2003,
            # not a holiday before 1996, special dates in 2020/2021 (Olympics)
            or (
                w == Weekday.Monday
                and 15 <= day <= 21
                and m == Month.July
                and (2003 <= y < 2020 or y >= 2022)
            )
            or ((day == 20 or (day == 21 and w == Weekday.Monday)) and m == Month.July and 1996 <= y < 2003)
            or (day == 23 and m == Month.July and y == 2020)
            or (day == 22 and m == Month.July and y == 2021)
            # Mountain Day (moved in 2020/2021 due to Olympics)
            or (
                (day == 11 or (day == 12 and w == Weekday.Monday))
                and m == Month.August
                and (2016 <= y < 2020 or y >= 2022)
            )
            or (day == 10 and m == Month.August and y == 2020)
            or (day == 9 and m == Month.August and y == 2021)
            # Respect for the Aged Day (3rd Monday in September), was September 15th until 2003
            or (w == Weekday.Monday and 15 <= day <= 21 and m == Month.September and y >= 2003)
            or ((day == 15 or (day == 16 and w == Weekday.Monday)) and m == Month.September and y < 2003)
            # If a single day falls between Respect for the Aged Day and the Autumnal Equinox,
            # it is also a holiday
            or (
                w == Weekday.Tuesday
                and day + 1 == ae
                and 16 <= day <= 22
                and m == Month.September
                and y >= 2003
            )
            # Autumnal Equinox
            or ((day == ae or (day == ae + 1 and w == Weekday.Monday)) and m == Month.September)
            # Health and Sports Day (2nd Monday in October), was October 10th until 2000,
            # special dates in 2020/2021 (Olympics)
            or (
                w == Weekday.Monday
                and 8 <= day <= 14
                and m == Month.October
                and (2000 <= y < 2020 or y >= 2022)
            )
            or ((day == 10 or (day == 11 and w == Weekday.Monday)) and m == Month.October and y < 2000)
            or (day == 24 and m == Month.July and y == 2020)
            or (day == 23 and m == Month.July and y == 2021)
            # National Culture Day
            or ((day == 3 or (day == 4 and w == Weekday.Monday)) and m == Month.November)
            # Labor Thanksgiving Day
            or ((day == 23 or (day == 24 and w == Weekday.Monday)) and m == Month.November)
            # Bank Holiday
            or (day == 31 and m == Month.December)
            # one-shot holidays
            # Marriage of Prince Akihito
            or (day == 10 and m == Month.April and y == 1959)
            # Rites of Imperial Funeral
            or (day == 24 and m == Month.February and y == 1989)
            # Enthronement Ceremony (Emperor Akihito)
            or (day == 12 and m == Month.November and y == 1990)
            # Marriage of Prince Naruhito
            or (day == 9 and m == Month.June and y == 1993)
            # Special holiday based on Japanese public holidays law
            or (day == 30 and m == Month.April and y == 2019)
            # Enthronement Day (Emperor Naruhito)
            or (day == 1 and m == Month.May and y == 2019)
            # Special holiday based on Japanese public holidays law
            or (day == 2 and m == Month.May and y == 2019)
            # Enthronement Ceremony (Emperor Naruhito)
            or (day == 22 and m == Month.October and y == 2019)
        )
