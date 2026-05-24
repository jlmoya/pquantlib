"""ASX (Australian Securities Exchange) date helpers.

# C++ parity: ql/time/asx.hpp + ql/time/asx.cpp (v1.42.1).

Structurally identical to IMM, but with:
- Day-of-week: Friday (vs Wednesday).
- Day-of-month range: [8, 14] (vs [15, 21]).
- Anchor: 2nd Friday of month (vs 3rd Wednesday).
- Month-letter codes: same alphabet (F..Z) since both follow futures-market
  convention; main cycle is HMUZ (Mar/Jun/Sep/Dec) for both.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday

_LETTER_TO_MONTH: dict[str, Month] = {
    "F": Month.January,
    "G": Month.February,
    "H": Month.March,
    "J": Month.April,
    "K": Month.May,
    "M": Month.June,
    "N": Month.July,
    "Q": Month.August,
    "U": Month.September,
    "V": Month.October,
    "X": Month.November,
    "Z": Month.December,
}
_MONTH_TO_LETTER: dict[Month, str] = {m: ltr for ltr, m in _LETTER_TO_MONTH.items()}
_MAIN_CYCLE_LETTERS: frozenset[str] = frozenset("HMUZ")
_ALL_LETTERS: frozenset[str] = frozenset(_LETTER_TO_MONTH.keys())


def is_asx_date(d: Date, main_cycle: bool = True) -> bool:
    if d.weekday() != Weekday.Friday:
        return False
    day = d.day_of_month()
    if day < 8 or day > 14:
        return False
    if not main_cycle:
        return True
    return d.month() in (Month.March, Month.June, Month.September, Month.December)


def is_asx_code(code: str, main_cycle: bool = True) -> bool:
    if len(code) != 2:
        return False
    if not code[1].isdigit():
        return False
    letter = code[0].upper()
    if main_cycle:
        return letter in _MAIN_CYCLE_LETTERS
    return letter in _ALL_LETTERS


def code(d: Date) -> str:
    qassert.require(is_asx_date(d, main_cycle=False), f"{d} is not an ASX date")
    return f"{_MONTH_TO_LETTER[d.month()]}{d.year() % 10}"


def date(asx_code: str, reference_date: Date | None = None) -> Date:
    qassert.require(is_asx_code(asx_code, main_cycle=False), f"{asx_code} is not a valid ASX code")
    ref = reference_date if reference_date is not None else Date.todays_date()

    upper = asx_code.upper()
    month = _LETTER_TO_MONTH[upper[0]]
    y = int(upper[1])
    if y == 0 and ref.year() <= 1909:
        y += 10
    y += ref.year() - (ref.year() % 10)
    result = next_date(Date.from_ymd(1, month, y), main_cycle=False)
    if result < ref:
        return next_date(Date.from_ymd(1, month, y + 10), main_cycle=False)
    return result


def next_date(d: Date | None = None, main_cycle: bool = True) -> Date:
    ref = d if d is not None else Date.todays_date()
    y = ref.year()
    m = int(ref.month())

    offset = 3 if main_cycle else 1
    skip_months = offset - (m % offset)
    # ASX anchor is day 14 (2nd Friday day-of-month upper bound), not 21.
    if skip_months != offset or ref.day_of_month() > 14:
        skip_months += m
        if skip_months <= 12:
            m = skip_months
        else:
            m = skip_months - 12
            y += 1

    result = Date.nth_weekday(2, Weekday.Friday, Month(m), y)
    if result <= ref:
        result = next_date(Date.from_ymd(15, Month(m), y), main_cycle)
    return result


def next_code(d: Date | None = None, main_cycle: bool = True) -> str:
    return code(next_date(d, main_cycle))
