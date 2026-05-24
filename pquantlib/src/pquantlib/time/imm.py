"""IMM (International Money Market) date helpers.

# C++ parity: ql/time/imm.hpp + ql/time/imm.cpp (v1.42.1).

C++ exposes these as static methods on a ``struct IMM``; the Python port
flattens them to module-level free functions, matching the parsers/IMM
pattern (no shared state to encapsulate).

The IMM main cycle is the four months {March, June, September, December};
the non-main cycle is any month. An IMM date is the 3rd Wednesday of the
relevant month (i.e. ``Date.nth_weekday(3, Wednesday, m, y)``).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday

# Month-letter mapping (futures market convention).
# C++ parity: ql/time/imm.hpp enum Month { F..Z }.
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

_MAIN_CYCLE_LETTERS: frozenset[str] = frozenset("HMZU")
_ALL_LETTERS: frozenset[str] = frozenset(_LETTER_TO_MONTH.keys())


def is_imm_date(d: Date, main_cycle: bool = True) -> bool:
    """Whether ``d`` is an IMM date (3rd Wednesday of month, day in [15, 21])."""
    if d.weekday() != Weekday.Wednesday:
        return False
    day = d.day_of_month()
    if day < 15 or day > 21:
        return False
    if not main_cycle:
        return True
    return d.month() in (Month.March, Month.June, Month.September, Month.December)


def is_imm_code(code: str, main_cycle: bool = True) -> bool:
    """Whether ``code`` is a valid IMM code (e.g. ``"H3"`` for Mar 2013)."""
    if len(code) != 2:
        return False
    if not code[1].isdigit():
        return False
    letter = code[0].upper()
    if main_cycle:
        return letter in _MAIN_CYCLE_LETTERS
    return letter in _ALL_LETTERS


def code(d: Date) -> str:
    """IMM code for ``d``. Raises if ``d`` is not an IMM date."""
    qassert.require(is_imm_date(d, main_cycle=False), f"{d} is not an IMM date")
    letter = _MONTH_TO_LETTER[d.month()]
    return f"{letter}{d.year() % 10}"


def date(imm_code: str, reference_date: Date | None = None) -> Date:
    """IMM date for ``imm_code`` on or after ``reference_date``.

    If ``reference_date`` is ``None`` the IMM date is found relative to
    today's date.
    """
    qassert.require(is_imm_code(imm_code, main_cycle=False), f"{imm_code} is not a valid IMM code")
    ref = reference_date if reference_date is not None else Date.todays_date()

    upper = imm_code.upper()
    month = _LETTER_TO_MONTH[upper[0]]
    y = int(upper[1])

    # year<1900 are not valid pquantlib years: add 10 years if 0 in 1900s decade.
    if y == 0 and ref.year() <= 1909:
        y += 10
    y += ref.year() - (ref.year() % 10)
    result = next_date(Date.from_ymd(1, month, y), main_cycle=False)
    if result < ref:
        return next_date(Date.from_ymd(1, month, y + 10), main_cycle=False)
    return result


def next_date(d: Date | None = None, main_cycle: bool = True) -> Date:
    """Next IMM date strictly after ``d``.

    With ``d=None``, uses today's date.
    """
    ref = d if d is not None else Date.todays_date()
    y = ref.year()
    m = int(ref.month())

    offset = 3 if main_cycle else 1
    skip_months = offset - (m % offset)
    if skip_months != offset or ref.day_of_month() > 21:
        skip_months += m
        if skip_months <= 12:
            m = skip_months
        else:
            m = skip_months - 12
            y += 1

    result = Date.nth_weekday(3, Weekday.Wednesday, Month(m), y)
    if result <= ref:
        result = next_date(Date.from_ymd(22, Month(m), y), main_cycle)
    return result


def next_code(d: Date | None = None, main_cycle: bool = True) -> str:
    """IMM code for the next IMM date strictly after ``d``."""
    return code(next_date(d, main_cycle))
