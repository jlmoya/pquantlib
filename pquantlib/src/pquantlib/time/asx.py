"""ASX (Australian Securities Exchange) date helpers.

# C++ parity: ql/time/asx.hpp + ql/time/asx.cpp (v1.42.1).

C++ exposes these as static methods on a ``struct ASX``. This module
carries them as module-level free functions (the implementation) and also
as the :class:`ASX` namespace-only class, which mirrors the C++ API
name-for-name.

Structurally identical to IMM, but with:
- Day-of-week: Friday (vs Wednesday).
- Day-of-month range: [8, 14] (vs [15, 21]).
- Anchor: 2nd Friday of month (vs 3rd Wednesday).
- Month-letter codes: same alphabet (F..Z) since both follow futures-market
  convention; main cycle is HMUZ (Mar/Jun/Sep/Dec) for both.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NoReturn

from pquantlib import qassert
from pquantlib.patterns.observable_settings import ObservableSettings
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
    # C++ parity: ASX::date resolves an unset reference date to
    # Settings::instance().evaluationDate() (asx.cpp:95-97), not the wall clock.
    ref = reference_date if reference_date is not None else ObservableSettings().evaluation_date_or_today()

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


def next_date(
    d: Date | str | None = None,
    main_cycle: bool = True,
    reference_date: Date | None = None,
) -> Date:
    """Next ASX date strictly after ``d``.

    Two C++ overloads collapse into one signature:

    - ``next_date(Date, main_cycle)`` — ``ASX::nextDate(const Date&, bool)``
      (asx.cpp:118). With ``d=None``, uses today's date.
    - ``next_date(str, main_cycle, reference_date)`` —
      ``ASX::nextDate(const std::string&, bool, const Date&)``
      (asx.cpp:144-149), i.e. ``nextDate(date(asxCode, referenceDate) + 1)``.
    """
    if isinstance(d, str):
        return next_date(date(d, reference_date) + 1, main_cycle)
    ref = d if d is not None else ObservableSettings().evaluation_date_or_today()
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


def next_code(
    d: Date | str | None = None,
    main_cycle: bool = True,
    reference_date: Date | None = None,
) -> str:
    """ASX code for the next ASX date strictly after ``d``.

    Mirrors both C++ overloads: ``ASX::nextCode(const Date&, bool)``
    (asx.cpp:151-155) and
    ``ASX::nextCode(const std::string&, bool, const Date&)``
    (asx.cpp:157-162).
    """
    return code(next_date(d, main_cycle, reference_date))


class ASX:
    """Namespace-only class — direct construction is disabled.

    C++ parity: ``ql/time/asx.hpp:36`` ``struct ASX``. See
    :class:`pquantlib.time.imm.IMM` for the rationale behind mirroring the
    C++ static-member struct as a namespace-only Python class.
    """

    class Month(IntEnum):
        """C++ parity: ``ql/time/asx.hpp:37-40`` ``ASX::Month``.

        Futures-market month letters. Distinct from
        :class:`pquantlib.time.month.Month`, which is the calendar month.
        """

        F = 1
        G = 2
        H = 3
        J = 4
        K = 5
        M = 6
        N = 7
        Q = 8
        U = 9
        V = 10
        X = 11
        Z = 12

    def __init__(self) -> NoReturn:
        msg = "ASX is a namespace; use staticmethods only"
        raise TypeError(msg)

    is_asx_date = staticmethod(is_asx_date)
    is_asx_code = staticmethod(is_asx_code)
    code = staticmethod(code)
    date = staticmethod(date)
    next_date = staticmethod(next_date)
    next_code = staticmethod(next_code)
