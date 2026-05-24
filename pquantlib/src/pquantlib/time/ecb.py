"""ECB (European Central Bank) reserve-maintenance date helpers.

# C++ parity: ql/time/ecb.hpp + ql/time/ecb.cpp (v1.42.1).

ECB maintenance periods do NOT follow a closed-form rule like IMM/ASX —
they are published by the ECB year-by-year. The C++ source embeds the
table verbatim (200 entries spanning 2005..2024); the Python port does
the same, in ``_KNOWN_DATE_SERIALS`` below.

Module-level state ``_known_dates`` is a mutable ``set[Date]`` so callers
can extend / remove dates at runtime, mirroring the C++ static-set
``addDate`` / ``removeDate`` API. Codes are ``MMM<YY>`` (e.g. ``"MAR10"``).
"""

from __future__ import annotations

import bisect
from typing import Final

from pquantlib import qassert
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# Month name <-> Month mapping (case-insensitive at parse time).
_NAME_TO_MONTH: Final[dict[str, Month]] = {
    "JAN": Month.January,
    "FEB": Month.February,
    "MAR": Month.March,
    "APR": Month.April,
    "MAY": Month.May,
    "JUN": Month.June,
    "JUL": Month.July,
    "AUG": Month.August,
    "SEP": Month.September,
    "OCT": Month.October,
    "NOV": Month.November,
    "DEC": Month.December,
}
_MONTH_TO_NAME: Final[dict[Month, str]] = {m: nm for nm, m in _NAME_TO_MONTH.items()}

# Published ECB maintenance period start dates (2005..2024).
# Serial numbers extracted verbatim from ql/time/ecb.cpp ``ecbKnownDateSet``.
_KNOWN_DATE_SERIALS: Final[tuple[int, ...]] = (
    38371,
    38391,
    38420,
    38455,
    38483,
    38511,
    38546,
    38574,
    38602,
    38637,
    38665,
    38692,
    38735,
    38756,
    38784,
    38819,
    38847,
    38883,
    38910,
    38938,
    38966,
    39001,
    39029,
    39064,
    39099,
    39127,
    39155,
    39190,
    39217,
    39246,
    39274,
    39302,
    39337,
    39365,
    39400,
    39428,
    39463,
    39491,
    39519,
    39554,
    39582,
    39610,
    39638,
    39673,
    39701,
    39729,
    39764,
    39792,
    39834,
    39855,
    39883,
    39911,
    39946,
    39974,
    40002,
    40037,
    40065,
    40100,
    40128,
    40155,
    40198,
    40219,
    40247,
    40282,
    40310,
    40345,
    40373,
    40401,
    40429,
    40464,
    40492,
    40520,
    40562,
    40583,
    40611,
    40646,
    40674,
    40709,
    40737,
    40765,
    40800,
    40828,
    40856,
    40891,
    40926,
    40954,
    40982,
    41010,
    41038,
    41073,
    41101,
    41129,
    41164,
    41192,
    41227,
    41255,
    41290,
    41318,
    41346,
    41374,
    41402,
    41437,
    41465,
    41493,
    41528,
    41556,
    41591,
    41619,
    41654,
    41682,
    41710,
    41738,
    41773,
    41801,
    41829,
    41864,
    41892,
    41920,
    41955,
    41983,
    42032,
    42074,
    42116,
    42165,
    42207,
    42256,
    42305,
    42347,
    42396,
    42445,
    42487,
    42529,
    42578,
    42627,
    42669,
    42718,
    42760,
    42809,
    42858,
    42900,
    42942,
    42991,
    43040,
    43089,
    43131,
    43167,
    43216,
    43265,
    43307,
    43356,
    43398,
    43447,
    43495,
    43537,
    43572,
    43628,
    43677,
    43726,
    43768,
    43817,
    43859,
    43908,
    43957,
    43992,
    44034,
    44090,
    44139,
    44181,
    44223,
    44272,
    44314,
    44363,
    44405,
    44454,
    44503,
    44552,
    44601,
    44636,
    44671,
    44727,
    44769,
    44818,
    44867,
    44916,
    44965,
    45007,
    45056,
    45098,
    45140,
    45189,
    45231,
    45280,
    45322,
    45364,
    45399,
    45455,
    45497,
    45553,
    45588,
    45644,
)

_known_dates: set[Date] = {Date(s) for s in _KNOWN_DATE_SERIALS}


def known_dates() -> frozenset[Date]:
    """Snapshot of the current ECB known-date set (mutable via add_date / remove_date)."""
    return frozenset(_known_dates)


def add_date(d: Date) -> None:
    _known_dates.add(d)


def remove_date(d: Date) -> None:
    _known_dates.discard(d)


def is_ecb_code(ecb_code: str) -> bool:
    if len(ecb_code) != 5:
        return False
    if ecb_code[:3].upper() not in _NAME_TO_MONTH:
        return False
    return ecb_code[3].isdigit() and ecb_code[4].isdigit()


def date(ecb_code_or_month: str | Month, year_or_ref: int | Date | None = None) -> Date:
    """Two overloads matching C++:

    - ``date(code, reference_date=None)``: parse ``MMMYY`` and return the
      ECB date in that month/year (relative to ``reference_date``'s century).
    - ``date(Month, year)``: return the ECB date in (year, month).
    """
    # Overload A: code + optional reference date
    if isinstance(ecb_code_or_month, str):
        return _date_from_code(ecb_code_or_month, year_or_ref if isinstance(year_or_ref, Date) else None)
    # Overload B: (Month, int)
    qassert.require(isinstance(year_or_ref, int), "date(Month, year): year must be int")
    assert isinstance(year_or_ref, int)
    return next_date(Date.from_ymd(1, ecb_code_or_month, year_or_ref) - 1)


def _date_from_code(ecb_code: str, reference_date: Date | None) -> Date:
    qassert.require(is_ecb_code(ecb_code), f"{ecb_code} is not a valid ECB code")
    month = _NAME_TO_MONTH[ecb_code[:3].upper()]
    y = int(ecb_code[3]) * 10 + int(ecb_code[4])
    ref = reference_date if reference_date is not None else Date.todays_date()
    reference_year_lo = ref.year() % 100
    y += ref.year() - reference_year_lo
    if y < Date.min_date().year():
        return next_date(Date.min_date())
    return next_date(Date.from_ymd(1, month, y) - 1)


def code(ecb_date: Date) -> str:
    qassert.require(ecb_date in _known_dates, f"{ecb_date} is not a valid ECB date")
    return f"{_MONTH_TO_NAME[ecb_date.month()]}{ecb_date.year() % 100:02d}"


def next_date(d: Date | None = None) -> Date:
    """Smallest known ECB date strictly after ``d``."""
    ref = d if d is not None else Date.todays_date()
    sorted_known = sorted(_known_dates)
    idx = bisect.bisect_right(sorted_known, ref)
    qassert.require(idx < len(sorted_known), f"ECB dates after {sorted_known[-1]} are unknown")
    return sorted_known[idx]


def next_dates(d: Date | None = None) -> tuple[Date, ...]:
    """All known ECB dates strictly after ``d`` (in order)."""
    ref = d if d is not None else Date.todays_date()
    sorted_known = sorted(_known_dates)
    idx = bisect.bisect_right(sorted_known, ref)
    qassert.require(idx < len(sorted_known), f"ECB dates after {sorted_known[-1]} are unknown")
    return tuple(sorted_known[idx:])


def next_code(ecb_code: str) -> str:
    """Mirrors C++ ``ECB::nextCode(const std::string&)`` — bump month then year on December overflow."""
    qassert.require(is_ecb_code(ecb_code), f"{ecb_code} is not a valid ECB code")
    month = _NAME_TO_MONTH[ecb_code[:3].upper()]
    yy = ecb_code[3:5]
    if month != Month.December:
        next_month = Month(int(month) + 1)
        return f"{_MONTH_TO_NAME[next_month]}{yy}"
    # December → JAN of next year. Increment the 2-digit year string with overflow.
    digits = list(yy)

    def inc_with_overflow(digit_idx: int) -> bool:
        if digits[digit_idx] == "9":
            digits[digit_idx] = "0"
            return True
        digits[digit_idx] = chr(ord(digits[digit_idx]) + 1)
        return False

    if inc_with_overflow(1):
        inc_with_overflow(0)
    return f"JAN{''.join(digits)}"
