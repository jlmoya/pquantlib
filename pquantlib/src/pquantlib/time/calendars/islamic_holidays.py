"""Islamic holiday date tables — new in C++ QuantLib v1.43.

Faithful port of ``ql/time/calendars/islamicholidays.{hpp,cpp}`` from QuantLib
v1.43 @ ``6b57206e04598f092efee66e3b367efc84771995``.

C++ exposes these as free functions in the ``QuantLib::MoonSightingMethod``
namespace. Eid al-Fitr and Eid al-Adha are moon-sighting based and cannot be
computed arithmetically, so upstream tabulates them — here for 2026-2040,
exactly as C++ does.

The moon-sighting calendar is used across South Asia, Central Asia, the Middle
East and North Africa. (Saudi Arabia and some Gulf states instead use the Umm
al-Qura calendar, which C++ v1.43 does not tabulate here.)
"""

from __future__ import annotations

from pquantlib.time.date import Date
from pquantlib.time.month import Month

_EID_AL_FITR_YMD: tuple[tuple[int, Month, int], ...] = (
    (20, Month.March, 2026),
    (10, Month.March, 2027),
    (27, Month.February, 2028),
    (15, Month.February, 2029),
    (5, Month.February, 2030),
    (25, Month.January, 2031),
    (14, Month.January, 2032),
    (2, Month.January, 2033),
    (23, Month.December, 2033),
    (12, Month.December, 2034),
    (1, Month.December, 2035),
    (19, Month.November, 2036),
    (8, Month.November, 2037),
    (29, Month.October, 2038),
    (19, Month.October, 2039),
    (7, Month.October, 2040),
)

_EID_AL_ADHA_YMD: tuple[tuple[int, Month, int], ...] = (
    (27, Month.May, 2026),
    (17, Month.May, 2027),
    (5, Month.May, 2028),
    (24, Month.April, 2029),
    (13, Month.April, 2030),
    (3, Month.April, 2031),
    (22, Month.March, 2032),
    (11, Month.March, 2033),
    (28, Month.February, 2034),
    (18, Month.February, 2035),
    (7, Month.February, 2036),
    (27, Month.January, 2037),
    (16, Month.January, 2038),
    (5, Month.January, 2039),
    (26, Month.December, 2039),
    (15, Month.December, 2040),
)

_EID_AL_FITR: frozenset[Date] = frozenset(Date.from_ymd(d, m, y) for d, m, y in _EID_AL_FITR_YMD)
_EID_AL_ADHA: frozenset[Date] = frozenset(Date.from_ymd(d, m, y) for d, m, y in _EID_AL_ADHA_YMD)


def is_eid_al_fitr(d: Date) -> bool:
    """C++ parity: ``MoonSightingMethod::isEidAlFitr``."""
    return d in _EID_AL_FITR


def is_eid_al_adha(d: Date) -> bool:
    """C++ parity: ``MoonSightingMethod::isEidAlAdha``."""
    return d in _EID_AL_ADHA
