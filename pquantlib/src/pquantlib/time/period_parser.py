"""Period string parsing.

# C++ parity: ql/utilities/dataparsers.hpp ``class PeriodParser`` (v1.42.1).

The C++ static class is flattened to a module of free functions. Mirrors
the C++ algorithm exactly:

1. Split the input on each occurrence of ``DdWwMmYy`` (case-insensitive).
2. Each segment is ``<signed_int><unit_letter>``.
3. Sum the segments via ``Period`` addition — composite tokens like ``1Y6M``
   work because ``Period(1, Years) + Period(6, Months) == Period(18, Months)``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_UNIT_LETTERS: frozenset[str] = frozenset("DdWwMmYy")


def _parse_one_period(token: str) -> Period:
    qassert.require(
        len(token) > 1,
        "single period require a string of at least 2 characters",
    )
    # Last char is the unit letter; everything before is the (signed) number.
    last = token[-1]
    qassert.require(last in _UNIT_LETTERS, f"unknown '{last}' unit")
    unit_char = last.upper()
    if unit_char == "D":
        units = TimeUnit.Days
    elif unit_char == "W":
        units = TimeUnit.Weeks
    elif unit_char == "M":
        units = TimeUnit.Months
    else:  # 'Y'
        units = TimeUnit.Years

    number_part = token[:-1]
    qassert.require(
        len(number_part) > 0 and any(c in "0123456789-+" for c in number_part),
        f"no numbers of {units.name} provided",
    )
    try:
        n = int(number_part)
    except ValueError as exc:
        qassert.fail(
            f"unable to parse the number of units of {units.name} in '{token}'. Error: {exc}",
        )
    return Period(n, units)


def parse(s: str) -> Period:
    """Parse a period string like ``"3M"``, ``"1Y6M"``, ``"-2W"``, ``"+5D"``."""
    qassert.require(len(s) > 1, "period string length must be at least 2")

    # Split into segments, each ending at a unit letter.
    segments: list[str] = []
    start = 0
    for i, ch in enumerate(s):
        if ch in _UNIT_LETTERS:
            segments.append(s[start : i + 1])
            start = i + 1
    qassert.require(start == len(s), f"unknown '{s}' unit")

    result = _parse_one_period(segments[0])
    for seg in segments[1:]:
        result = result + _parse_one_period(seg)
    return result
