"""Business-day-adjustment convention enum.

# C++ parity: ql/time/businessdayconvention.hpp (v1.42.1) —
# enum BusinessDayConvention.

Values are sequential starting at 0. The first three (Following,
ModifiedFollowing, Preceding) are ISDA-standard; the remaining four
are non-ISDA conventions used in specific markets.
"""

from __future__ import annotations

from enum import IntEnum


class BusinessDayConvention(IntEnum):
    Following = 0
    ModifiedFollowing = 1
    Preceding = 2
    ModifiedPreceding = 3
    Unadjusted = 4
    HalfMonthModifiedFollowing = 5
    Nearest = 6
