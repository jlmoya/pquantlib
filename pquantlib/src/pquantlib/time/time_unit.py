"""Time-unit enum for Period arithmetic.

# C++ parity: ql/time/timeunit.hpp (v1.42.1) — enum TimeUnit.

Values are sequential starting at 0.
"""

from __future__ import annotations

from enum import IntEnum


class TimeUnit(IntEnum):
    Days = 0
    Weeks = 1
    Months = 2
    Years = 3
    Hours = 4
    Minutes = 5
    Seconds = 6
    Milliseconds = 7
    Microseconds = 8
