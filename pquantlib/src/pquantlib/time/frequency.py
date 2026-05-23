"""Frequency-of-events-per-year enum.

# C++ parity: ql/time/frequency.hpp (v1.42.1) — enum Frequency.

Values are integers-per-year where it makes sense (e.g. ``Monthly = 12``,
``Daily = 365``); ``NoFrequency = -1`` and ``Once = 0`` are sentinels;
``OtherFrequency = 999`` is the catch-all for non-canonical frequencies.
"""

from __future__ import annotations

from enum import IntEnum


class Frequency(IntEnum):
    NoFrequency = -1
    Once = 0
    Annual = 1
    Semiannual = 2
    EveryFourthMonth = 3
    Quarterly = 4
    Bimonthly = 6
    Monthly = 12
    EveryFourthWeek = 13
    Biweekly = 26
    Weekly = 52
    Daily = 365
    OtherFrequency = 999
