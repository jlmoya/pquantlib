"""Month-of-year enum.

# C++ parity: ql/time/date.hpp (v1.42.1) — enum Month.

C++ exposes both long names (January..December) and 3-letter aliases
(Jan..Dec, except May which has no abbreviation in the C++ enum because
it is already three letters). Python ``IntEnum`` represents aliases by
assigning the same value.
"""

from __future__ import annotations

from enum import IntEnum


class Month(IntEnum):
    January = 1
    February = 2
    March = 3
    April = 4
    May = 5
    June = 6
    July = 7
    August = 8
    September = 9
    October = 10
    November = 11
    December = 12
    # 3-letter aliases (no separate "May" abbreviation in C++ — it's already 3 letters).
    Jan = 1
    Feb = 2
    Mar = 3
    Apr = 4
    Jun = 6
    Jul = 7
    Aug = 8
    Sep = 9
    Oct = 10
    Nov = 11
    Dec = 12
