"""Day-of-week enum.

# C++ parity: ql/time/weekday.hpp (v1.42.1) — enum Weekday.

C++ exposes both long names (Sunday..Saturday) and 3-letter aliases
(Sun..Sat) for the same integer values. Python ``IntEnum`` represents
aliases by assigning the same value; ``Weekday.Sun is Weekday.Sunday``
holds, both deserialize from int 1.
"""

from __future__ import annotations

from enum import IntEnum


class Weekday(IntEnum):
    Sunday = 1
    Monday = 2
    Tuesday = 3
    Wednesday = 4
    Thursday = 5
    Friday = 6
    Saturday = 7
    # 3-letter aliases (resolve to the canonical members above).
    Sun = 1
    Mon = 2
    Tue = 3
    Wed = 4
    Thu = 5
    Fri = 6
    Sat = 7
