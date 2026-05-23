"""Date-generation rule enum for Schedule construction.

# C++ parity: ql/time/dategenerationrule.hpp (v1.42.1) —
# struct DateGeneration { enum Rule { ... } };

The C++ wraps the enum in a ``DateGeneration`` namespace-class so values
read as ``DateGeneration::Backward``. The Python port flattens that:
``DateGeneration.Backward`` is an ``IntEnum`` member directly.
"""

from __future__ import annotations

from enum import IntEnum


class DateGeneration(IntEnum):
    Backward = 0
    Forward = 1
    Zero = 2
    ThirdWednesday = 3
    ThirdWednesdayInclusive = 4
    Twentieth = 5
    TwentiethIMM = 6
    OldCDS = 7
    CDS = 8
    CDS2015 = 9
