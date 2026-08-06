"""Invert a day counter: find the date at a given year fraction.

# C++ parity: ql/time/daycounters/yearfractiontodate.{hpp,cpp} (v1.43).

``year_fraction_to_date(dc, reference_date, t)`` returns the date ``d`` for
which ``dc.year_fraction(reference_date, d)`` is closest to ``t``. Day
counters are not analytically invertible in general (Business252, ActualActual
with its period conventions), so C++ does a two-step guess at 365.25 days per
year followed by a coarse-to-fine search over years, then months, then days,
and finally picks whichever of the two bracketing dates is nearer.

The search is reproduced step for step, including the
``t += direction * 100 * QL_EPSILON`` nudge that keeps the loop from stalling
on a day counter whose year fraction is exactly ``t`` at the guess.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import QL_EPSILON
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _round_half_away(x: float) -> int:
    """C ``std::round``: ties go away from zero, unlike Python's ``round``."""
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def year_fraction_to_date(
    day_counter: DayCounter, reference_date: Date, t: float
) -> Date:
    """Date whose year fraction from ``reference_date`` is closest to ``t``."""
    guess_date = reference_date + Period(_round_half_away(t * 365.25), TimeUnit.Days)
    guess_time = day_counter.year_fraction(reference_date, guess_date)

    guess_date = guess_date + Period(
        _round_half_away((t - guess_time) * 365.25), TimeUnit.Days
    )
    guess_time = day_counter.year_fraction(reference_date, guess_date)

    if close_enough(guess_time, t):
        return guess_date

    search_direction = int(math.copysign(1.0, t - guess_time))

    t += search_direction * 100 * QL_EPSILON

    for unit in (TimeUnit.Years, TimeUnit.Months, TimeUnit.Days):
        while True:
            next_date = guess_date + Period(search_direction, unit)
            if (
                search_direction
                * (day_counter.year_fraction(reference_date, next_date) - t)
                >= 0.0
            ):
                break
            guess_date = next_date

    guess_time = day_counter.year_fraction(reference_date, guess_date)
    neighbour = guess_date + Period(search_direction, TimeUnit.Days)
    if close_enough(guess_time, t) or abs(
        day_counter.year_fraction(reference_date, neighbour) - t
    ) > abs(guess_time - t):
        return guess_date
    return neighbour


__all__ = ["year_fraction_to_date"]
