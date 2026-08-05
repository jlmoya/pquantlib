"""year_fraction_to_date — the "inverse" of a day counter.

# C++ parity: ql/time/daycounters/yearfractiontodate.{hpp,cpp} (v1.43).

Given a day counter, a reference date and a year fraction ``t``, returns the
date ``d`` for which ``day_counter.year_fraction(reference_date, d)`` is
closest to ``t``.  Two rounded 365.25-day guesses get within a few days, then a
Years/Months/Days ladder walks to the bracketing date and the nearer of the two
candidates wins (yearfractiontodate.cpp:29-63).
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import QL_EPSILON
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_DAYS_PER_YEAR_GUESS = 365.25


def _round_half_away_from_zero(x: float) -> int:
    """C++ ``std::round`` — ties go away from zero, unlike Python's ``round``."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def year_fraction_to_date(day_counter: DayCounter, reference_date: Date, t: float) -> Date:
    """# C++ parity: ``yearFractionToDate`` (yearfractiontodate.cpp:29-63)."""
    guess_date = reference_date + Period(_round_half_away_from_zero(t * _DAYS_PER_YEAR_GUESS), TimeUnit.Days)
    guess_time = day_counter.year_fraction(reference_date, guess_date)

    guess_date = guess_date + Period(
        _round_half_away_from_zero((t - guess_time) * _DAYS_PER_YEAR_GUESS), TimeUnit.Days
    )
    guess_time = day_counter.year_fraction(reference_date, guess_date)

    if close_enough(guess_time, t):
        return guess_date

    search_direction = int(math.copysign(1.0, t - guess_time))

    t += search_direction * 100 * QL_EPSILON

    for u in (TimeUnit.Years, TimeUnit.Months, TimeUnit.Days):
        while True:
            next_date = guess_date + Period(search_direction, u)
            if search_direction * (day_counter.year_fraction(reference_date, next_date) - t) < 0.0:
                guess_date = next_date
            else:
                break

    guess_time = day_counter.year_fraction(reference_date, guess_date)
    neighbour = guess_date + Period(search_direction, TimeUnit.Days)
    if close_enough(guess_time, t) or abs(day_counter.year_fraction(reference_date, neighbour) - t) > abs(
        guess_time - t
    ):
        return guess_date
    return neighbour


__all__ = ["year_fraction_to_date"]
