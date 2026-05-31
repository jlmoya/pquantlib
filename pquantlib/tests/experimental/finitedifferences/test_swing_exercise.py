"""Tests for :class:`SwingExercise`.

# C++ parity reference: ql/instruments/vanillaswingoption.{hpp,cpp}
# v1.42.1 + test-suite/vpp.cpp / swingoption.cpp setup conventions.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.experimental.finitedifferences.swing_exercise import SwingExercise
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _d(day: int, month: Month, year: int) -> Date:
    return Date.from_ymd(day, month, year)


def test_dates_only_ctor_defaults_seconds_to_zero() -> None:
    """The ``dates``-only constructor pads seconds with zeros."""
    today = _d(18, Month.December, 2011)
    dates = [today, today + 1, today + 2]
    ex = SwingExercise(dates)
    assert ex.dates() == dates
    assert ex.seconds() == [0, 0, 0]
    assert ex.type() == Exercise.Type.Bermudan


def test_dates_and_seconds_ctor_roundtrip() -> None:
    """Explicit seconds are preserved 1:1."""
    today = _d(18, Month.December, 2011)
    dates = [today, today, today + 1]
    secs = [0, 3600, 0]
    ex = SwingExercise(dates, secs)
    assert ex.dates() == dates
    assert ex.seconds() == secs


def test_seconds_must_be_strictly_less_than_86400() -> None:
    today = _d(18, Month.December, 2011)
    with pytest.raises(LibraryException):
        SwingExercise([today], [86400])


def test_size_mismatch_raises() -> None:
    today = _d(18, Month.December, 2011)
    with pytest.raises(LibraryException):
        SwingExercise([today, today + 1], [0])


def test_seconds_must_be_sorted_when_dates_equal() -> None:
    """C++ check: when consecutive dates are equal, seconds must be strictly
    increasing. Out-of-order dates alone don't trigger the check since
    :class:`BermudanExercise` sorts them on construction — matching C++.
    """
    today = _d(18, Month.December, 2011)
    # Same date, descending seconds.
    with pytest.raises(LibraryException):
        SwingExercise([today, today], [10, 5])


def test_from_range_canonical_vpp_setup() -> None:
    """``SwingExercise.from_range(today, today+6, 3600)`` produces 168 hourly
    instants — the canonical VPP test-suite setup (testVPPIntrinsicValue).
    """
    today = _d(18, Month.December, 2011)
    ex = SwingExercise.from_range(today, today + 6, 3600)
    # 7 days * 24h = 168 hourly exercises.
    assert len(ex.dates()) == 168
    # Seconds wrap at the day boundary.
    assert ex.seconds()[0] == 0
    assert ex.seconds()[1] == 3600
    assert ex.seconds()[23] == 23 * 3600
    assert ex.seconds()[24] == 0  # rolled to the next day
    # The 24th date should be ``today + 1``.
    assert ex.dates()[24] == today + 1


def test_exercise_times_monotone_for_canonical_range() -> None:
    today = _d(18, Month.December, 2011)
    # `from_range(today, today+1, 3600)` walks today + (today+1) since
    # ``iter_date <= to`` is inclusive: 24 hours/day * 2 days = 48 instants.
    ex = SwingExercise.from_range(today, today + 1, 3600)
    dc = ActualActual(Convention.ISDA)
    times = ex.exercise_times(dc, today)
    assert len(times) == 48
    # Monotonically increasing.
    for i in range(1, len(times)):
        assert times[i] > times[i - 1]
    # First instant is exactly t=0 (today midnight reference).
    tight(times[0], 0.0)


def test_from_range_rejects_nonpositive_step() -> None:
    today = _d(18, Month.December, 2011)
    with pytest.raises(LibraryException):
        SwingExercise.from_range(today, today + 1, 0)
