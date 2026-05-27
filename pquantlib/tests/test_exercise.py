"""Tests for the Exercise hierarchy (European/American/Bermudan)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import (
    AmericanExercise,
    BermudanExercise,
    EarlyExercise,
    EuropeanExercise,
    Exercise,
)
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _d(day: int, month: Month, year: int) -> Date:
    return Date.from_ymd(day, month, year)


# --- abstract guardrails ------------------------------------------------


def test_exercise_bare_construction_has_no_dates() -> None:
    """C++ ``Exercise(Type)`` is a public ctor that leaves ``dates_`` empty.

    Python preserves the same shape (no abstract methods declared); subclasses
    populate ``_dates``. ``last_date()`` on the bare instance raises.
    """
    e = Exercise(Exercise.Type.European)
    assert e.dates() == []
    with pytest.raises(LibraryException, match="no exercise date"):
        e.last_date()


def test_early_exercise_bare_construction_carries_payoff_at_expiry() -> None:
    """C++ ``EarlyExercise(Type, bool)`` is a public ctor."""
    ee = EarlyExercise(Exercise.Type.American, payoff_at_expiry=True)
    assert ee.payoff_at_expiry() is True
    assert ee.type() == Exercise.Type.American


# --- Exercise.Type ------------------------------------------------------


def test_exercise_type_integer_values_match_cpp() -> None:
    """C++ ql/exercise.hpp Exercise::Type: American=0, Bermudan=1, European=2."""
    assert int(Exercise.Type.American) == 0
    assert int(Exercise.Type.Bermudan) == 1
    assert int(Exercise.Type.European) == 2


# --- EuropeanExercise ---------------------------------------------------


def test_european_exercise_stores_single_date() -> None:
    d = _d(15, Month.June, 2027)
    e = EuropeanExercise(d)
    assert e.type() == Exercise.Type.European
    assert e.dates() == [d]
    assert e.last_date() == d
    assert e.date(0) == d


def test_european_exercise_is_not_early_exercise() -> None:
    e = EuropeanExercise(_d(15, Month.June, 2027))
    # European has no payoff_at_expiry / early-exercise concept.
    assert not isinstance(e, EarlyExercise)


# --- AmericanExercise ---------------------------------------------------


def test_american_exercise_two_date_constructor() -> None:
    earliest = _d(15, Month.June, 2026)
    latest = _d(15, Month.June, 2027)
    e = AmericanExercise(earliest, latest)
    assert e.type() == Exercise.Type.American
    assert e.dates() == [earliest, latest]
    assert e.earliest_date() == earliest
    assert e.latest_date() == latest
    assert e.last_date() == latest
    assert not e.payoff_at_expiry()


def test_american_exercise_single_date_constructor_uses_min_date_for_earliest() -> None:
    """C++ AmericanExercise(latestDate, ...) sets dates_[0] = Date::minDate()."""
    latest = _d(15, Month.June, 2027)
    e = AmericanExercise(latest)
    assert e.earliest_date() == Date.min_date()
    assert e.latest_date() == latest


def test_american_exercise_rejects_earliest_after_latest() -> None:
    earliest = _d(15, Month.June, 2027)
    latest = _d(15, Month.June, 2026)
    with pytest.raises(LibraryException, match="earliest"):
        AmericanExercise(earliest, latest)


def test_american_exercise_payoff_at_expiry_flag() -> None:
    e = AmericanExercise(
        _d(15, Month.June, 2026),
        _d(15, Month.June, 2027),
        payoff_at_expiry=True,
    )
    assert e.payoff_at_expiry() is True


# --- BermudanExercise ---------------------------------------------------


def test_bermudan_exercise_sorts_dates() -> None:
    """C++ BermudanExercise sorts the input dates."""
    d1 = _d(15, Month.January, 2027)
    d2 = _d(15, Month.June, 2027)
    d3 = _d(15, Month.December, 2026)
    e = BermudanExercise([d2, d1, d3])
    assert e.dates() == [d3, d1, d2]
    assert e.last_date() == d2


def test_bermudan_exercise_rejects_empty_dates() -> None:
    with pytest.raises(LibraryException, match="no exercise date"):
        BermudanExercise([])


def test_bermudan_exercise_payoff_at_expiry_flag() -> None:
    e = BermudanExercise(
        [_d(15, Month.June, 2026), _d(15, Month.December, 2026)],
        payoff_at_expiry=True,
    )
    assert e.payoff_at_expiry() is True


def test_bermudan_exercise_type_id() -> None:
    e = BermudanExercise([_d(15, Month.June, 2026)])
    assert e.type() == Exercise.Type.Bermudan


# --- Exercise.last_date error path -------------------------------------


def test_last_date_raises_when_no_dates() -> None:
    """Direct guarding: if a subclass somehow has empty _dates, lastDate() raises.

    Mirrors C++ ``Exercise::lastDate`` ``QL_REQUIRE(!dates_.empty(),...)``.
    """
    e = EuropeanExercise(_d(15, Month.June, 2026))
    e._dates = []  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(LibraryException, match="no exercise date"):
        e.last_date()
