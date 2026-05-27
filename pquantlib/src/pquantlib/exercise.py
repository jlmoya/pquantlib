"""Option exercise classes.

# C++ parity: ql/exercise.{hpp,cpp} (v1.42.1).

Three concrete exercise variants:

* ``EuropeanExercise(date)`` — exercisable only at ``date``.
* ``AmericanExercise(earliest, latest, payoff_at_expiry=False)`` —
  exercisable between two dates (``earliest`` may be omitted by passing
  ``Date.min_date()`` semantics — but C++ also exposes a one-date
  constructor that uses ``Date::minDate()`` as the earliest, which we
  mirror via the ``latest`` parameter alias).
* ``BermudanExercise(dates, payoff_at_expiry=False)`` — exercisable on
  the supplied date list; the C++ constructor sorts the dates, and we
  do the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.date import Date


class Exercise:
    """Abstract-by-convention exercise base.

    # C++ parity: ql/exercise.hpp ``class Exercise`` — C++ also exposes
    # a public ctor that leaves ``dates_`` empty; concrete subclasses
    # populate it. The Python port preserves this shape (no
    # ``@abstractmethod``) — direct instantiation is allowed but only
    # useful for testing or as a generic carrier.

    Subclasses populate ``_dates`` (1 element for European, 2 for
    American = ``[earliest, latest]``, >=1 sorted for Bermudan). The
    base does not enforce any specific cardinality.
    """

    class Type(IntEnum):
        """Exercise style.

        # C++ parity: ql/exercise.hpp ``Exercise::Type``: American=0,
        # Bermudan=1, European=2 (declaration order). Python preserves
        # the integer values for round-trips.
        """

        American = 0
        Bermudan = 1
        European = 2

    def __init__(self, exercise_type: Exercise.Type) -> None:
        self._type: Exercise.Type = exercise_type
        self._dates: list[Date] = []

    def type(self) -> Exercise.Type:
        return self._type

    def dates(self) -> list[Date]:
        return self._dates

    def date(self, index: int) -> Date:
        """C++ ``Exercise::date(Size index)`` — unchecked index access."""
        return self._dates[index]

    def last_date(self) -> Date:
        """C++ ``Exercise::lastDate()`` — last exercise date."""
        qassert.require(len(self._dates) > 0, "no exercise date given")
        return self._dates[-1]


class EarlyExercise(Exercise):
    """Early-exercise base.

    # C++ parity: ql/exercise.hpp ``class EarlyExercise``.

    Carries the ``payoff_at_expiry`` flag for American / Bermudan
    products that pay the residual at expiry instead of immediately on
    exercise. Like ``Exercise``, this base is abstract-by-convention —
    not enforced by ``@abstractmethod``.
    """

    def __init__(self, exercise_type: Exercise.Type, payoff_at_expiry: bool = False) -> None:
        super().__init__(exercise_type)
        self._payoff_at_expiry: bool = payoff_at_expiry

    def payoff_at_expiry(self) -> bool:
        return self._payoff_at_expiry


class EuropeanExercise(Exercise):
    """European: single exercise date.

    # C++ parity: ql/exercise.{hpp,cpp} ``class EuropeanExercise``.
    """

    def __init__(self, date: Date) -> None:
        super().__init__(Exercise.Type.European)
        self._dates = [date]


class AmericanExercise(EarlyExercise):
    """American: exercisable anywhere in [earliest, latest].

    # C++ parity: ql/exercise.{hpp,cpp} ``class AmericanExercise``.

    Two constructor forms (C++ has both):

    * ``AmericanExercise(earliest, latest, payoff_at_expiry=False)``.
    * ``AmericanExercise(latest, payoff_at_expiry=False)`` (with
      ``earliest = Date.min_date()``).

    Python collapses these into one signature using ``Date | None``
    for ``latest`` — if only one positional date is given, it's
    interpreted as ``latest`` and ``earliest`` defaults to
    ``Date.min_date()``.
    """

    def __init__(
        self,
        earliest_date: Date,
        latest_date: Date | None = None,
        payoff_at_expiry: bool = False,
    ) -> None:
        super().__init__(Exercise.Type.American, payoff_at_expiry)
        if latest_date is None:
            # Single-date form: earliest_date is actually the latest.
            earliest = Date.min_date()
            latest = earliest_date
        else:
            earliest = earliest_date
            latest = latest_date
        qassert.require(earliest <= latest, f"earliest ({earliest}) > latest ({latest}) exercise date")
        self._dates = [earliest, latest]

    def earliest_date(self) -> Date:
        return self._dates[0]

    def latest_date(self) -> Date:
        # C++: in EarlyExercise: not overridden; the protected dates_[1]
        # is exposed via Exercise::lastDate(). Provide a named accessor
        # for clarity.
        return self._dates[1]


class BermudanExercise(EarlyExercise):
    """Bermudan: exercisable on any of a fixed list of dates.

    # C++ parity: ql/exercise.{hpp,cpp} ``class BermudanExercise``.
    Sorts the dates on construction (mirrors C++).
    """

    def __init__(self, dates: Sequence[Date], payoff_at_expiry: bool = False) -> None:
        super().__init__(Exercise.Type.Bermudan, payoff_at_expiry)
        qassert.require(len(dates) > 0, "no exercise date given")
        self._dates = sorted(dates)


__all__ = [
    "AmericanExercise",
    "BermudanExercise",
    "EarlyExercise",
    "EuropeanExercise",
    "Exercise",
]
