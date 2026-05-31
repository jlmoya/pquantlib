"""SwingExercise — bermudan exercise with optional intraday-second offsets.

# C++ parity: ql/instruments/vanillaswingoption.{hpp,cpp} (v1.42.1) —
# ``class SwingExercise : public BermudanExercise``.

A swing exercise carries one ``seconds`` integer per date (0 by
default) so the consumer can express sub-day exercise resolution
(e.g. hourly power-market exercises spread across each day). The
exercise-time helper ``exercise_times(dc, ref_date)`` interpolates
between the date's start-of-day and start-of-next-day using the
seconds offset.

Two constructor forms (matching C++):

* ``SwingExercise(dates, seconds=None)`` — direct list of dates with
  optional matching seconds vector (defaults to ``[0]*len(dates)``).
* ``SwingExercise.from_range(from_date, to_date, step_size_secs)`` —
  builds dates + seconds by walking from ``from_date`` to ``to_date``
  with a fixed second step that wraps to the next day at 86_400s.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import BermudanExercise
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_SEC_PER_DAY: int = 24 * 3600


class SwingExercise(BermudanExercise):
    """Bermudan exercise + per-date intraday-second offsets.

    # C++ parity: ``class SwingExercise : public BermudanExercise``.
    """

    def __init__(
        self,
        dates: Sequence[Date],
        seconds: Sequence[int] | None = None,
    ) -> None:
        # C++ parity: ``SwingExercise::SwingExercise(const std::vector<Date>&,
        # const std::vector<Size>&)``. ``dates`` empty case falls through to
        # BermudanExercise which raises ``no exercise date given``.
        super().__init__(dates)
        if seconds is None or len(tuple(seconds)) == 0:
            sec_list: list[int] = [0] * len(self._dates)
        else:
            sec_list = [int(s) for s in seconds]
        qassert.require(
            len(self._dates) == len(sec_list),
            "dates and seconds must have the same size",
        )
        for i, sec in enumerate(sec_list):
            qassert.require(
                sec < _SEC_PER_DAY,
                "a date can not have more than 24*3600 seconds",
            )
            if i > 0:
                # C++ parity: ordering check (dates[i-1] < dates[i]) or
                # (dates[i-1] == dates[i] and seconds[i-1] < seconds[i]).
                prev = self._dates[i - 1]
                curr = self._dates[i]
                qassert.require(
                    prev < curr or (prev == curr and sec_list[i - 1] < sec),
                    "date times must be sorted",
                )
        self._seconds: list[int] = sec_list

    @classmethod
    def from_range(
        cls,
        from_date: Date,
        to_date: Date,
        step_size_secs: int,
    ) -> SwingExercise:
        """Build a swing exercise over a date range with fixed seconds step.

        # C++ parity: ``SwingExercise::SwingExercise(const Date& from,
        # const Date& to, Size stepSizeSecs)``.

        Walks from ``from_date`` to ``to_date`` advancing the seconds
        counter by ``step_size_secs`` each step; wraps to the next
        day at the 86_400-second mark.
        """
        qassert.require(step_size_secs > 0, "step_size_secs must be positive")
        dates: list[Date] = []
        secs: list[int] = []
        iter_date: Date = from_date
        iter_step: int = 0
        # C++ parity: ``while (iterDate <= to)`` loop in createDateTimes().
        while iter_date <= to_date:
            dates.append(iter_date)
            secs.append(iter_step)
            iter_step += step_size_secs
            if iter_step >= _SEC_PER_DAY:
                iter_date = iter_date + Period(1, TimeUnit.Days)
                iter_step %= _SEC_PER_DAY
        return cls(dates, secs)

    def seconds(self) -> list[int]:
        """Per-date intraday-second offsets (length = ``len(dates())``)."""
        return list(self._seconds)

    def exercise_times(
        self, day_counter: DayCounter, ref_date: Date
    ) -> list[float]:
        """Year-fractions of the exercise instants from ``ref_date``.

        # C++ parity: ``SwingExercise::exerciseTimes(const DayCounter&,
        # const Date&)``.

        For each exercise date ``D_i`` with seconds offset ``s_i``:
        ``t_i = dc.year_fraction(ref, D_i) + dt_i * s_i / 86_400``,
        where ``dt_i`` is the year-fraction of a single day starting
        at ``D_i``.
        """
        out: list[float] = []
        for d, sec in zip(self._dates, self._seconds, strict=True):
            t = day_counter.year_fraction(ref_date, d)
            d_next = d + Period(1, TimeUnit.Days)
            dt = day_counter.year_fraction(ref_date, d_next) - t
            t += dt * sec / _SEC_PER_DAY
            qassert.require(t >= 0, "exercise dates must not contain past date")
            out.append(t)
        return out


__all__ = ["SwingExercise"]
