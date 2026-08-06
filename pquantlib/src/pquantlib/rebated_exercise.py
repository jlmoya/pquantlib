"""RebatedExercise — an exercise that pays (or charges) a rebate on exercise.

# C++ parity: ql/rebatedexercise.{hpp,cpp} @ v1.43.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit


class RebatedExercise(Exercise):
    """Exercise carrying a per-date rebate.

    # C++ parity: ``class RebatedExercise : public Exercise``
    # (ql/rebatedexercise.hpp:38, ql/rebatedexercise.cpp:24-52).
    #
    # C++ has two constructors; the scalar one broadcasts a single rebate over
    # every exercise date, the vector one is Bermudan-only and length-checked.
    # Python folds them into one signature discriminated on the argument type —
    # the C++ validation of each overload is reproduced exactly.

    On exercise the holder RECEIVES the rebate when positive and PAYS it when
    negative, on the rebate settlement date.
    """

    def __init__(
        self,
        exercise: Exercise,
        rebate: float | Sequence[float] = 0.0,
        rebate_settlement_days: int = 0,
        rebate_payment_calendar: Calendar | None = None,
        rebate_payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
    ) -> None:
        # C++ copy-constructs the Exercise base: `Exercise(exercise)`.
        super().__init__(exercise.type())
        self._dates = list(exercise.dates())

        if isinstance(rebate, (int, float)):
            # C++ ctor #1 (rebatedexercise.cpp:26-33): broadcast over dates().
            self._rebates: list[float] = [float(rebate)] * len(self._dates)
        else:
            # C++ ctor #2 (rebatedexercise.cpp:35-52).
            qassert.require(
                self.type() == Exercise.Type.Bermudan,
                "a rebate vector is allowed only for a bermudan style exercise",
            )
            qassert.require(
                len(rebate) == len(self._dates),
                f"the number of rebates ({len(rebate)}) must be equal to "
                f"the number of exercise dates ({len(self._dates)})",
            )
            self._rebates = [float(r) for r in rebate]

        self._rebate_settlement_days: int = rebate_settlement_days
        self._rebate_payment_calendar: Calendar = (
            NullCalendar() if rebate_payment_calendar is None else rebate_payment_calendar
        )
        self._rebate_payment_convention: BusinessDayConvention = rebate_payment_convention

    def rebate(self, index: int) -> float:
        """Rebate for exercise date ``index``.

        # C++ parity: ``RebatedExercise::rebate`` (ql/rebatedexercise.hpp:66-71).
        """
        qassert.require(
            0 <= index < len(self._rebates),
            f"rebate with index {index} does not exist (0...{len(self._rebates) - 1})",
        )
        return self._rebates[index]

    def rebates(self) -> list[float]:
        """# C++ parity: ``RebatedExercise::rebates()`` (ql/rebatedexercise.hpp:56)."""
        return self._rebates

    def rebate_payment_date(self, index: int) -> Date:
        """Settlement date of the rebate for exercise date ``index``.

        # C++ parity: ``RebatedExercise::rebatePaymentDate``
        # (ql/rebatedexercise.hpp:73-80). American style is rejected: C++ leaves
        # that computation to client code.
        """
        qassert.require(
            self.type() in (Exercise.Type.European, Exercise.Type.Bermudan),
            "for american style exercises the rebate payment date "
            "has to be calculted in the client code",
        )
        return self._rebate_payment_calendar.advance(
            self._dates[index],
            self._rebate_settlement_days,
            TimeUnit.Days,
            self._rebate_payment_convention,
        )

    def rebate_settlement_days(self) -> int:
        """Settlement lag in business days (C++ ``rebateSettlementDays_``)."""
        return self._rebate_settlement_days

    def rebate_payment_calendar(self) -> Calendar:
        """Calendar used to roll the rebate payment date (C++ ``rebatePaymentCalendar_``)."""
        return self._rebate_payment_calendar

    def rebate_payment_convention(self) -> BusinessDayConvention:
        """Roll convention for the rebate payment date (C++ ``rebatePaymentConvention_``)."""
        return self._rebate_payment_convention


__all__ = ["RebatedExercise"]
