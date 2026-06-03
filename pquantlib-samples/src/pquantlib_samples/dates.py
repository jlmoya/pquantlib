"""Dates sample — explores the :class:`~pquantlib.time.date.Date` interface.

Port of ``org.jquantlib.samples.Dates``. Walks through date construction,
month / weekday / day-of-month / day-of-year accessors, leap-year detection,
the ``next_weekday`` / ``nth_weekday`` / ``end_of_month`` class helpers, period
arithmetic, comparisons, and increment / decrement loops, printing each result.

PQuantLib's ``Date`` is immutable (value type), so the Java in-place ``inc()`` /
``dec()`` / ``addAssign`` mutations are expressed by rebinding to ``date + 1`` /
``date - 1`` — the printed results are identical.
"""

from __future__ import annotations

from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from pquantlib.time.weekday import Weekday
from pquantlib_samples.util.stop_clock import StopClock


def run() -> None:
    print("::::: Dates :::::")

    clock = StopClock()
    clock.start_clock()

    # Take today's date to explore the Date interface.
    today = Date.todays_date()
    print(f"Today's date is = {today}")

    # Month enum + its integer equivalent.
    month = today.month()
    print(f"The month of today's date is = {month.name}")
    print(f"The integer equivalent of this month as obtained from the date is = {int(month)}")
    print(f"The integer equivalent of the date as obtained from the Month is also = {int(month)}")

    # Weekday.
    weekday = today.weekday()
    print(f"The weekday of this date is = {weekday.name}")

    # Day of month (1-31) and day of year (1-366).
    print(f"The day of the date as a day in this date's month(1-31) is = {today.day_of_month()}")
    print(f"The day of the date as day in it's year(1-366) is = {today.day_of_year()}")

    # Leap-year check.
    if Date.is_leap(today.year()):
        print("Today's date belong to leap year")

    # Next Tuesday on/after today.
    next_weekday_date = Date.next_weekday(today, Weekday.Tuesday)
    print(f"The date of the next weekday is = {next_weekday_date}")

    # 4th Tuesday of today's month.
    fourth_weekday_date = Date.nth_weekday(4, Weekday.Tuesday, today.month(), today.year())
    print(f"The fourthWeekdayDate which is TUESDAY is = {fourth_weekday_date}")

    # First date of today's month, derived from end-of-month.
    date_end_of_month = Date.end_of_month(today)
    day_of_end_of_month = date_end_of_month.day_of_month()
    date_start_of_month = date_end_of_month + (-day_of_end_of_month + 1)
    print(f"The first date of the month to which todays date belong to is = {date_start_of_month}")

    # First date of today's month, derived via a Period.
    period = Period(-today.day_of_month() + 1, TimeUnit.Days)
    date_start_of_month_using_period = today + period
    print(
        "The first date of the month to which today's date belong to using period is = "
        f"{date_start_of_month_using_period}"
    )
    print(
        "The first date of the month to which today's date belong to using adjustment of period is = "
        f"{date_start_of_month_using_period}"
    )

    # Date comparisons.
    if date_start_of_month_using_period <= date_end_of_month:
        print("Start date is less than end date?")
    if date_end_of_month >= date_start_of_month_using_period:
        print("End date is greater than start date")

    # Increment today up to end-of-month.
    date = today
    while date != date_end_of_month:
        date = date + 1
    print(f"The date variable has been incremented to endOfMonth and is = {date}")

    # Decrement back to start-of-month.
    date = today
    while date != date_start_of_month:
        date = date - 1
    print(f"The date variable has been decremented to startOfMonth and is = {date}")

    # In-place-style update (immutable: rebind).
    today_plus = today + 1
    print(f"Today's date dateToday has been updated to = {today_plus}")
    today_back = today_plus - 1
    print(f"Today's date dateToday has been updated to = {today_back}")

    clock.stop_clock()
    clock.log()


if __name__ == "__main__":
    run()
