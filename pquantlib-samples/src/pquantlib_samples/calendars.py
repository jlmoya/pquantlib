"""Calendars sample — explores the :class:`~pquantlib.time.calendar.Calendar` API.

Port of ``org.jquantlib.samples.Calendars``. Builds the NYSE calendar, lists
holidays and business days over the next 90 days (cross-checking the
``business_days_between`` / ``is_business_day`` / ``is_holiday`` / ``holiday_list``
accessors against hand loops), exercises the business-day-convention variants of
``adjust`` and the ``advance`` overloads, and finally joins two calendars with
``JointCalendarRule.JoinBusinessDays`` and verifies its holiday list.

PQuantLib's ``Date`` is immutable, so the Java in-place ``inc()`` loops are
expressed by rebinding to ``date + 1`` — results are identical.
"""

from __future__ import annotations

from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.joint_calendar import JointCalendar, JointCalendarRule
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from pquantlib_samples.util.stop_clock import StopClock


def run() -> None:
    print("::::: Calendars :::::")

    clock = StopClock()
    clock.start_clock()

    # --- Basic calendar functions: NYSE ----------------------------------
    nyse = UnitedStates(UnitedStates.Market.NYSE)
    print(f"The name of this calendar is = {nyse.name()}")

    # Holidays from today out to today + 90 days.
    today = Date.todays_date()
    advanced = today + 90
    holidays = nyse.holiday_list(today, advanced, True)
    print(
        f"The holidays between dateToday = {today} till the date dateAdvanced = {advanced} are as shown below"
    )
    buffer = "".join(f"{d}::" for d in holidays)
    print(f"The holidays separated by :: are = {buffer}")

    # Business days via the is_business_day loop.
    business_via_is_bus_day = [d for d in _date_range(today, advanced) if nyse.is_business_day(d)]
    size_via_loop = len(business_via_is_bus_day)

    # Business days via the calendar API.
    size_via_api = nyse.business_days_between(today, advanced, True, True)
    if size_via_loop == size_via_api:
        print("The sizes are same")

    # Business days via the is_holiday loop.
    business_via_is_hol_day = [d for d in _date_range(today, advanced) if not nyse.is_holiday(d)]
    if business_via_is_bus_day == business_via_is_hol_day:
        print("The lists businessDaysInBetweenUsingIsBusDay and businessDaysInBetweenUsingIsHolDay are same")

    # --- Calendar arithmetic ---------------------------------------------
    # First holiday on/after today.
    first_holiday = today
    while not nyse.is_holiday(first_holiday):
        first_holiday = first_holiday + 1
    if nyse.is_holiday(first_holiday):
        print(f"FirstHolidayDate = {first_holiday} is a holiday date")

    # Next business day after that holiday.
    next_business_day = nyse.advance_period(first_holiday, Period(1, TimeUnit.Days))
    if nyse.is_business_day(next_business_day):
        print(f"NextBusinessDayFromFirstHolidayFromToday = {next_business_day} is a business date")

    # Adjust today under each business-day convention.
    _print_adjust_conventions(nyse, today)

    # advance(...) overloads.
    _print_advance_overloads(nyse, today)

    # --- Joining calendars -----------------------------------------------
    govt_bond = UnitedStates(UnitedStates.Market.GovernmentBond)
    joint = JointCalendar([nyse, govt_bond], JointCalendarRule.JoinBusinessDays)

    holidays_from_api = joint.holiday_list(today, advanced, True)
    holidays_from_loop = tuple(d for d in _date_range(today, advanced) if joint.is_holiday(d))
    if holidays_from_api == holidays_from_loop:
        print("Lists listOfHoliDays and holidayListObtainedUsingCalAPI of joint calendar are same")

    clock.stop_clock()
    clock.log()


def _print_adjust_conventions(nyse: UnitedStates, today: Date) -> None:
    """Adjust ``today`` under each business-day convention and print the result."""
    following = nyse.adjust(today, BusinessDayConvention.Following)
    if nyse.is_business_day(following):
        print(f"NextFollowingBusinessDate = {following} from today is a business date")

    mod_following = nyse.adjust(today, BusinessDayConvention.ModifiedFollowing)
    if nyse.is_business_day(mod_following):
        print(f"NextModified_FollowingBusinessDate = {mod_following} from today is a business date")

    preceding = nyse.adjust(today, BusinessDayConvention.Preceding)
    if nyse.is_business_day(preceding):
        print(f"NextPrecidingBusinessDay = {preceding} from today is a business date")

    mod_preceding = nyse.adjust(today, BusinessDayConvention.ModifiedPreceding)
    if nyse.is_business_day(mod_preceding):
        print(f"NextModified_PrecidingBusinessDay = {mod_preceding} from today is a business date")

    unadjusted = nyse.adjust(today, BusinessDayConvention.Unadjusted)
    # Java parity: Calendars.java:152 uses the modified-preceding day as the guard here; reproduced verbatim.
    if nyse.is_business_day(mod_preceding):
        print(f"NextUnadjustedBusinessDay = {unadjusted} from today is a business date and is same as today")


def _print_advance_overloads(nyse: UnitedStates, today: Date) -> None:
    """Exercise the ``advance(date, n, unit)`` / ``advance(date, period, conv)`` overloads."""
    print(
        "Next business date when today's date is advanced by 10 days = "
        f"{nyse.advance(today, 10, TimeUnit.Days)}"
    )
    print(
        "Next business date when today's date is advanced by 10 weeks = "
        f"{nyse.advance(today, 10, TimeUnit.Weeks)}"
    )
    print(
        "Next business date when today's date is advanced by 10 months = "
        f"{nyse.advance(today, 10, TimeUnit.Months)}"
    )
    print(
        "Next business date when today's date is advanced by 10 years = "
        f"{nyse.advance(today, 10, TimeUnit.Years)}"
    )
    print(
        "Next business date when today's date is advanced 1 day = "
        f"{nyse.advance_period(today, Period(1, TimeUnit.Days), BusinessDayConvention.Following)}"
    )
    print(
        "Next business date when today's date is advanced 1 week = "
        f"{nyse.advance_period(today, Period(1, TimeUnit.Weeks), BusinessDayConvention.ModifiedFollowing)}"
    )
    print(
        "Next business date when today's date is advanced 1 month = "
        f"{nyse.advance_period(today, Period(1, TimeUnit.Months), BusinessDayConvention.ModifiedPreceding)}"
    )
    print(
        "Next business date when today's date is advanced 1 year = "
        f"{nyse.advance_period(today, Period(1, TimeUnit.Years), BusinessDayConvention.Preceding)}"
    )


def _date_range(start: Date, stop: Date) -> list[Date]:
    """Dates in ``[start, stop)`` (half-open, mirrors the Java ``!eq(end)`` loop)."""
    out: list[Date] = []
    d = start
    while d != stop:
        out.append(d)
        d = d + 1
    return out


if __name__ == "__main__":
    run()
